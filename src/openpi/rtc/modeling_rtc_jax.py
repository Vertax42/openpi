import math
from typing import Callable

import jax
import jax.numpy as jnp
from openpi.rtc.configuration_rtc import RTCAttentionSchedule
from openpi.rtc.configuration_rtc import RTCConfig
from openpi.rtc.debug_tracker_jax import TrackerJax


class RTCProcessorJax:
    """JAX implementation of Real-Time Chunking processor."""

    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config
        self.tracker = None
        if rtc_config.debug:
            self.tracker = TrackerJax(
                enabled=rtc_config.debug,
                maxlen=rtc_config.debug_maxlen,
            )

    # ====================== Tracker Proxy Methods ======================
    def track(
        self,
        time: float | jax.Array,
        x_t: jax.Array | None = None,
        v_t: jax.Array | None = None,
        x1_t: jax.Array | None = None,
        correction: jax.Array | None = None,
        err: jax.Array | None = None,
        weights: jax.Array | None = None,
        guidance_weight: float | jax.Array | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        **metadata,
    ) -> None:
        """Proxy method to track debug information.

        Handles JAX JIT compatibility using jax.debug.callback if tracker is enabled.
        """
        if self.tracker is None or not self.tracker.enabled:
            return

        # Helper function to be called from host
        def _track_callback(
            time, x_t, v_t, x1_t, correction, err, weights, guidance_weight
        ):
            # Since we are in callback, inputs are numpy arrays
            self.tracker.track(
                time=time,
                x_t=x_t,
                v_t=v_t,
                x1_t=x1_t,
                correction=correction,
                err=err,
                weights=weights,
                guidance_weight=guidance_weight,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                **metadata,
            )

        # Define inputs for callback (handle None)
        # JAX callbacks require valid arrays. We use dummy if None.
        def _ensure_array(x):
            return x if x is not None else jnp.array(0.0)  # Dummy

        jax.debug.callback(
            _track_callback,
            time,
            _ensure_array(x_t),
            _ensure_array(v_t),
            _ensure_array(x1_t),
            _ensure_array(correction),
            _ensure_array(err),
            _ensure_array(weights),
            _ensure_array(guidance_weight),
        )

    def get_all_debug_steps(self) -> list:
        if self.tracker is not None:
            return self.tracker.get_all_steps()
        return []

    def is_debug_enabled(self) -> bool:
        return self.tracker is not None and self.tracker.enabled

    def reset_tracker(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()

    # ====================== End Tracker Proxy Methods ======================

    def get_prefix_weights(
        self, start: int | jax.Array, end: int | jax.Array, total: int
    ) -> jax.Array:
        """Generates prefix weights for RTC guidance."""
        # Ensure start is not greater than end
        start = jnp.minimum(start, end)

        idx = jnp.arange(total)

        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ZEROS:
            weights = jnp.zeros(total)
            weights = jnp.where(idx < start, 1.0, weights)
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ONES:
            weights = jnp.ones(total)
            weights = jnp.where(idx >= end, 0.0, weights)
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR:
            # Region 1: idx < start -> 1.0
            # Region 2: start <= idx < end -> linear decay from 1.0 to 0.0
            # Region 3: idx >= end -> 0.0

            denom = jnp.maximum(end - start + 1, 1e-6)
            # Linear ramp: 1.0 at start, >0 at end
            # Matched to torch.linspace(1.0, 0.0, steps + 2)[1:-1] logic
            # steps = end - start
            # torch logic produces points between 1 and 0.
            # Let's approximate: 1.0 - (idx - start + 1) / (end - start + 1)
            # If idx = start, val = 1 - 1/denom.
            # Let's use simple linear interpolation: 1 - (idx - start) / (end - start)

            # Original logic:
            # lin_weights = jnp.linspace(1.0, 0.0, linspace_steps + 2)[1:-1]
            # This implies weights strictly between 0 and 1 exclusive? No.

            # Simplified logic compatible with JAX tracers:
            denom = jnp.maximum(end - start, 1e-6)
            linear_decay = 1.0 - (idx - start) / denom

            weights = jnp.where(idx < start, 1.0, linear_decay)
            weights = jnp.where(idx >= end, 0.0, weights)

        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.EXP:
            denom = jnp.maximum(end - start, 1e-6)
            linear_decay = 1.0 - (idx - start) / denom
            linear_decay = jnp.clip(linear_decay, 0.0, 1.0)

            # Apply exp transformation
            # lin_weights * (exp(lin_weights) - 1) / (e - 1)
            exp_weights = linear_decay * jnp.expm1(linear_decay) / (math.e - 1)

            weights = jnp.where(idx < start, 1.0, exp_weights)
            weights = jnp.where(idx >= end, 0.0, weights)
        else:
            # Default fallback
            weights = jnp.zeros(total)

        return weights

    def compute_guidance(
        self,
        x_t: jax.Array,
        time: float | jax.Array,
        prev_chunk_left_over: jax.Array | None,
        inference_delay: int | None,
        execution_horizon: int | None,
        model_fn: Callable[[jax.Array], jax.Array],
    ) -> jax.Array:
        """
        Computes the guided velocity v_t using RTC.
        """
        # Invert time (PyTorch logic adaptation)
        tau = 1.0 - time

        if prev_chunk_left_over is None:
            # Should be handled by caller, but for safety
            v_t = model_fn(x_t)
            return v_t

        # Handle 2D inputs (add batch dimension)
        squeezed = False
        if x_t.ndim < 3:
            x_t = jnp.expand_dims(x_t, axis=0)
            squeezed = True

        if prev_chunk_left_over.ndim < 3:
            prev_chunk_left_over = jnp.expand_dims(prev_chunk_left_over, axis=0)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon

        batch_size, action_chunk_size, action_dim = x_t.shape

        # Truncate execution horizon if needed
        # Use jnp.minimum to support JAX Tracers (e.g. inside JIT)
        execution_horizon = jnp.minimum(
            execution_horizon, prev_chunk_left_over.shape[1]
        )

        # Pad prev_chunk_left_over if needed
        if (
            prev_chunk_left_over.shape[1] < action_chunk_size
            or prev_chunk_left_over.shape[2] < action_dim
        ):
            padded = jnp.zeros(
                (batch_size, action_chunk_size, action_dim),
                dtype=prev_chunk_left_over.dtype,
            )
            padded = padded.at[
                :, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]
            ].set(prev_chunk_left_over)
            prev_chunk_left_over = padded

        # Note: assert shapes match is tricky in JAX tracing, we skip explicit assert
        # or rely on shape incompatibility errors later.

        # Compute weights
        # weights: (seq,) -> (1, seq, 1) for broadcasting
        weights_1d = self.get_prefix_weights(
            inference_delay, execution_horizon, action_chunk_size
        )
        weights = weights_1d[None, :, None]  # (1, T, 1)

        # Forward pass to get x1 and v
        def forward_fn(x):
            v = model_fn(x)
            x1 = x - time * v
            return x1, v

        (x1_val, v_val), vjp_fn = jax.vjp(forward_fn, x_t)

        err = (prev_chunk_left_over - x1_val) * weights

        # correction = vjp_fn(err)[0] corresponds to: err^T * Jacobian(x1_wrt_x)
        # We pass zeros for v because v is an auxiliary output for this gradient calculation
        # JAX requires gradients to match the output structure of forward_fn (x1, v)
        correction = vjp_fn((err, jnp.zeros_like(v_val)))[0]

        # Compute guidance weight
        max_guidance_weight = self.rtc_config.max_guidance_weight

        squared_one_minus_tau = (1 - tau) ** 2
        inv_r2 = (squared_one_minus_tau + tau**2) / squared_one_minus_tau

        # Avoid div by zero if tau is 0
        c = jnp.where(tau > 1e-6, (1 - tau) / tau, max_guidance_weight)

        guidance_weight = c * inv_r2
        guidance_weight = jnp.minimum(guidance_weight, max_guidance_weight)

        # Final result
        v_t_guided = v_val - guidance_weight * correction

        # Remove batch dimension if added
        if squeezed:
            v_t_guided = jnp.squeeze(v_t_guided, axis=0)
            x1_val = jnp.squeeze(x1_val, axis=0)
            correction = jnp.squeeze(correction, axis=0)
            err = jnp.squeeze(err, axis=0)
            # weights was broadcasted to (1, T, 1), maybe we want to squeeze it too?
            # PyTorch version: weights was (1, T, 1). It is NOT squeezed in PyTorch code provided?
            # Let's check PyTorch code again.
            # PyTorch:
            #   if squeezed: result = result.squeeze(0); correction=...; x1_t=...; err=...
            #   self.track(...)
            # It squeezes result, correction, x1_t, err. It does NOT squeeze weights explicitly in the snippet provided?
            # Wait, in PyTorch snippet:
            #   weights = ... .unsqueeze(0).unsqueeze(-1)
            # It is (1, T, 1).
            # The PyTorch track call receives `weights`.
            # The PyTorch squeeze block does NOT list weights.
            # So weights remains (1, T, 1).

        # Track debug info
        self.track(
            time=time,
            x1_t=x1_val,  # Note: track original batch/seq
            correction=correction,
            err=err,
            weights=weights,
            guidance_weight=guidance_weight,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )

        return v_t_guided
