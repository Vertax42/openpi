from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class DebugStepJax:
    """Container for debug information from a single denoising step (JAX version)."""

    step_idx: int = 0
    x_t: jax.Array | None = None
    v_t: jax.Array | None = None
    x1_t: jax.Array | None = None
    correction: jax.Array | None = None
    err: jax.Array | None = None
    weights: jax.Array | None = None
    guidance_weight: float | jax.Array | None = None
    time: float | jax.Array | None = None
    inference_delay: int | None = None
    execution_horizon: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_tensors: bool = False) -> dict[str, Any]:
        """Convert debug step to dictionary."""

        def get_scalar(val):
            if isinstance(val, (jax.Array, np.ndarray)):
                return float(val)
            return val

        result = {
            "step_idx": self.step_idx,
            "guidance_weight": get_scalar(self.guidance_weight),
            "time": get_scalar(self.time),
            "inference_delay": self.inference_delay,
            "execution_horizon": self.execution_horizon,
            "metadata": self.metadata.copy(),
        }

        # Add tensor information
        tensor_fields = ["x_t", "v_t", "x1_t", "correction", "err", "weights"]
        for field_name in tensor_fields:
            tensor = getattr(self, field_name)
            if tensor is not None:
                # Convert to numpy array for inspection/serialization
                np_tensor = np.array(tensor)
                if include_tensors:
                    result[field_name] = np_tensor
                else:
                    result[f"{field_name}_stats"] = {
                        "shape": tuple(np_tensor.shape),
                        "mean": float(np_tensor.mean()),
                        "std": float(np_tensor.std()),
                        "min": float(np_tensor.min()),
                        "max": float(np_tensor.max()),
                    }

        return result


class TrackerJax:
    """Collects and manages debug information for RTC processing (JAX version)."""

    def __init__(self, enabled: bool = False, maxlen: int = 100):
        self.enabled = enabled
        self._steps = {} if enabled else None
        self._maxlen = maxlen
        self._step_counter = 0

    def reset(self) -> None:
        if self.enabled and self._steps is not None:
            self._steps.clear()
        self._step_counter = 0

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
        if not self.enabled:
            return

        # In JAX, inside compiled functions (jit), we cannot easily use side effects like this
        # if the tracker object is external.
        # However, for debugging purposes, usually we use host_callback or assume non-jitted context
        # or we rely on the fact that this runs eagerly.
        # Given the context, we'll assume we can use np.array() to detach/convert.

        # Convert time to float and round
        time_val = float(np.array(time))
        time_key = round(time_val, 6)

        if time_key in self._steps:
            step = self._steps[time_key]
            # Update fields if provided
            if x_t is not None:
                step.x_t = x_t
            if v_t is not None:
                step.v_t = v_t
            if x1_t is not None:
                step.x1_t = x1_t
            if correction is not None:
                step.correction = correction
            if err is not None:
                step.err = err
            if weights is not None:
                step.weights = weights
            if guidance_weight is not None:
                step.guidance_weight = guidance_weight
            if inference_delay is not None:
                step.inference_delay = inference_delay
            if execution_horizon is not None:
                step.execution_horizon = execution_horizon
            if metadata:
                step.metadata.update(metadata)
        else:
            step = DebugStepJax(
                step_idx=self._step_counter,
                x_t=x_t,
                v_t=v_t,
                x1_t=x1_t,
                correction=correction,
                err=err,
                weights=weights,
                guidance_weight=guidance_weight,
                time=time_val,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                metadata=metadata,
            )
            self._steps[time_key] = step
            self._step_counter += 1

            if self._maxlen is not None and len(self._steps) > self._maxlen:
                oldest_key = next(iter(self._steps))
                del self._steps[oldest_key]

    def get_all_steps(self) -> list[DebugStepJax]:
        if not self.enabled or self._steps is None:
            return []
        return list(self._steps.values())
