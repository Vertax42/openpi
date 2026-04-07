import math
import threading
import time
from typing import Dict, Optional

import numpy as np
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client.action_queue import ActionQueue
from openpi_client.latency_tracker import LatencyTracker
from lerobot.utils.robot_utils import get_logger

logger = get_logger("RTCActionChunkBroker")


class RTCActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy for training-time RTC inference.

    Runs a background thread that fetches action chunks from the policy and
    maintains a thread-safe queue of actions.  Before each inference call the
    broker builds ``action_prefix`` (padded to ``action_horizon``) and
    ``inference_delay`` so the model receives clean, ready-to-use inputs.

    Args:
        policy: Underlying policy (e.g. WebsocketClientPolicy).
        frequency_hz: Control loop frequency in Hz.
        action_horizon: Model's prediction horizon H (must match model config).
        action_dim: Model's action dimension (must match model config).
        request_threshold: Request new chunk when remaining actions <= this value.
        default_delay: Fallback inference_delay used for the first real inference.
        dry_run: If True, infer() includes ``rtc_metrics`` for debugging.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        frequency_hz: float = 50.0,
        action_horizon: int = 50,
        action_dim: int = 32,
        request_threshold: int = 20,
        default_delay: int = 4,
        dry_run: bool = False,
    ):
        self._policy = policy
        self._frequency_hz = frequency_hz
        self._time_per_step = 1.0 / frequency_hz
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._request_threshold = request_threshold
        self._default_delay = default_delay
        self._dry_run = dry_run

        self._action_queue = ActionQueue()
        self._latency_tracker = LatencyTracker()
        self._last_real_delay: Optional[int] = None
        self._jit_done = False
        self._inference_seq = 0

        self._latest_obs: Optional[Dict] = None
        self._latest_obs_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._last_inference_metrics: Optional[Dict] = None

        self._stop_event = threading.Event()
        self._first_inference_done = threading.Event()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread_started = False

    # ------------------------------------------------------------------
    # Action prefix preparation (paper Figure 1)
    # ------------------------------------------------------------------

    def _prepare_action_prefix(self) -> tuple[np.ndarray, int]:
        """Build ``action_prefix`` (H, ad) and ``estimated_delay`` from queue state.

        The prefix is zero-padded to ``action_horizon``; only the first
        ``estimated_delay`` positions carry valid data from the previous
        chunk's remaining actions.
        """
        remaining = self._action_queue.get_remaining_original()
        prefix = np.zeros((self._action_horizon, self._action_dim), dtype=np.float32)

        if remaining is None:
            # First inference – no previous chunk yet
            return prefix, 0

        n = min(len(remaining), self._action_horizon)
        prefix[:n] = remaining[:n]

        estimated_delay = self._last_real_delay if self._last_real_delay is not None else self._default_delay
        # Clamp to valid range
        estimated_delay = max(0, min(estimated_delay, n))
        return prefix, estimated_delay

    # ------------------------------------------------------------------
    # Background inference loop
    # ------------------------------------------------------------------

    def _start_thread_if_needed(self):
        if not self._thread_started:
            self._thread.start()
            self._thread_started = True
            logger.info("RTCActionChunkBroker background thread started")

    def _inference_loop(self):
        while not self._stop_event.is_set():
            try:
                if self._action_queue.qsize() > self._request_threshold:
                    time.sleep(0.001)
                    continue

                with self._latest_obs_lock:
                    obs = self._latest_obs
                if obs is None:
                    time.sleep(0.001)
                    continue

                # JIT warmup: first call with delay=0 triggers compilation,
                # result is used normally (just without prefix constraint).
                action_prefix, est_delay = self._prepare_action_prefix()
                if not self._jit_done:
                    est_delay = 0
                    logger.info("JIT warmup inference (delay=0)")

                cursor_before = self._action_queue.get_cursor()
                t0 = time.perf_counter()

                results = self._policy.infer(
                    obs,
                    action_prefix=action_prefix,
                    inference_delay=est_delay,
                )

                latency = time.perf_counter() - t0
                self._latency_tracker.add(latency)
                latency_steps = math.ceil(latency / self._time_per_step)

                original_actions = results.get("actions_original")
                processed_actions = results.get("actions")
                if processed_actions is None:
                    logger.error("Policy returned no 'actions'")
                    continue
                if original_actions is None:
                    original_actions = processed_actions

                # Compute real delay: how many actions the queue consumed during inference
                real_delay = self._action_queue.get_cursor() - cursor_before
                if not self._jit_done:
                    # JIT compilation inflates latency; use 0 for first merge
                    real_delay = 0
                    self._jit_done = True
                    logger.info(f"JIT warmup done. Latency: {latency * 1000:.0f}ms")

                # Replace queue with new chunk, skip already-consumed prefix
                self._action_queue.replace(original_actions, processed_actions, start_from=real_delay)
                self._last_real_delay = max(real_delay, 1)

                if self._dry_run:
                    self._inference_seq += 1
                    with self._metrics_lock:
                        self._last_inference_metrics = {
                            "inference_seq": self._inference_seq,
                            "infer_round_trip_ms": latency * 1000.0,
                            "latency_steps": latency_steps,
                            "estimated_delay": est_delay,
                            "real_delay": real_delay,
                            "queue_size_after": self._action_queue.qsize(),
                            "latency_p95_ms": (self._latency_tracker.p95() or 0.0) * 1000.0,
                        }

                if not self._first_inference_done.is_set():
                    logger.info("First inference complete, action queue ready")
                    self._first_inference_done.set()

            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @override
    def infer(self, obs: Dict) -> Dict:
        self._start_thread_if_needed()

        with self._latest_obs_lock:
            self._latest_obs = obs

        if not self._first_inference_done.is_set():
            logger.info("Waiting for first inference (JIT compilation)...")
            if not self._first_inference_done.wait(timeout=120.0):
                raise RuntimeError("Timeout waiting for first inference")
            logger.info("First inference done")

        action = self._action_queue.get()
        if action is None:
            start = time.time()
            while action is None and (time.time() - start) < 5.0:
                time.sleep(0.005)
                action = self._action_queue.get()
            if action is None:
                raise RuntimeError("Action queue empty after waiting")

        out: Dict = {"actions": action}
        if self._dry_run:
            with self._metrics_lock:
                snap = dict(self._last_inference_metrics) if self._last_inference_metrics else None
            if snap is not None:
                out["rtc_metrics"] = snap
        return out

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._action_queue.clear()
        self._first_inference_done.clear()
        self._last_real_delay = None
        self._jit_done = False
        self._inference_seq = 0
        with self._metrics_lock:
            self._last_inference_metrics = None
        with self._latest_obs_lock:
            self._latest_obs = None

    @override
    def warmup(self, obs: Dict) -> None:
        """Pre-warm JIT before the control loop.

        Blocks until the first inference completes so the first real
        control step has no JIT delay.
        """
        logger.info("Pre-episode warmup...")
        self._start_thread_if_needed()
        with self._latest_obs_lock:
            self._latest_obs = obs
        if not self._first_inference_done.wait(timeout=120.0):
            raise RuntimeError("Warmup timed out")
        logger.info("Warmup complete")

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
