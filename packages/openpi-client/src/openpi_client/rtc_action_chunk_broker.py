import math
import threading
import time
from typing import Dict, Optional

import numpy as np  # noqa: F401
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from openpi_client.action_queue import ActionQueue
from openpi_client.latency_tracker import LatencyTracker
from openpi_client.logger import get_logger

logger = get_logger("RTCActionChunkBroker")


class RTCActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return actions using an RTC-style ActionQueue.

    This broker runs a background thread to fetch action chunks from the policy
    and maintains a thread-safe queue of actions. It handles:
    - Asynchronous action fetching
    - Latency tracking (basic)
    - Action queue management (merging/replacing based on delay)

    Args:
        policy: The underlying policy (e.g., WebsocketClientPolicy).
        frequency_hz: The control frequency in Hz.
        action_queue_size_to_get_new_actions: Threshold to request new actions.
        rtc_enabled: Whether to enable RTC mode (replace queue) or append mode.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        frequency_hz: float = 50.0,
        action_queue_size_to_get_new_actions: int = 20,
        rtc_enabled: bool = True,
        execution_horizon: int = 20,
        blend_steps: int = 0,
        default_delay: int = 4,  # Default inference_delay for warmup and fallback
    ):
        self._policy = policy
        self._frequency_hz = frequency_hz
        self._time_per_chunk = 1.0 / frequency_hz
        self._action_queue_size_to_get_new_actions = action_queue_size_to_get_new_actions
        self._execution_horizon = execution_horizon
        self._default_delay = default_delay

        self._action_queue = ActionQueue(rtc_enabled=rtc_enabled, blend_steps=blend_steps)
        self._latency_tracker = LatencyTracker()
        self._latest_obs: Optional[Dict] = None
        self._latest_obs_lock = threading.Lock()

        # Track last real_delay to use as next estimated_delay
        self._last_real_delay: Optional[int] = None

        # Track warmup state: first inference is for JIT, second is for real execution
        self._warmup_done = False
        self._warmup_prev_chunk: Optional[np.ndarray] = None  # Store first inference result for second inference

        self._stop_event = threading.Event()
        self._first_inference_done = threading.Event()  # Signal when actions are ready for execution (after warmup)
        self._thread = threading.Thread(target=self._get_actions_loop, daemon=True)
        self._thread_started = False

    def _start_thread_if_needed(self):
        if not self._thread_started:
            self._thread.start()
            self._thread_started = True
            logger.info("RTCActionChunkBroker background thread started")

    def _get_actions_loop(self):
        while not self._stop_event.is_set():
            try:
                # Check if we need more actions
                if self._action_queue.qsize() <= self._action_queue_size_to_get_new_actions:
                    # Get latest observation
                    with self._latest_obs_lock:
                        obs = self._latest_obs

                    if obs is None:
                        # No observation yet, wait a bit
                        time.sleep(0.001)
                        continue

                    # Prepare for inference
                    current_time = time.perf_counter()
                    action_index_before_inference = self._action_queue.get_action_index()

                    # Determine prev_chunk_left_over and estimated_delay based on warmup state
                    if not self._warmup_done:
                        if self._warmup_prev_chunk is None:
                            # ===== WARMUP PHASE 1: First inference (JIT compilation) =====
                            # Send None, use default_delay, result will NOT be executed
                            prev_chunk_left_over = None
                            estimated_delay_steps = self._default_delay
                            logger.info(
                                f"🔥 WARMUP Phase 1: First inference (JIT). "
                                f"Sending prev_chunk_left_over=None, inference_delay={estimated_delay_steps}"
                            )
                        else:
                            # ===== WARMUP PHASE 2: Second inference (with prev_chunk from phase 1) =====
                            # Use last N actions from warmup result as prev_chunk_left_over
                            # This ensures consistent shape for JAX JIT
                            prev_chunk_left_over = self._warmup_prev_chunk[
                                -self._action_queue_size_to_get_new_actions :
                            ]
                            estimated_delay_steps = self._default_delay
                            logger.info(
                                f"🔥 WARMUP Phase 2: Second inference with prev_chunk. "
                                f"Sending prev_chunk_left_over shape: {prev_chunk_left_over.shape}, "
                                f"inference_delay={estimated_delay_steps}"
                            )
                    else:
                        # ===== NORMAL OPERATION =====
                        # Estimate inference delay using last real_delay if available
                        if self._last_real_delay is not None:
                            estimated_delay_steps = self._last_real_delay
                        else:
                            estimated_delay_steps = self._default_delay

                        # Get leftover actions for RTC guidance
                        prev_chunk_left_over = self._action_queue.get_left_over()
                        logger.info(
                            f"RTC: Starting inference. Queue size: {self._action_queue.qsize()}, "
                            f"estimated_delay={estimated_delay_steps}, "
                            f"prev_chunk shape: {prev_chunk_left_over.shape if prev_chunk_left_over is not None else None}"
                        )

                    results = self._policy.infer(
                        obs,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=estimated_delay_steps,
                        execution_horizon=self._execution_horizon,
                    )

                    # Calculate actual latency for next time
                    latency = time.perf_counter() - current_time
                    self._latency_tracker.add(latency)
                    inference_delay_steps = math.ceil(latency / self._time_per_chunk)

                    # Get actions
                    processed_actions = results.get("actions")
                    original_actions = results.get("actions_original")

                    if processed_actions is None:
                        logger.error("Policy returned no 'actions' key")
                        continue

                    if original_actions is None:
                        original_actions = processed_actions

                    # Handle warmup phases
                    if not self._warmup_done:
                        if self._warmup_prev_chunk is None:
                            # ===== WARMUP PHASE 1 COMPLETE =====
                            # Save original_actions for next inference, do NOT execute
                            self._warmup_prev_chunk = original_actions
                            logger.info(
                                f"✅ WARMUP Phase 1 complete. Latency: {latency * 1000:.0f}ms. "
                                f"Saved {original_actions.shape} for Phase 2. Actions NOT executed."
                            )
                            # Don't set _first_inference_done, continue to Phase 2
                            continue
                        else:
                            # ===== WARMUP PHASE 2 COMPLETE =====
                            # This inference has correct prev_chunk shape, execute these actions
                            self._warmup_done = True
                            # Use estimated_delay (default_delay) instead of actual latency for warmup
                            # because JIT compilation causes artificially high latency
                            inference_delay_steps = estimated_delay_steps
                            logger.info(
                                f"✅ WARMUP Phase 2 complete. Latency: {latency * 1000:.0f}ms (JIT). "
                                f"Using default_delay={estimated_delay_steps} for merge. Warmup done."
                            )
                            # Fall through to normal merge logic below

                    # Normal operation: merge actions into queue
                    # Skip CRITICAL check during warmup (Phase 2) since JIT causes high latency
                    if not self._warmup_done or inference_delay_steps < len(processed_actions):
                        pass  # OK
                    else:
                        logger.error(
                            f"RTC: CRITICAL - Inference delay ({inference_delay_steps} steps) "
                            f"exceeds action length ({len(processed_actions)} steps). "
                            "All actions will be discarded!"
                        )

                    merge_start = time.perf_counter()
                    real_delay_before_merge = self._action_queue.get_action_index() - action_index_before_inference
                    self._action_queue.merge(
                        new_original_actions=original_actions,
                        new_processed_actions=processed_actions,
                        estimated_delay=estimated_delay_steps,
                        action_index_before_inference=action_index_before_inference,
                    )
                    merge_ms = (time.perf_counter() - merge_start) * 1000
                    logger.info(f"RTC: Merge total time: {merge_ms:.2f}ms")

                    # Update last_real_delay for next inference
                    if real_delay_before_merge > 0:
                        self._last_real_delay = real_delay_before_merge
                    else:
                        # Only use time-based if it's reasonable (< 10 steps)
                        if inference_delay_steps <= 10:
                            self._last_real_delay = inference_delay_steps
                        else:
                            # JIT compilation case: use default
                            self._last_real_delay = 4
                            logger.info(
                                f"RTC: First inference took {inference_delay_steps} steps "
                                f"(likely JIT), using default delay=4 for next inference"
                            )

                    # Signal that first inference is done (queue now has actions)
                    if not self._first_inference_done.is_set():
                        logger.info("First inference completed, action queue initialized")
                        self._first_inference_done.set()
                else:
                    # Sleep to prevent busy waiting
                    time.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in RTC background thread: {e}")
                time.sleep(0.1)

    @override
    def infer(self, obs: Dict) -> Dict:
        self._start_thread_if_needed()

        # Update latest observation for the background thread
        with self._latest_obs_lock:
            self._latest_obs = obs

        # Wait for first inference to complete (handles JIT compilation delay)
        if not self._first_inference_done.is_set():
            logger.info("Waiting for first inference to complete (JIT compilation)...")
            # Wait up to 120 seconds for first inference (JIT can be slow)
            if not self._first_inference_done.wait(timeout=120.0):
                raise RuntimeError(
                    "RTCActionChunkBroker: Timeout waiting for first inference. "
                    "Check if the policy server is running and responsive."
                )
            logger.info("First inference done, proceeding with action queue")

        # Get action from queue
        action = self._action_queue.get()

        if action is None:
            # Queue empty after first inference - this shouldn't happen often
            # Wait briefly and retry
            logger.warn("Action queue empty! Waiting...")
            start_wait = time.time()
            while action is None and (time.time() - start_wait) < 5.0:
                time.sleep(0.005)
                action = self._action_queue.get()

            if action is None:
                logger.error("Action queue empty after waiting!")
                raise RuntimeError("RTCActionChunkBroker: Action queue is empty.")

        # Return in the format expected by the agent (dict)
        # The agent expects a dict that might contain other keys, but usually just "actions"
        # Since we only queue the action array, we reconstruct the dict.
        return {"actions": action}

    @override
    def reset(self) -> None:
        self._policy.reset()
        # Clear the action queue for next episode
        self._action_queue.clear()
        self._first_inference_done.clear()  # Reset for next episode
        self._last_real_delay = None  # Reset delay tracking
        # Reset warmup state for next episode
        self._warmup_done = False
        self._warmup_prev_chunk = None
        with self._latest_obs_lock:
            self._latest_obs = None

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
