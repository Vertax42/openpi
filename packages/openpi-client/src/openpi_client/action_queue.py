from threading import Lock
from typing import Optional

import numpy as np

from openpi_client.logger import get_logger

logger = get_logger("ActionQueue")


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control (NumPy version).

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity
    """

    def __init__(self, rtc_enabled: bool = True, blend_steps: int = 0):
        """Initialize the action queue.

        Args:
            rtc_enabled: Whether Real-Time Chunking is enabled.
            blend_steps: Number of steps to blend between old and new actions at merge.
                        0 = no blending (hard switch), >0 = linear blend over N steps.
        """
        self.queue: Optional[np.ndarray] = None  # Processed actions for robot rollout
        self.original_queue: Optional[np.ndarray] = None  # Original actions for RTC
        self.lock = Lock()
        self.last_index = 0
        self.rtc_enabled = rtc_enabled
        self.blend_steps = blend_steps

    def get(self) -> Optional[np.ndarray]:
        """Get the next action from the queue.

        Returns:
            np.ndarray | None: The next action (action_dim,) or None if queue is empty.
                              Returns a copy to prevent external modifications.
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                logger.warn(
                    "Action queue exhausted! No actions available. "
                    "This may cause robot to stall. Consider increasing execution_horizon "
                    "or reducing action_queue_size_to_get_new_actions."
                )
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.copy()

    def qsize(self) -> int:
        """Get the number of remaining actions in the queue.

        Returns:
            int: Number of unconsumed actions.
        """
        if self.queue is None:
            return 0
        length = len(self.queue)
        return length - self.last_index

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if no actions remain, False otherwise.
        """
        if self.queue is None:
            return True

        length = len(self.queue)
        return length - self.last_index <= 0

    def get_action_index(self) -> int:
        """Get the current action consumption index.

        Returns:
            int: Index of the next action to be consumed.
        """
        return self.last_index

    def clear(self) -> None:
        """Clear the queue, removing all actions.

        This resets the queue to its initial empty state while preserving
        the rtc_enabled setting.
        """
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.last_index = 0

    def get_left_over(self) -> Optional[np.ndarray]:
        """Get leftover original actions for RTC prev_chunk_left_over.

        These are the unconsumed actions from the current chunk, which will be
        used by RTC to compute corrections for the next chunk.

        Returns:
            np.ndarray | None: Remaining original actions (remaining_steps, action_dim),
                              or None if no original queue exists.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :]

    def merge(
        self,
        new_original_actions: np.ndarray,
        new_processed_actions: np.ndarray,
        estimated_delay: int,
        action_index_before_inference: Optional[int] = 0,
    ):
        """Merge new actions into the queue.

        Args:
            new_original_actions: Unprocessed actions from policy (time_steps, action_dim).
            new_processed_actions: Post-processed actions for robot (time_steps, action_dim).
            estimated_delay: estimated delay steps passed to model.
            action_index_before_inference: Index before inference started.

        Note:
            We use real_delay for truncation to avoid skipping actions:
            - real_delay = actual steps consumed during inference
            - Truncating at real_delay ensures no action is skipped
            - RTC guidance ensures new_actions[real_delay] is smooth with old trajectory
        """
        real_delay = None
        truncate_delay = None
        with self.lock:
            if self.rtc_enabled:
                real_delay = self.get_action_index() - action_index_before_inference
                # Use real_delay for truncation to avoid skipping actions
                # RTC guidance ensures actions near real_delay are smooth
                truncate_delay = real_delay
                self._replace_actions_queue(
                    new_original_actions,
                    new_processed_actions,
                    truncate_delay,
                )
            else:
                self._append_actions_queue(new_original_actions, new_processed_actions)

        # Log outside lock to avoid blocking get() calls
        if real_delay is not None:
            logger.info(f"RTC: Truncate at {truncate_delay}, " f"estimated={estimated_delay}, real={real_delay}")

    def _replace_actions_queue(
        self,
        new_original_actions: np.ndarray,
        new_processed_actions: np.ndarray,
        truncate_delay: int,
    ):
        """Replace the queue with new actions (RTC mode).

        Args:
            truncate_delay: The delay used to truncate actions (should be estimated_delay
                           that was passed to the model, ensuring consistency).
        """
        truncate_idx = max(0, min(truncate_delay, len(new_original_actions)))

        # Debug: Verify RTC alignment in the delay region
        if self.original_queue is not None and truncate_idx > 0:
            # RTC should align: new_actions[0:delay] ≈ prev_left_over[0:delay]
            # prev_left_over was self.original_queue[self.last_index:] at inference start
            # So we compare new_actions[0:delay] with old_queue[last_index : last_index+delay]
            remaining_old = len(self.original_queue) - self.last_index
            align_len = min(truncate_idx, remaining_old, len(new_original_actions))
            if align_len > 0:
                # This is what was passed to model as prev_chunk_left_over
                old_aligned = self.original_queue[self.last_index : self.last_index + align_len]
                new_aligned = new_original_actions[:align_len]
                diff_rtc = np.abs(new_aligned - old_aligned)
                max_diff_rtc = np.max(diff_rtc)
                mean_diff_rtc = np.mean(diff_rtc)
                logger.info(
                    f"RTC Alignment Check: new[0:{align_len}] vs old[{self.last_index}:{self.last_index + align_len}] "
                    f"max={max_diff_rtc:.4f}, mean={mean_diff_rtc:.4f} "
                    f"(should be small if RTC works correctly)"
                )

        # Debug: Check actual jump at merge point
        if self.queue is not None and truncate_idx < len(new_processed_actions):
            next_action = new_processed_actions[truncate_idx]
            # Compare with last executed action (the actual jump)
            last_executed_idx = self.last_index - 1
            if last_executed_idx >= 0 and last_executed_idx < len(self.queue):
                last_action = self.queue[last_executed_idx]
                diff_jump = np.abs(next_action - last_action)
                max_diff_jump = np.max(diff_jump)
                mean_diff_jump = np.mean(diff_jump)
                max_diff_dim = np.argmax(diff_jump)
                logger.info(
                    f"RTC Actual Jump: new[{truncate_idx}] vs old[{last_executed_idx}] "
                    f"max={max_diff_jump:.4f} (dim {max_diff_dim}), mean={mean_diff_jump:.4f}"
                )

        # Get the new actions starting from truncate_idx
        new_original = new_original_actions[truncate_idx:].copy()
        new_processed = new_processed_actions[truncate_idx:].copy()

        # Apply blending if enabled and we have old actions
        if self.blend_steps > 0 and self.queue is not None and self.last_index < len(self.queue):
            # Get remaining old actions
            old_remaining = self.queue[self.last_index :]
            logger.debug(f"old_remaining steps: {len(old_remaining)}")
            blend_len = min(self.blend_steps, len(old_remaining), len(new_processed))

            if blend_len > 0:
                # Linear blend: alpha goes from 0 to 1 over blend_steps
                for i in range(blend_len):
                    alpha = (i + 1) / (blend_len + 1)  # 0 < alpha < 1
                    new_processed[i] = (1 - alpha) * old_remaining[i] + alpha * new_processed[i]
                    new_original[i] = (1 - alpha) * self.original_queue[self.last_index + i] + alpha * new_original[i]

        self.original_queue = new_original
        self.queue = new_processed
        self.last_index = 0

    def _append_actions_queue(self, new_original_actions: np.ndarray, new_processed_actions: np.ndarray):
        """Append new actions to the queue (non-RTC mode)."""
        if self.queue is None:
            self.original_queue = new_original_actions.copy()
            self.queue = new_processed_actions.copy()
            return

        # Remove consumed actions
        self.original_queue = self.original_queue[self.last_index :]
        self.queue = self.queue[self.last_index :]

        # Append new actions
        self.original_queue = np.concatenate([self.original_queue, new_original_actions])
        self.queue = np.concatenate([self.queue, new_processed_actions])

        self.last_index = 0
