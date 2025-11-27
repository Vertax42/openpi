import logging
from threading import Lock
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control (NumPy version).

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity
    """

    def __init__(self, rtc_enabled: bool = True):
        """Initialize the action queue.

        Args:
            rtc_enabled: Whether Real-Time Chunking is enabled.
        """
        self.queue: Optional[np.ndarray] = None  # Processed actions for robot rollout
        self.original_queue: Optional[np.ndarray] = None  # Original actions for RTC
        self.lock = Lock()
        self.last_index = 0
        self.rtc_enabled = rtc_enabled

    def get(self) -> Optional[np.ndarray]:
        """Get the next action from the queue.

        Returns:
            np.ndarray | None: The next action (action_dim,) or None if queue is empty.
                              Returns a copy to prevent external modifications.
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
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
            logger.info(
                f"RTC: Truncate at {truncate_delay}, "
                f"estimated={estimated_delay}, real={real_delay}"
            )

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

        # Debug: Check action continuity at merge point
        if self.queue is not None and self.last_index > 0:
            # The action we're about to execute from new queue
            if truncate_idx < len(new_processed_actions):
                next_action = new_processed_actions[truncate_idx]

                # Compare with: the action at same position in OLD queue
                # (this is what RTC guidance aligns to)
                if self.last_index < len(self.queue):
                    old_action_at_same_pos = self.queue[self.last_index]
                    diff_aligned = np.abs(next_action - old_action_at_same_pos)
                    max_diff_aligned = np.max(diff_aligned)
                    mean_diff_aligned = np.mean(diff_aligned)
                    max_diff_dim = np.argmax(diff_aligned)
                    logger.info(
                        f"RTC Merge: diff(new[{truncate_idx}] vs old[{self.last_index}]) "
                        f"max={max_diff_aligned:.4f} (dim {max_diff_dim}), "
                        f"mean={mean_diff_aligned:.4f}"
                    )

                # Also compare with last executed action (for reference)
                last_executed_idx = self.last_index - 1
                if last_executed_idx >= 0 and last_executed_idx < len(self.queue):
                    last_action = self.queue[last_executed_idx]
                    diff_prev = np.abs(next_action - last_action)
                    max_diff_prev = np.max(diff_prev)
                    logger.info(
                        f"RTC Merge: diff(new[{truncate_idx}] vs old[{last_executed_idx}]) "
                        f"max={max_diff_prev:.4f} (this is the actual jump)"
                    )

        self.original_queue = new_original_actions[truncate_idx:].copy()
        self.queue = new_processed_actions[truncate_idx:].copy()
        self.last_index = 0

    def _append_actions_queue(
        self, new_original_actions: np.ndarray, new_processed_actions: np.ndarray
    ):
        """Append new actions to the queue (non-RTC mode)."""
        if self.queue is None:
            self.original_queue = new_original_actions.copy()
            self.queue = new_processed_actions.copy()
            return

        # Remove consumed actions
        self.original_queue = self.original_queue[self.last_index :]
        self.queue = self.queue[self.last_index :]

        # Append new actions
        self.original_queue = np.concatenate(
            [self.original_queue, new_original_actions]
        )
        self.queue = np.concatenate([self.queue, new_processed_actions])

        self.last_index = 0
