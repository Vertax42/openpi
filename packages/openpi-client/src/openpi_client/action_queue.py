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
            real_delay: real delay steps based on current action queue index.
            estimated_delay: estimated delay steps based on history max inference latency.
            action_index_before_inference: Index before inference started, for validation.
        """
        with self.lock:
            # self._check_delays(estimated_delay, action_index_before_inference)

            if self.rtc_enabled:
                real_delay = self.get_action_index() - action_index_before_inference
                logger.info(
                    f"RTC: Real delay before merge: {real_delay} | "
                    f"Estimated delay: {estimated_delay}"
                )

                self._replace_actions_queue(
                    new_original_actions,
                    new_processed_actions,
                    estimated_delay,
                    real_delay,
                )
                return

            self._append_actions_queue(new_original_actions, new_processed_actions)

    def _replace_actions_queue(
        self,
        new_original_actions: np.ndarray,
        new_processed_actions: np.ndarray,
        estimated_delay: int,
        real_delay: int,
    ):
        """Replace the queue with new actions (RTC mode)."""

        # Determine start index for new actions based on estimated delay
        # The model was conditioned on 'estimated_delay', so the new trajectory
        # is optimized to start smoothly from 'estimated_delay' relative to the
        # previous snapshot.
        start_idx = estimated_delay

        # Ensure start_idx doesn't exceed action length
        start_idx = min(start_idx, len(new_original_actions))

        # Slice the new actions
        new_original_sliced = new_original_actions[start_idx:].copy()
        new_processed_sliced = new_processed_actions[start_idx:].copy()

        # Calculate the gap we need to fill using old actions
        # We are currently at 'real_delay' steps past the snapshot.
        # The new actions start at 'estimated_delay' steps past the snapshot.
        # We need to bridge the gap from 'real_delay' to 'estimated_delay'.
        gap = start_idx - real_delay

        if gap > 0 and self.queue is not None:
            # We need to continue executing old actions for 'gap' more steps.
            # These correspond to the next 'gap' items in the current queue.

            # Check if we have enough old actions remaining
            remaining_len = len(self.queue) - self.last_index
            take_len = min(gap, remaining_len)

            # Slice remaining old actions
            fill_processed = self.queue[self.last_index : self.last_index + take_len]

            if self.original_queue is not None:
                fill_original = self.original_queue[
                    self.last_index : self.last_index + take_len
                ]
            else:
                # Fallback if original queue missing (shouldn't happen usually)
                fill_original = fill_processed

            # Prepend the old actions to the new sliced actions
            new_processed_final = np.concatenate([fill_processed, new_processed_sliced])
            new_original_final = np.concatenate([fill_original, new_original_sliced])

            # If gap > remaining_len (underflow), we just run out of old actions and jump to new.
            # This is unavoidable if we ran out of buffer.
        elif gap < 0:
            # Case: real_delay > estimated_delay (Underestimation)
            # We are already past the point where new actions were supposed to start.
            # We must skip ahead in the new actions to catch up to 'real_delay'.
            # The slice start should be 'real_delay'.

            catchup_idx = real_delay
            catchup_idx = min(catchup_idx, len(new_original_actions))

            new_original_final = new_original_actions[catchup_idx:].copy()
            new_processed_final = new_processed_actions[catchup_idx:].copy()
        else:
            # Case: real_delay == estimated_delay (Perfect match)
            new_original_final = new_original_sliced
            new_processed_final = new_processed_sliced

        self.original_queue = new_original_final
        self.queue = new_processed_final

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

    def _check_delays(
        self, estimated_delay: int, action_index_before_inference: Optional[int] = None
    ):
        """Validate that computed delays match expectations."""
        if action_index_before_inference is None:
            return

        real_delay = self.last_index - action_index_before_inference
        if real_delay != estimated_delay:
            logger.warning(
                f"[ACTION_QUEUE] Real delay is not equal to estimated delay. "
                f"Real delay: {real_delay}, estimated delay: {estimated_delay}"
            )
