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
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
        action_index_before_inference: Optional[int] = 0,
    ):
        """Merge new actions into the queue.

        Args:
            original_actions: Unprocessed actions from policy (time_steps, action_dim).
            processed_actions: Post-processed actions for robot (time_steps, action_dim).
            real_delay: Number of time steps of inference delay.
            action_index_before_inference: Index before inference started, for validation.
        """
        with self.lock:
            self._check_delays(real_delay, action_index_before_inference)

            if self.rtc_enabled:
                self._replace_actions_queue(
                    original_actions, processed_actions, real_delay
                )
                return

            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(
        self,
        original_actions: np.ndarray,
        processed_actions: np.ndarray,
        real_delay: int,
    ):
        """Replace the queue with new actions (RTC mode)."""
        # Ensure delay doesn't exceed action length
        start_idx = min(real_delay, len(original_actions))

        self.original_queue = original_actions[start_idx:].copy()
        self.queue = processed_actions[start_idx:].copy()

        self.last_index = 0

    def _append_actions_queue(
        self, original_actions: np.ndarray, processed_actions: np.ndarray
    ):
        """Append new actions to the queue (non-RTC mode)."""
        if self.queue is None:
            self.original_queue = original_actions.copy()
            self.queue = processed_actions.copy()
            return

        # Remove consumed actions
        self.original_queue = self.original_queue[self.last_index :]
        self.queue = self.queue[self.last_index :]

        # Append new actions
        self.original_queue = np.concatenate([self.original_queue, original_actions])
        self.queue = np.concatenate([self.queue, processed_actions])

        self.last_index = 0

    def _check_delays(
        self, real_delay: int, action_index_before_inference: Optional[int] = None
    ):
        """Validate that computed delays match expectations."""
        if action_index_before_inference is None:
            return

        indexes_diff = self.last_index - action_index_before_inference
        if indexes_diff != real_delay:
            logger.warning(
                f"[ACTION_QUEUE] Indexes diff is not equal to real delay. "
                f"Indexes diff: {indexes_diff}, real delay: {real_delay}"
            )
