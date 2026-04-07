from threading import Lock
from typing import Optional

import numpy as np

from lerobot.utils.robot_utils import get_logger

logger = get_logger("ActionQueue")


class ActionQueue:
    """Thread-safe action buffer for real-time chunking.

    Stores two parallel arrays:
    - original: actions in model (normalized) space, used to build action_prefix for the next inference
    - processed: actions in robot (denormalized) space, consumed by the control loop
    """

    def __init__(self):
        self._original: Optional[np.ndarray] = None  # (chunk_len, action_dim)
        self._processed: Optional[np.ndarray] = None  # (chunk_len, action_dim)
        self._cursor: int = 0
        self._lock = Lock()

    def get(self) -> Optional[np.ndarray]:
        """Pop the next processed action for robot execution.

        Returns:
            (action_dim,) array or None if queue is empty.
        """
        with self._lock:
            if self._processed is None or self._cursor >= len(self._processed):
                return None
            action = self._processed[self._cursor].copy()
            self._cursor += 1
            return action

    def qsize(self) -> int:
        """Number of remaining actions."""
        with self._lock:
            if self._processed is None:
                return 0
            return len(self._processed) - self._cursor

    def get_cursor(self) -> int:
        """Current consumption position (thread-safe)."""
        with self._lock:
            return self._cursor

    def get_remaining_original(self) -> Optional[np.ndarray]:
        """Get unconsumed original (model-space) actions from cursor onward.

        Used by the broker to build action_prefix for the next inference.

        Returns:
            (remaining_len, action_dim) array or None if no actions.
        """
        with self._lock:
            if self._original is None:
                return None
            remaining = self._original[self._cursor:]
            return remaining.copy() if len(remaining) > 0 else None

    def replace(self, original: np.ndarray, processed: np.ndarray, start_from: int):
        """Replace the buffer with a new chunk, starting execution from `start_from`.

        Args:
            original: New chunk in model space (chunk_len, action_dim).
            processed: New chunk in robot space (chunk_len, action_dim).
            start_from: Index to start executing from (typically real_delay).
        """
        with self._lock:
            self._original = original.copy()
            self._processed = processed.copy()
            self._cursor = max(0, min(start_from, len(processed)))

    def clear(self):
        """Reset to empty state."""
        with self._lock:
            self._original = None
            self._processed = None
            self._cursor = 0
