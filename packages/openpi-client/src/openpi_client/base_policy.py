import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass

    def warmup(self, obs: Dict) -> None:
        """Pre-warm the policy before the episode control loop starts.

        Override in subclasses that benefit from pre-compilation (e.g. JAX JIT).
        Default is a no-op.

        Args:
            obs: A real observation from the environment, used to trigger
                 inference so JIT compilation finishes before the loop.
        """
        pass
