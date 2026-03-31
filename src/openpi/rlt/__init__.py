"""RLT: Reinforcement Learning Token for VLA models.

Implements the RL Token approach from "RL Token: Bootstrapping Online RL with
Vision-Language-Action Models" (Physical Intelligence, 2025).

Phase 1: Train an encoder-decoder to compress VLA embeddings into a compact RL token.
Phase 2: Train lightweight actor-critic networks on the RL token for online RL.
"""

from openpi.rlt.actor import RLTActor
from openpi.rlt.config import RLTTrainConfig
from openpi.rlt.critic import RLTCritic
from openpi.rlt.encoder_decoder import RLTokenDecoder
from openpi.rlt.encoder_decoder import RLTokenEncoder
from openpi.rlt.encoder_decoder import RLTokenEncoderDecoder
from openpi.rlt.replay_buffer import ReplayBuffer
from openpi.rlt.td3 import TD3

__all__ = [
    "TD3",
    "RLTActor",
    "RLTCritic",
    "RLTTrainConfig",
    "RLTokenDecoder",
    "RLTokenEncoder",
    "RLTokenEncoderDecoder",
    "ReplayBuffer",
]
