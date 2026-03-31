"""VLA embedding extraction interface.

Wraps a frozen PI0Pytorch model to extract VLA embeddings and reference actions
for the RLT pipeline. The VLA model is completely frozen (no gradients).
"""

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


class VLAEmbeddingExtractor(nn.Module):
    """Wraps a frozen PI0Pytorch model for embedding extraction.

    Provides two key operations:
    1. Extract final-layer VLA embeddings z_{1:M} from the prefix (observation) stream
    2. Generate reference action chunks from the VLA's diffusion process
    """

    def __init__(self, pi0_model: PI0Pytorch):
        super().__init__()
        self.pi0 = pi0_model
        # Freeze all VLA parameters
        self.pi0.eval()
        for param in self.pi0.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_embeddings(self, observation) -> tuple[Tensor, Tensor, Tensor]:
        """Extract final-layer VLA prefix embeddings.

        Args:
            observation: Model observation (images, state, tokenized_prompt, etc.)

        Returns:
            vla_embeddings: [B, M, D] final-layer prefix embeddings (D=2048 for gemma_2b)
            pad_mask: [B, M] boolean padding mask
            state: [B, proprio_dim] proprioceptive state
        """
        images, img_masks, lang_tokens, lang_masks, state = self.pi0._preprocess_observation(observation, train=False)

        # Get raw prefix embeddings from SigLIP + language embedding
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.pi0.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        # Run prefix-only forward through PaliGemma to get final-layer embeddings
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.pi0._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Cast to model dtype if needed
        model_dtype = self.pi0.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if prefix_embs.dtype != model_dtype:
            prefix_embs = prefix_embs.to(dtype=model_dtype)

        # Prefix-only forward (inputs_embeds[1] is None path)
        (prefix_output, _), _ = self.pi0.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )

        return prefix_output.float(), prefix_pad_masks, state

    @torch.no_grad()
    def get_reference_actions(self, observation, device: torch.device, num_steps: int = 10) -> Tensor:
        """Generate reference action chunks from the frozen VLA.

        Args:
            observation: Model observation
            device: Target device
            num_steps: Number of diffusion denoising steps

        Returns:
            actions: [B, H, action_dim] full VLA action chunk (H=action_horizon)
        """
        return self.pi0.sample_actions(device, observation, num_steps=num_steps)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "VLAEmbeddingExtractor should not be called directly. Use extract_embeddings() or get_reference_actions()."
        )
