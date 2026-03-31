"""RL Token encoder and decoder modules.

The encoder compresses VLA final-layer embeddings into a compact RL token z_rl.
The decoder reconstructs VLA embeddings from z_rl for training the information bottleneck.
"""

import torch
from torch import Tensor
from torch import nn


class RLTokenEncoder(nn.Module):
    """Encoder transformer g_phi that compresses VLA embeddings into an RL token.

    Appends a learnable RL token embedding to the VLA embedding sequence,
    processes through a small transformer, and extracts the RL token output.

    z_rl = g_phi([z_{1:M}, e_rl])_{M+1}
    """

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.vla_embed_dim

        # Learnable RL token embedding
        self.rl_token_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)

        # Small transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.final_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, vla_embeddings: Tensor, pad_mask: Tensor) -> Tensor:
        """Encode VLA embeddings into RL token.

        Args:
            vla_embeddings: [B, M, D] VLA final-layer embeddings
            pad_mask: [B, M] boolean mask (True = valid token)

        Returns:
            z_rl: [B, D] compressed RL token embedding
        """
        batch_size = vla_embeddings.shape[0]

        # Append learnable RL token embedding to sequence
        e_rl = self.rl_token_embedding.expand(batch_size, 1, self.embed_dim)
        tokens = torch.cat([vla_embeddings, e_rl], dim=1)  # [B, M+1, D]

        # Extend pad_mask for RL token position (always valid)
        rl_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=pad_mask.device)
        extended_mask = torch.cat([pad_mask, rl_mask], dim=1)  # [B, M+1]

        # PyTorch TransformerEncoder uses True=ignore for src_key_padding_mask
        src_key_padding_mask = ~extended_mask  # [B, M+1]

        # Forward through transformer
        out = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        out = self.final_norm(out)

        # Extract RL token at last position
        z_rl = out[:, -1, :]  # [B, D]
        return z_rl


class RLTokenDecoder(nn.Module):
    """Decoder transformer d_phi + linear head h_phi.

    Autoregressively reconstructs VLA embeddings from the RL token z_rl.

    L_rto = E[sum_i ||h_phi(d_phi([z_rl, z_tilde_{1:i-1}]))_i - z_tilde_i||^2]
    """

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.vla_embed_dim
        self.max_seq_len = config.max_seq_len

        # Positional embeddings for target sequence
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.embed_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.final_norm = nn.LayerNorm(self.embed_dim)

        # Linear output head h_phi
        self.output_head = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, z_rl: Tensor, target_embeddings: Tensor, pad_mask: Tensor) -> Tensor:
        """Reconstruct VLA embeddings from RL token.

        Args:
            z_rl: [B, D] RL token embedding
            target_embeddings: [B, M, D] target VLA embeddings (stop-gradient)
            pad_mask: [B, M] boolean mask (True = valid)

        Returns:
            reconstructed: [B, M, D] reconstructed embeddings
        """
        _, seq_len, _ = target_embeddings.shape

        # Build shifted decoder input: [z_rl, z_tilde_1, ..., z_tilde_{M-1}]
        z_rl_expanded = z_rl.unsqueeze(1)  # [B, 1, D]
        shifted_input = torch.cat([z_rl_expanded, target_embeddings[:, :-1, :]], dim=1)  # [B, M, D]

        # Add positional embeddings
        positions = torch.arange(seq_len, device=shifted_input.device)
        shifted_input = shifted_input + self.pos_embedding(positions).unsqueeze(0)

        # Memory is z_rl
        memory = z_rl.unsqueeze(1)  # [B, 1, D]

        # Causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=shifted_input.device, dtype=shifted_input.dtype
        )

        # Target padding mask (True=ignore for PyTorch)
        tgt_key_padding_mask = ~pad_mask  # [B, M]

        # Forward through decoder
        out = self.transformer(
            tgt=shifted_input,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        out = self.final_norm(out)
        reconstructed = self.output_head(out)  # [B, M, D]

        return reconstructed

    def compute_loss(self, z_rl: Tensor, target_embeddings: Tensor, pad_mask: Tensor) -> Tensor:
        """Compute reconstruction loss L_rto.

        Args:
            z_rl: [B, D] RL token
            target_embeddings: [B, M, D] target embeddings (detached)
            pad_mask: [B, M] boolean mask

        Returns:
            loss: scalar reconstruction loss
        """
        target_detached = target_embeddings.detach()
        reconstructed = self.forward(z_rl, target_detached, pad_mask)

        # MSE loss with masking
        error = (reconstructed - target_detached) ** 2  # [B, M, D]
        # Mask out padded positions
        mask = pad_mask.unsqueeze(-1).float()  # [B, M, 1]
        masked_error = error * mask
        loss = masked_error.sum() / mask.sum().clamp(min=1.0) / self.embed_dim

        return loss


class RLTokenEncoderDecoder(nn.Module):
    """Combined encoder-decoder for Phase 1 RL token training."""

    def __init__(self, config):
        super().__init__()
        self.encoder = RLTokenEncoder(config)
        self.decoder = RLTokenDecoder(config)

    def forward(self, vla_embeddings: Tensor, pad_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Encode and reconstruct VLA embeddings.

        Args:
            vla_embeddings: [B, M, D] VLA embeddings
            pad_mask: [B, M] padding mask

        Returns:
            z_rl: [B, D] RL token
            loss: scalar reconstruction loss
        """
        z_rl = self.encoder(vla_embeddings, pad_mask)
        loss = self.decoder.compute_loss(z_rl, vla_embeddings, pad_mask)
        return z_rl, loss
