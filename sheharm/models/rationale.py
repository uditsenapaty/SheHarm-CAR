"""Evidence- and rule-grounded rationale generation.

The decoder is conditioned on the final reasoning representation u, the soft target t~,
the retrieved knowledge k~, and the activated rules, and is trained with autoregressive
cross-entropy (L_rat). Decoding uses beam search with beam size 4 and at most 64 tokens
(paper Table `tab:hyperparameters`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RationaleDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.2,
        max_len: int = 64,
        pad_token_id: int = 1,
        memory_slots: int = 4,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.max_len = max_len
        self.memory_slots = memory_slots
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.memory_norm = nn.LayerNorm(hidden_size)
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        """Transformer-standard std=0.02.

        nn.Embedding defaults to N(0, 1); because lm_head is tied to it, that default puts
        the initial rationale cross-entropy near 380 instead of ln(vocab) ~ 10.8, which
        would swamp every other loss term for the first epochs.
        """
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.pad_token_id].zero_()

    def build_memory(self, u, soft_target, retrieved, contrastive) -> torch.Tensor:
        """One memory slot per evidence source, so cross-attention can select among them."""
        return self.memory_norm(torch.stack([u, soft_target, retrieved, contrastive], dim=1))

    def forward(self, memory: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        length = decoder_input_ids.size(1)
        positions = torch.arange(length, device=decoder_input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(decoder_input_ids) + self.position_embedding(positions.clamp_max(self.max_len - 1))
        causal_mask = torch.triu(
            torch.ones(length, length, device=hidden.device, dtype=torch.bool), diagonal=1
        )
        output = self.decoder(
            tgt=hidden, memory=memory, tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_input_ids.eq(self.pad_token_id),
        )
        return self.lm_head(output)

    @torch.no_grad()
    def generate(self, memory, bos_token_id: int, eos_token_id: int, beam_size: int = 4,
                 max_len: int | None = None, length_penalty: float = 1.0) -> torch.Tensor:
        """Batched beam search."""
        max_len = max_len or self.max_len
        batch, slots, width = memory.shape
        device = memory.device

        expanded = memory.unsqueeze(1).expand(batch, beam_size, slots, width).reshape(batch * beam_size, slots, width)
        tokens = torch.full((batch * beam_size, 1), bos_token_id, dtype=torch.long, device=device)
        scores = torch.full((batch, beam_size), float("-inf"), device=device)
        scores[:, 0] = 0.0
        scores = scores.reshape(-1)
        finished = torch.zeros(batch * beam_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            log_probabilities = F.log_softmax(self.forward(expanded, tokens)[:, -1], dim=-1)
            vocabulary = log_probabilities.size(-1)
            # A finished beam may only extend with padding, at zero additional cost.
            log_probabilities = log_probabilities.masked_fill(finished.unsqueeze(1), float("-inf"))
            log_probabilities[finished, self.pad_token_id] = 0.0

            candidates = (scores.unsqueeze(1) + log_probabilities).view(batch, beam_size * vocabulary)
            scores, flat_index = candidates.topk(beam_size, dim=-1)
            beam_index = flat_index // vocabulary
            token_index = flat_index % vocabulary

            offsets = (torch.arange(batch, device=device) * beam_size).unsqueeze(1)
            gather_index = (beam_index + offsets).reshape(-1)
            tokens = torch.cat([tokens[gather_index], token_index.reshape(-1, 1)], dim=1)
            finished = finished[gather_index] | token_index.reshape(-1).eq(eos_token_id)
            scores = scores.reshape(-1)
            if finished.all():
                break

        lengths = tokens.ne(self.pad_token_id).sum(dim=1).clamp_min(1)
        normalized = (scores / lengths.to(scores.dtype).pow(length_penalty)).view(batch, beam_size)
        best = normalized.argmax(dim=-1) + torch.arange(batch, device=device) * beam_size
        return tokens[best]
