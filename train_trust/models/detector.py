"""
Vulnerability detector with line-level attention.

Slide 8 calls this the "Vulnerability Detector" block — it must produce
both a function-level prediction AND a per-line attention distribution
that L_trust can consume.

Architecture (LineVul-style, simplified):
    code → tokenizer → transformer backbone → [CLS] head for label
                                            → token attentions → line-level pool
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class VulnDetector(nn.Module):
    """
    Outputs:
        logits:    [B, 2]    classification logits (benign / vulnerable)
        attention: [B, L]    softmax distribution over the L source-code lines
    """

    def __init__(self, backbone: str, num_lines: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone,
            attn_implementation="eager",
            output_attentions=True,
        )

        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.num_lines = num_lines

    def forward(
        self,
        input_ids: torch.Tensor,           # [B, T]
        attention_mask: torch.Tensor,      # [B, T]
        line_ids: torch.Tensor,            # [B, T]   which line each token belongs to (-1 = pad)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        # Pooled hidden state ([CLS]) → classifier
        cls_hidden = out.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_hidden)

        # Per-token attention weight = mean of attention *received* by each
        # token from the [CLS] token, averaged across heads of the last layer.
        # This is the LineVul "important score" idea.
        last_attn = out.attentions[-1]               # [B, H, T, T]
        cls_to_tok = last_attn[:, :, 0, :].mean(1)   # [B, T] — how much [CLS] attends each token
        cls_to_tok = cls_to_tok * attention_mask     # mask out padding

        # Aggregate token attention to line-level by summing tokens per line.
        line_attn = self._tokens_to_lines(cls_to_tok, line_ids)
        # Re-normalise so each row sums to 1 (a proper distribution over lines).
        line_attn = line_attn / (line_attn.sum(dim=-1, keepdim=True) + 1e-8)
        return logits, line_attn

    def _tokens_to_lines(self, tok_attn: torch.Tensor, line_ids: torch.Tensor) -> torch.Tensor:
        """Sum token attention into per-line buckets.

        tok_attn:  [B, T]   per-token weight
        line_ids:  [B, T]   each entry in {0, ..., L-1, -1 (pad)}

        Returns:   [B, L]
        """
        B, T = tok_attn.shape
        L = self.num_lines
        line_attn = torch.zeros(B, L, device=tok_attn.device, dtype=tok_attn.dtype)
        # Replace pad (-1) with 0 and zero out their contributions.
        valid = (line_ids >= 0).float()
        safe_ids = line_ids.clamp(min=0)             # [B, T]
        contrib = tok_attn * valid                   # zero contribution for pad tokens
        line_attn.scatter_add_(1, safe_ids.long(), contrib)
        return line_attn
