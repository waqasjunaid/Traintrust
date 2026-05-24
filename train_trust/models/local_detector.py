"""
Local detector for offline smoke testing.

Uses a from-scratch tiny transformer instead of HuggingFace, so the
training loop can be exercised without network access.

For real experiments, use VulnDetector with a HuggingFace backbone like
'microsoft/graphcodebert-base'. This class exists ONLY to verify the
pipeline plumbing without any downloads.
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


class TinyTokenizer:
    """Whitespace + punctuation tokenizer with a small vocab.

    Public interface mimics HuggingFace tokenizers just enough for
    the collator (encode, cls_token_id, sep_token_id).
    """

    PAD = "[PAD]"
    CLS = "[CLS]"
    SEP = "[SEP]"
    UNK = "[UNK]"

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self._next_id = 0
        self.token_to_id: dict = {}
        for tok in [self.PAD, self.CLS, self.SEP, self.UNK]:
            self._add(tok)

    def _add(self, tok: str) -> int:
        if tok not in self.token_to_id:
            if self._next_id >= self.vocab_size:
                return self.token_to_id[self.UNK]
            self.token_to_id[tok] = self._next_id
            self._next_id += 1
        return self.token_to_id[tok]

    def _split(self, text: str) -> list:
        # Whitespace + simple punctuation split.
        out = []
        word = ""
        for ch in text:
            if ch.isalnum() or ch == "_":
                word += ch
            else:
                if word:
                    out.append(word); word = ""
                if not ch.isspace():
                    out.append(ch)
        if word:
            out.append(word)
        return out

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        ids = []
        if add_special_tokens:
            ids.append(self.token_to_id[self.CLS])
        for tok in self._split(text):
            ids.append(self._add(tok))
        if add_special_tokens:
            ids.append(self.token_to_id[self.SEP])
        return ids

    @property
    def cls_token_id(self) -> int:
        return self.token_to_id[self.CLS]

    @property
    def sep_token_id(self) -> int:
        return self.token_to_id[self.SEP]


class TinyTransformer(nn.Module):
    """A 2-layer transformer that outputs both logits and last-layer attention."""

    def __init__(self, vocab_size: int, hidden: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 256, num_classes: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(max_len, hidden)
        self.layers = nn.ModuleList([
            _TinyEncoderLayer(hidden, n_heads) for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Dropout(0.1), nn.Linear(hidden, num_classes),
        )
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        h = self.embed(input_ids) + self.pos(pos)
        last_attn = None
        for layer in self.layers:
            h, attn = layer(h, attention_mask)
            last_attn = attn   # [B, H, T, T]
        cls_h = h[:, 0, :]
        logits = self.classifier(cls_h)
        return logits, last_attn


class _TinyEncoderLayer(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(),
            nn.Linear(4 * hidden, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        # mask: [B, T] with 1 for real tokens, 0 for pad
        # nn.MultiheadAttention expects key_padding_mask with True = ignore
        kpm = (mask == 0)
        a, attn_weights = self.attn(h, h, h, key_padding_mask=kpm,
                                    need_weights=True, average_attn_weights=False)
        h = self.ln1(h + a)
        h = self.ln2(h + self.ff(h))
        return h, attn_weights        # attn_weights: [B, H, T, T]


class LocalVulnDetector(nn.Module):
    """Drop-in replacement for VulnDetector that uses TinyTransformer.

    Same forward signature, same output shapes — so train.py doesn't change.
    """

    def __init__(self, vocab_size: int, num_lines: int, **kwargs):
        super().__init__()
        self.transformer = TinyTransformer(vocab_size=vocab_size, **kwargs)
        self.num_lines = num_lines

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        line_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, last_attn = self.transformer(input_ids, attention_mask)
        # last_attn: [B, H, T, T] — average heads, take CLS row.
        cls_to_tok = last_attn.mean(dim=1)[:, 0, :]   # [B, T]
        cls_to_tok = cls_to_tok * attention_mask
        line_attn = self._tokens_to_lines(cls_to_tok, line_ids)
        line_attn = line_attn / (line_attn.sum(dim=-1, keepdim=True) + 1e-8)
        return logits, line_attn

    def _tokens_to_lines(self, tok_attn: torch.Tensor, line_ids: torch.Tensor) -> torch.Tensor:
        B, T = tok_attn.shape
        L = self.num_lines
        out = torch.zeros(B, L, device=tok_attn.device, dtype=tok_attn.dtype)
        valid = (line_ids >= 0).float()
        safe = line_ids.clamp(min=0)
        out.scatter_add_(1, safe.long(), tok_attn * valid)
        return out
