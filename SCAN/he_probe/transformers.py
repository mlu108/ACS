# Adapted from ryeii/Representational-Homomorphism-for-Transformer-Language-Models
# (github.com/ryeii/Representational-Homomorphism-for-Transformer-Language-Models),
# he_probe/transformers.py.

import math
from typing import Optional, List

import torch
import torch.nn as nn


# ------------------------
# Positional Encoding
# ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1), :]


# ------------------------
# Transformer Blocks
# ------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: (T, B, D)
            attn_mask: (T, T) boolean mask, True means masked
            key_padding_mask: (B, T) boolean mask, True means ignore / pad
        """
        h = x
        x2, _ = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.ln1(h + self.dropout(x2))
        x = self.ln2(x + self.dropout(self.ff(x)))
        return x


# ------------------------
# Decoder-Only Transformer
# ------------------------
class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=256,
        max_len=50,
        dropout=0.1,
        pad_id=0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.n_layers = n_layers
        self.pad_id = pad_id
        self.max_len = max_len

    def _causal_mask(self, size, device):
        # shape: (T, T), True means masked
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None, return_hidden_states: bool = True):
        """
        Args:
            x: (B, T)
            padding_mask: (B, T) boolean, True for real tokens, False for pad
            return_hidden_states: whether to return hidden states per layer

        Returns:
            out: (B, T, V)
            hidden_states: list[(B, T, D)] if requested, else []
        """
        if padding_mask is None:
            padding_mask = (x != self.pad_id)

        x = self.token_emb(x)       # (B, T, D)
        x = self.pos_emb(x)
        x = x.transpose(0, 1)       # (T, B, D) for MultiheadAttention

        causal_mask = self._causal_mask(x.size(0), x.device)
        key_padding_mask = ~padding_mask  # MultiheadAttention expects True = ignore

        hidden_states: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=causal_mask,
                key_padding_mask=key_padding_mask,
            )
            if return_hidden_states:
                hidden_states.append(x.transpose(0, 1))  # (B, T, D)

        x = self.ln_final(x)
        out = self.head(x.transpose(0, 1))  # (B, T, V)
        return out, hidden_states

    @torch.no_grad()
    def generate(self, prefix_ids, eos_id, max_new_tokens):
        """
        Greedy generation.

        Args:
            prefix_ids: (B, T_prefix)
            eos_id: token id marking end of sequence
            max_new_tokens: maximum number of tokens to generate

        Returns:
            full generated sequence including prefix: (B, T_prefix + new_tokens)
        """
        self.eval()
        x = prefix_ids

        for _ in range(max_new_tokens):
            padding_mask = (x != self.pad_id)
            logits, _ = self.forward(
                x,
                padding_mask=padding_mask,
                return_hidden_states=False,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)

            if (next_token.squeeze(1) == eos_id).all():
                break

        return x

    def get_hidden_states(self, x, padding_mask: Optional[torch.Tensor] = None):
        """Return a list of hidden states (B, T, D) per layer."""
        self.eval()
        with torch.no_grad():
            _, hidden_states = self.forward(
                x,
                padding_mask=padding_mask,
                return_hidden_states=True,
            )
        return hidden_states


# ------------------------
# Encoder-Decoder Transformer
# ------------------------
class EncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size_src,
        vocab_size_tgt,
        d_model=128,
        n_layers_enc=4,
        n_layers_dec=4,
        n_heads=4,
        d_ff=256,
        max_len=50,
        dropout=0.1,
    ):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size_src, d_model)
        self.tgt_emb = nn.Embedding(vocab_size_tgt, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.pos_dec = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_enc)]
        )
        self.encoder_ln = nn.LayerNorm(d_model)

        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_dec)]
        )
        self.decoder_ln = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, vocab_size_tgt)
        self.d_model = d_model

    def _causal_mask(self, size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, src, tgt):
        # src: (B, T_src), tgt: (B, T_tgt)
        src_h = self.src_emb(src)
        src_h = self.pos_enc(src_h).transpose(0, 1)  # (T, B, D)
        for layer in self.encoder_layers:
            src_h = layer(src_h)
        src_h = self.encoder_ln(src_h)

        tgt_h = self.tgt_emb(tgt)
        tgt_h = self.pos_dec(tgt_h).transpose(0, 1)
        for layer in self.decoder_layers:
            tgt_h = layer(tgt_h, attn_mask=self._causal_mask(tgt_h.size(0), tgt_h.device))
        tgt_h = self.decoder_ln(tgt_h)
        out = self.head(tgt_h.transpose(0, 1))
        return out

    def get_hidden_states(self, src, tgt):
        """Return decoder hidden states list (B, T, D) per layer"""
        self.eval()
        with torch.no_grad():
            src_h = self.src_emb(src)
            src_h = self.pos_enc(src_h).transpose(0, 1)
            for layer in self.encoder_layers:
                src_h = layer(src_h)
            src_h = self.encoder_ln(src_h)

            tgt_h = self.tgt_emb(tgt)
            tgt_h = self.pos_dec(tgt_h).transpose(0, 1)
            hidden_states = []
            for layer in self.decoder_layers:
                tgt_h = layer(tgt_h, attn_mask=self._causal_mask(tgt_h.size(0), tgt_h.device))
                hidden_states.append(tgt_h.transpose(0, 1))
        return hidden_states