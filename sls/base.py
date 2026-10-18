import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class PathEncoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=6,
        proj=False,
        d_model_out=512,
        num_tokens=1024,
        dim_feedforward=2048,
        use_flash=False,
        cls_mode="cls",
        l2norm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cls_mode = cls_mode
        self.l2norm = l2norm
        if self.cls_mode == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.num_tokens = num_tokens
        self.embedding_head = nn.Embedding(num_tokens, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        if use_flash:
            raise NotImplementedError("FlashMHA is not implemented in this snippet.")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, d_model_out) if proj else None

    def forward(
        self,
        x,
        mask=None,
        src_key_padding_mask=None,
        is_causal=None,
        return_l2norm=False,
    ):
        """add positional encoding and cls token"""
        x = self.embedding_head(x)
        if self.cls_mode == "cls":
            x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        elif src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask[:, 1:]
        x = self.pos_enc(x.transpose(0, 1)).transpose(0, 1)
        x = self.encoder(
            x, mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal
        )
        if self.cls_mode == "cls":
            out_cls = x[:, 0, :]
        elif self.cls_mode == "mean":
            if src_key_padding_mask is not None:
                x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0)
            out_cls = x.mean(dim=1)
        elif self.cls_mode == "last":
            if src_key_padding_mask is not None:
                last_non_pad = (~src_key_padding_mask).long().cumsum(dim=1) - 1
                last_non_pad = last_non_pad.clamp(min=0)
                last_indices = last_non_pad[:, -1]
                batch_idx = torch.arange(x.size(0), device=x.device)
                out_cls = x[batch_idx, last_indices, :]
            else:
                out_cls = x[:, -1, :]
        else:
            raise ValueError(
                f"Invalid cls_mode: {self.cls_mode}. Must be 'cls', 'last' or 'mean'."
            )
        if self.proj:
            out_cls = self.proj(out_cls)
        if return_l2norm:
            batch_l2norm = (
                out_cls.norm(p=2, dim=-1, keepdim=False).detach().clone().mean().cpu()
            )
        if self.l2norm:
            out_cls = out_cls / out_cls.norm(p=2, dim=-1, keepdim=True)
        if return_l2norm:
            return (out_cls, batch_l2norm)
        return out_cls


class PathDecoder(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=6,
        proj=False,
        d_model_in=512,
        num_tokens=1024,
        dim_feedforward=2048,
        use_flash=False,
        **kwargs,
    ):
        """
        Args:
            d_model: int, the number of expected features in the input
            nhead: int, the number of heads in the multiheadattention models
            num_layers: int, the number of sub-encoder-layers in the encoder
            num_tokens: int, the number of tokens in the vocabulary
        """
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.embedding_head = nn.Embedding(num_tokens, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        if use_flash:
            raise NotImplementedError("FlashMHA is not implemented in this snippet.")
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model_in, d_model) if proj else None

    def forward(self, cls_vec, src, mask, src_key_padding_mask, is_causal=True):
        """
        Args:
            cls: Tensor, shape ``[batch_size, d_model]``
            src: Tensor, shape ``[batch_size, seq_len, d_model]``
            mask: Tensor, shape ``[seq_len?]``
            src_key_padding_mask: Tensor, shape ``[batch_size, seq_len]``
            is_causal: bool, whether to use a causal mask or not
        """
        if self.proj:
            cls_vec = self.proj(cls_vec)
        if src is None:
            x = self.pos_enc(cls_vec.unsqueeze(0)).transpose(0, 1)
            x = self.decoder(
                x,
                mask=mask.to(torch.bool),
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
        else:
            x = self.embedding_head(src)
            x = torch.cat([cls_vec.unsqueeze(1), x], dim=1)
            x = self.pos_enc(x.transpose(0, 1)).transpose(0, 1)
            x = self.decoder(
                x,
                mask=mask.to(torch.bool),
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
        return x
