import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from .base import PathEncoder, PathDecoder
from .sampling import generation_methods


def add_sphere_noise(z, sigma=0.1):
    epsilon = torch.randn_like(z)
    epsilon_perp = epsilon - (epsilon * z).sum(dim=-1, keepdim=True) * z
    z_new = z + sigma * epsilon_perp
    z_new = z_new / z_new.norm(p=2, dim=-1, keepdim=True)
    return z_new


class SLS(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=6,
        num_tokens=1024,
        dim_feedforward=2048,
        **kwargs,
    ):
        super().__init__()
        use_flash = kwargs.get("use_flash", False)
        self.cls_mode = kwargs.get("cls_mode", "cls")
        self.l2norm = kwargs.get("l2norm", False)
        self.l2_regularization = kwargs.get("l2_regularization", 0.0)
        self.kl_loss = kwargs.get("kl_loss", 0.0)
        self.kl_target_scale = kwargs.get("kl_target_scale", 1.0)
        assert self.cls_mode in ["cls", "last", "mean"], (
            f"Invalid cls_mode: {self.cls_mode}. Must be 'cls', 'last' or 'mean'."
        )
        d_model_out = kwargs.get("d_model_out", d_model)
        proj = kwargs.get("proj", False)
        self.encoder = PathEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_tokens=num_tokens,
            dim_feedforward=dim_feedforward,
            d_model_out=d_model_out,
            proj=proj,
            use_flash=use_flash,
            l2norm=self.l2norm,
            cls_mode=self.cls_mode,
        )
        self.decoder = PathDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_tokens=num_tokens,
            dim_feedforward=dim_feedforward,
            d_model_in=d_model_out,
            proj=proj,
            use_flash=use_flash,
        )
        self.lm_head = nn.Linear(d_model, num_tokens)
        self.noise = kwargs.get("noise", 0.0)
        if self.l2norm:
            self.noise = torch.Tensor([self.noise]).float().item()
        self.command_loss = kwargs.get("command_loss", False)
        if self.command_loss:
            self.target_token_ids = kwargs.get("target_token_ids", [0, 1, 2, 3, 4])
            assert self.target_token_ids != [0, 1, 2, 3, 4]
        self.loss_fn = nn.CrossEntropyLoss()
        self.use_logit_scale = kwargs.get("use_logit_scale", False)
        stating_logit_scale = kwargs.get("stating_logit_scale", 0.07)
        if self.use_logit_scale:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1 / stating_logit_scale)
            )
        else:
            self.logit_scale = None
        ckpt_path = kwargs.get("ckpt_path", None)
        pca_config = kwargs.get("pca", None)
        if pca_config is not None and (not proj):
            raise ValueError("PCA requires projection layers")
        if ckpt_path is not None:
            self.from_pretrained(ckpt_path, pca_config=pca_config)

    @torch.no_grad()
    def encoder_forward(
        self, input_ids, attention_mask=None, src_key_padding_mask=None, **kwargs
    ):
        x = input_ids.detach().clone()
        attention_mask = (
            torch.cat(
                [
                    torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype),
                    attention_mask,
                ],
                dim=1,
            )
            if attention_mask is not None
            else None
        )
        src_key_padding_mask = (
            attention_mask == 0 if attention_mask is not None else None
        )
        cls_vec = self.encoder(
            x,
            mask=None,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=False,
            return_l2norm=False,
        )
        normalize = kwargs.get("normalize", False)
        if normalize:
            cls_vec = (cls_vec - self.embedd_mean) / self.embedd_std
        return cls_vec

    @torch.no_grad()
    def decoder_forward(
        self, cls_vec, input_ids, attention_mask=None, src_key_padding_mask=None
    ):
        seq_len = input_ids.size(1) + 1
        device = input_ids.device
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        out = self.decoder(
            cls_vec,
            input_ids,
            causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )
        out = self.lm_head(out)
        return out

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        src_key_padding_mask=None,
        **kwargs,
    ):
        x = input_ids.detach().clone()
        if attention_mask is not None and attention_mask.shape[1] == x.shape[1]:
            attention_mask = torch.cat(
                [torch.ones_like(attention_mask[:, :1]), attention_mask], dim=1
            )
        src_key_padding_mask = (
            attention_mask == 0 if attention_mask is not None else None
        )
        seq_len = x.size(1) + 1
        device = x.device
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        cls_vec, batch_l2norm = self.encoder(
            x,
            mask=None,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=False,
            return_l2norm=True,
        )
        if self.noise > 0.0:
            if self.l2norm:
                cls_vec = add_sphere_noise(cls_vec, sigma=self.noise)
            else:
                noise_vec = (
                    torch.normal(mean=0, std=self.noise, size=cls_vec.size())
                    .to(cls_vec.device)
                    .to(torch.bfloat16)
                )
                cls_vec = cls_vec + noise_vec
        if self.use_logit_scale:
            logit_scale = self.logit_scale.exp()
            cls_vec = cls_vec * logit_scale
        decoder_dtype = (
            torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else self.decoder.embedding_head.weight.dtype
        )
        out = self.decoder(
            cls_vec.to(decoder_dtype),
            x,
            causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )
        out = self.lm_head(out)
        pred = out.view(-1, out.size(-1))
        labels = labels.view(-1)
        if self.l2_regularization > 0.0:
            main_loss = (
                self.loss_fn(pred, labels)
                + cls_vec.norm(p=2, dim=-1).mean() * self.l2_regularization
            )
        elif self.kl_loss > 0.0:
            main_loss = (
                self.loss_fn(pred, labels)
                + F.kl_div(
                    F.log_softmax(pred, dim=-1),
                    F.softmax(self.kl_target_scale * pred.detach(), dim=-1),
                    reduction="batchmean",
                )
                * self.kl_loss
            )
        else:
            main_loss = self.loss_fn(pred, labels)
        loss = main_loss
        return {
            "loss": loss if self.training else loss.detach(),
            "eval_loss": loss.detach() if not self.training else None,
            "logits": out.detach() if not self.training else None,
            "batch_l2norm": batch_l2norm,
            "logit_scale": self.logit_scale.detach().cpu().item()
            if self.use_logit_scale
            else None,
        }

    @torch.no_grad()
    def generate(
        self,
        cls_token,
        sampling_method="greedy",
        bos_token_id=2,
        max_length=100,
        temperature=1.0,
        eos_token_id=3,
    ):
        if temperature <= 0 or max_length < 1:
            raise ValueError("temperature and max_length must be positive")
        device = cls_token.device
        batch_size = cls_token.size(0)
        x = None
        if bos_token_id is not None:
            x = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.int64, device=device
            )
            src_key_padding_mask = torch.zeros(
                batch_size, 2, dtype=torch.bool, device=device
            )
        else:
            src_key_padding_mask = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=device
            )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_length):
            if finished.all():
                break
            if x is None:
                seq_len = 1
            else:
                seq_len = x.size(1) + 1
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
            tmp = self.decoder(
                cls_token,
                x,
                causal_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=True,
            )
            logits = self.lm_head(tmp)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = generation_methods[sampling_method](next_token_logits)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1), eos_token_id, next_token
                )
            if x is None:
                x = next_token
            else:
                x = torch.cat([x, next_token], dim=1)
            if eos_token_id is not None:
                src_key_padding_mask = torch.cat(
                    [src_key_padding_mask, next_token == eos_token_id], dim=1
                )
            else:
                src_key_padding_mask = None
            if eos_token_id is not None:
                newly_finished = next_token.squeeze(1) == eos_token_id
                finished = finished | newly_finished
        return x

    @torch.no_grad()
    def from_pretrained(self, path, pca_config=None):
        w = load_file(f"{path}/model.safetensors")
        w = {k: v for k, v in w.items() if k not in ("embedd_mean", "embedd_std")}
        self.load_state_dict(w, strict=True)
        space_stats_path = f"{path}/space_stats.safetensors"
        if os.path.exists(space_stats_path):
            stats = load_file(space_stats_path)
            tmp_mean = stats["mean"].unsqueeze(0)
            tmp_std = stats["std"].unsqueeze(0)
        else:
            tmp_mean = torch.zeros((1, self.lm_head.in_features))
            tmp_std = torch.ones((1, self.lm_head.in_features))
        self.register_buffer("embedd_mean", tmp_mean.detach().clone(), persistent=False)
        self.register_buffer("embedd_std", tmp_std.detach().clone(), persistent=False)
        if pca_config is not None:
            pca_weight = pca_config.get("path", None)
            if os.path.exists(pca_weight):
                pca_dict = torch.load(pca_weight)
                self.encoder.proj.weight.data = (
                    pca_dict["weight"].t().data.cpu().clone()
                )
                self.decoder.proj.weight.data = pca_dict["weight"].data.cpu().clone()
        return self
