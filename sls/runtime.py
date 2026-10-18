"""Load configs with environment macros and paths relative to the config file."""

from pathlib import Path
import os
import re
import torch
import yaml

from .model import SLS
from .tokenizer import MySPTokenizer


def read_config(filename):
    filename = Path(filename).resolve()
    with filename.open() as stream:
        cfg = yaml.safe_load(stream)
    for section, key in (("model", "ckpt_path"), ("tokenizer", "path")):
        value = cfg[section].get(key)
        if value:
            variables = re.findall(r"\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)", value)
            for braced, plain in variables:
                name = braced or plain
                if not os.environ.get(name):
                    raise ValueError(
                        f"Set the {name} environment variable before loading this config"
                    )
            expanded = Path(os.path.expandvars(value)).expanduser()
            cfg[section][key] = str((filename.parent / expanded).resolve())
    return cfg


def load_model(config_file, device=None, checkpoint=None):
    cfg = read_config(config_file)
    tokenizer = MySPTokenizer.from_pretrained(cfg["tokenizer"]["path"])
    if len(tokenizer) != cfg["model"]["num_tokens"]:
        raise ValueError("Tokenizer vocabulary and model num_tokens differ")
    model_cfg = dict(cfg["model"])
    if checkpoint:
        model_cfg["ckpt_path"] = str(Path(checkpoint).resolve())
    if not model_cfg.get("ckpt_path"):
        raise ValueError("Inference requires a checkpoint")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SLS(**model_cfg).to(device).eval()
    return model, tokenizer, cfg


@torch.inference_mode()
def encode_paths(paths, model, tokenizer, max_length=1024):
    if not paths:
        raise ValueError("No paths to encode")
    tokens = tokenizer(
        ["<s>" + path + "</s>" for path in paths],
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    if tokens["input_ids"].shape[1] > max_length:
        raise ValueError(
            f"A path exceeds {max_length} tokens; split it before encoding"
        )
    return model.encoder_forward(**tokens.to(next(model.parameters()).device))


@torch.inference_mode()
def decode_paths(embeddings, model, tokenizer, generation):
    ids = model.generate(
        embeddings.to(next(model.parameters()).device),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **generation,
    )
    return tokenizer.batch_decode(ids.cpu().tolist(), skip_special_tokens=True)
