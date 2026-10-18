"""Reconstruct an SVG, encode its paths, or decode saved latent vectors."""

import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

from sls.preprocess import preprocess
from sls.runtime import load_model, encode_paths, decode_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["reconstruct", "encode", "decode"])
    parser.add_argument("--config-file", default="configs/sls.yaml")
    parser.add_argument("--checkpoint", help="Override checkpoint directory")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--sampling-method", choices=["greedy", "multinomial", "top_k", "top_p"]
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    model, tokenizer, cfg = load_model(args.config_file, args.device, args.checkpoint)
    generation = dict(cfg["generation"])
    if args.max_length is not None:
        generation["max_length"] = args.max_length
    if args.sampling_method:
        generation["sampling_method"] = args.sampling_method
    if args.mode == "decode":
        embeddings = load_file(args.input)["embeddings"]
    else:
        # Same path canonicalizer used in the ECCV reconstruction code.
        paths = [p for p in preprocess(args.input).splitlines() if p.strip()]
        if not paths:
            raise ValueError("Input contains no supported SVG paths")
        embeddings = torch.cat(
            [
                encode_paths(
                    paths[i : i + args.batch_size],
                    model,
                    tokenizer,
                    cfg.get("max_seq_length", 1024),
                ).cpu()
                for i in range(0, len(paths), args.batch_size)
            ]
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "encode":
        save_file(
            {"embeddings": embeddings.contiguous()},
            str(output),
            metadata={"representation": "raw SLS latent; no unit normalization"},
        )
    else:
        paths = []
        for i in range(0, len(embeddings), args.batch_size):
            paths.extend(
                decode_paths(
                    embeddings[i : i + args.batch_size], model, tokenizer, generation
                )
            )
        output.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">\n'
            + "\n".join(paths)
            + "\n</svg>\n"
        )


if __name__ == "__main__":
    main()
