# A Scalable Vector Graphics Latent Space

**ECCV 2026** · [Project page](https://aimagelab.github.io/svg_latent_space/)

[![arXiv](https://img.shields.io/badge/arXiv-2608.21893-b31b1b.svg)](https://arxiv.org/abs/2608.21893)

SLS encodes each SVG path into a 1024 dimensional vector and decodes it back
into symbolic geometry and style. This repository contains the model,
preprocessing, training and inference code.

## Installation

Use Python 3.11. Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements include PyTorch 2.9.1. Choose its CPU or CUDA build to match
your machine. Training defaults to BF16; CPU training uses `--cpu`.

## Required: configure weights and tokenizer

**In this version, you must set `SLS_ASSETS_DIR` before running training or
inference with the default `configs/sls.yaml`.** This environment variable
points to the folder containing both the checkpoint and the tokenizer.

Arrange the release assets as follows:

```text
sls_release/
├── weights/
│   └── sls/
│       └── model.safetensors
└── tokenizer/
    ├── tokenizer.model
    └── tokenizer.vocab
```

Replace `path/to/sls_release` below with the location of that folder, then run:

```bash
export SLS_ASSETS_DIR="$(realpath path/to/sls_release)"
```

**Repeat this export in every new terminal session**, or add the assignment to
your shell configuration. Activate the Python environment in that session too.
The code reports an error if the variable is unset or empty.

The default config uses the same macro for both assets:

```yaml
model:
  ckpt_path: ${SLS_ASSETS_DIR}/weights/sls
tokenizer:
  path: ${SLS_ASSETS_DIR}/tokenizer
```

`ckpt_path` names the directory containing `model.safetensors`.
`SLS_ASSETS_DIR` names the parent asset folder, not the model file or tokenizer
folder. Asset checksums and sizes are in [assets_manifest.json](assets_manifest.json).
The model checkpoint is excluded from Git.

## Inference

After completing the asset setup above, run from the repository root:

```bash
# Reconstruct an SVG.
python inference.py reconstruct --input examples/example.svg --output outputs/reconstructed.svg

# Save one latent vector per path.
python inference.py encode --input examples/example.svg --output outputs/embeddings.safetensors

# Decode saved vectors into an SVG.
python inference.py decode --input outputs/embeddings.safetensors --output outputs/decoded.svg
```

CUDA is selected when available. Useful options:

| Option | Purpose |
| --- | --- |
| `--device cpu` | Run on CPU. |
| `--batch-size 4` | Adjust the number of paths processed together. |
| `--sampling-method top_p` | Change greedy decoding to nucleus sampling. |
| `--checkpoint DIRECTORY` | Override the model directory; the tokenizer still uses `SLS_ASSETS_DIR`. |
| `--config-file FILE` | Select another model and generation config. |

Decoding supports `greedy`, `multinomial`, `top_k` (k=5), and `top_p` (p=0.9).

Preprocessing converts supported SVG shapes to paths, scales coordinates to a
512×512 canvas, rounds coordinates and serializes style. It does not implement
every SVG rendering feature, including arbitrary CSS and nested transforms.
Inference rejects paths exceeding 1024 tokens. Predicted paths may require
validation before use as XML.

Embeddings are raw SLS vectors: `cls_mode: last`, `l2norm: false`. Preserve their
magnitude when decoding. For cosine similarity, normalize a separate copy with
`torch.nn.functional.normalize(z, dim=-1)`.

For Python integration, use `load_model`, `encode_paths` and `decode_paths` from
`sls.runtime`, with canonical path strings from `sls.preprocess.preprocess`.

## Prepare training data

Training accepts JSON or JSONL containing one canonical SVG path per `text`
field, without `<s>` or `</s>` markers:

```json
{"text": "<path style=\"fill:rgb(0,0,0);\" d=\"M0,0L5,5Z\"/>"}
```

Split SVG files into train and evaluation directories before extracting paths:

```bash
python prepare_data.py --input-dir data/train_svgs --output data/train.jsonl
python prepare_data.py --input-dir data/eval_svgs --output data/eval.jsonl
```

Preprocessing does not require `SLS_ASSETS_DIR`. Training adds the sequence
markers and constructs the teacher-forcing targets.

## Training

Set `SLS_ASSETS_DIR` as described above, including when training from scratch:
training loads the tokenizer from that folder.

```bash
python train.py \
  --train-dataset-path data/train.jsonl \
  --eval-dataset-path data/eval.jsonl \
  --output-dir outputs/sls
```

For multiple GPUs:

```bash
torchrun --standalone --nproc-per-node=4 train.py \
  --train-dataset-path 'data/train_*.jsonl' \
  --eval-dataset-path data/eval.jsonl \
  --output-dir outputs/sls
```

`configs/sls.yaml` defines the architecture, Gaussian noise, augmentations and
optimizer settings. The default batch size is 48 per device. Use `--batch-size`
to adjust it, `--cpu` for CPU execution, and `--max-steps 2` for a short run.
Logging works without a W&B account.

Training starts from scratch. Add
`--init-checkpoint "$SLS_ASSETS_DIR/weights/sls"` to initialize from the released
weights. Add `--resume` to restore the latest Trainer checkpoint in the output
directory, including optimizer state, or `--resume DIRECTORY` to select one.

Trainer checkpoints are saved in `outputs/sls/checkpoint-*`. The final model,
tokenizer and config are saved together in `outputs/sls/final/`:

```bash
python inference.py reconstruct \
  --config-file outputs/sls/final/release_config.yaml \
  --input examples/example.svg --output outputs/finetuned.svg
```

This final bundle uses paths relative to its own config and does not require
`SLS_ASSETS_DIR`. Training datasets and previous optimizer/RNG states are not
included in the release assets.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover asset configuration, padding, gradients, checkpoint reload and
style augmentation. They do not require the released model checkpoint.

## Citation

If you use SLS in your research, please cite:

```bibtex
@inproceedings{zini2026scalable,
  title     = {A Scalable Vector Graphics Latent Space},
  author    = {Zini, Leonardo and Frigieri, Elia and Baraldi, Lorenzo},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
