"""Train the SLS path autoencoder, on one device or with torchrun."""

import argparse
from pathlib import Path
import yaml

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed

from sls.data import PathDataset, PathCollator
from sls.model import SLS
from sls.runtime import read_config
from sls.tokenizer import MySPTokenizer


class SLSTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default="configs/sls.yaml")
    parser.add_argument(
        "--train-dataset-path",
        nargs="+",
        required=True,
        help="JSON/JSONL files or quoted glob patterns",
    )
    parser.add_argument("--eval-dataset-path", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/sls")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Resume latest Trainer checkpoint, or supply a checkpoint directory",
    )
    parser.add_argument(
        "--init-checkpoint", help="Initialize weights without restoring optimizer state"
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    cfg = read_config(args.config_file)
    set_seed(42)
    tokenizer = MySPTokenizer.from_pretrained(cfg["tokenizer"]["path"])
    if len(tokenizer) != cfg["model"]["num_tokens"]:
        raise ValueError("Tokenizer vocabulary and model num_tokens differ")
    model_cfg = dict(cfg["model"])
    # Training starts from scratch unless explicitly initialized or resumed.
    model_cfg.pop("ckpt_path", None)
    if args.init_checkpoint:
        model_cfg["ckpt_path"] = args.init_checkpoint
    model = SLS(**model_cfg)
    options = dict(cfg["trainer"])
    options.update(
        output_dir=args.output_dir,
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        prediction_loss_only=True,
    )
    if args.max_steps is not None:
        options["max_steps"] = args.max_steps
    if args.batch_size is not None:
        options["per_device_train_batch_size"] = args.batch_size
        options["per_device_eval_batch_size"] = args.batch_size
    if args.cpu:
        options.update(use_cpu=True, bf16=False, fp16=False, dataloader_num_workers=0)
        options.pop("dataloader_prefetch_factor", None)
        options["dataloader_persistent_workers"] = False
    callbacks = []
    if cfg.get("early_stopping_patience") and options.get("load_best_model_at_end"):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"]
            )
        )
    trainer = SLSTrainer(
        model=model,
        args=TrainingArguments(**options),
        train_dataset=PathDataset(
            args.train_dataset_path, cfg.get("data_augmentation")
        ),
        eval_dataset=PathDataset(args.eval_dataset_path),
        data_collator=PathCollator(tokenizer, cfg.get("max_seq_length", 1024)),
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        cfg["model"]["ckpt_path"] = "."
        cfg["tokenizer"]["path"] = "."
        with (final_dir / "release_config.yaml").open("w") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)


if __name__ == "__main__":
    main()
