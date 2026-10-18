"""Path dataset and teacher-forcing targets from the SLS training pipeline."""

from datasets import load_dataset
from torch.utils.data import Dataset
import torch

from .transforms import transformations


class PathDataset(Dataset):
    def __init__(self, filenames, data_aug=None):
        self.dataset = load_dataset("json", data_files=filenames, split="train")
        self.transforms = []
        if data_aug:
            for name in data_aug["transformations"]:
                self.transforms.append(
                    transformations[name](threshold=data_aug.get("threshold", 0.33))
                )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text = self.dataset[index]["text"]
        for transform in self.transforms:
            text = transform(text)
        return {"text": "<s>" + text + "</s>"}


class PathCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        tokens = self.tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = tokens["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        # Latent prefix predicts BOS; the final decoder position is ignored.
        tokens["labels"] = torch.cat(
            [labels, labels.new_full((labels.shape[0], 1), -100)], dim=1
        )
        return tokens
