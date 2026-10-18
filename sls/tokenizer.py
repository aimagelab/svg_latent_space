import os
import json
import shutil
import torch
import sentencepiece as spm
from transformers import PreTrainedTokenizer


class MySPTokenizer(PreTrainedTokenizer):
    def __init__(self, sp_model_file, **kwargs):
        self.sp_model_file = sp_model_file
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_file)
        self._vocab_size = self.sp.get_piece_size()
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.mask_token = "<mask>"
        kwargs.update(
            {
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "mask_token": self.mask_token,
            }
        )
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return self._vocab_size

    def get_vocab(self):
        return {self.sp.id_to_piece(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text):
        return self.sp.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.sp.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp.id_to_piece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp.decode_pieces(tokens)

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        target = os.path.join(save_directory, "tokenizer.model")
        if os.path.abspath(self.sp_model_file) != os.path.abspath(target):
            shutil.copyfile(self.sp_model_file, target)
        return (target,)

    @classmethod
    def from_pretrained(cls, load_directory):
        sp_model_file = (
            os.path.join(load_directory, "tokenizer.model")
            if os.path.isfile(os.path.join(load_directory, "tokenizer.model"))
            else os.path.join(load_directory, "spiece.model")
        )
        return cls(sp_model_file)

    def _process_tokenizer_output(self, res_batch, add_extra_mask_token=True):
        """Helper method to clean tokenizer output and extend attention mask"""
        if "token_type_ids" in res_batch:
            del res_batch["token_type_ids"]
        if "attention_mask" in res_batch:
            res_batch["attention_mask"] = res_batch["attention_mask"].bool()
        if "attention_mask" in res_batch and add_extra_mask_token:
            res_batch["attention_mask"] = torch.cat(
                [
                    torch.ones((res_batch["attention_mask"].shape[0], 1)).to(
                        torch.bool
                    ),
                    res_batch["attention_mask"],
                ],
                dim=1,
            )
        return res_batch

    def __call__(self, *args, **kwargs):
        add_extra_mask_token = kwargs.pop("add_extra_mask_token", False)
        res_batch = super().__call__(*args, **kwargs)
        return self._process_tokenizer_output(
            res_batch, add_extra_mask_token=add_extra_mask_token
        )
