"""CPU checks for padding, training gradients and portable checkpoints."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from sls.data import PathCollator
from sls.model import SLS
from sls.runtime import encode_paths
from sls.tokenizer import MySPTokenizer
from sls.transforms import SwitchStyle

ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = MySPTokenizer.from_pretrained(ROOT / "tokenizer")
        cls.config = dict(
            d_model=32,
            nhead=4,
            num_layers=1,
            num_tokens=448,
            dim_feedforward=64,
            cls_mode="last",
            noise=1.0,
        )
        cls.paths = [
            '<path style="fill:rgb(0,0,0);" d="M0,0L5,5Z"/>',
            '<path style="fill:rgb(255,0,0);" d="M0,0C1,2,3,4,5,6L8,9Z"/>',
        ]

    def test_padding_does_not_change_path_embedding(self):
        model = SLS(**self.config).eval()
        one = encode_paths(self.paths[:1], model, self.tokenizer)
        batch = encode_paths(self.paths, model, self.tokenizer)
        torch.testing.assert_close(one[0], batch[0], atol=1e-5, rtol=1e-5)

    def test_gradients_and_checkpoint_roundtrip(self):
        model = SLS(**self.config).train()
        batch = PathCollator(self.tokenizer)(
            [{"text": "<s>" + p + "</s>"} for p in self.paths]
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.encoder.embedding_head.weight.detach().clone()
        loss = model(**batch)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(
            model.encoder.embedding_head.weight.grad.abs().sum().item(), 0
        )
        self.assertGreater(
            model.decoder.embedding_head.weight.grad.abs().sum().item(), 0
        )
        optimizer.step()
        self.assertFalse(torch.equal(before, model.encoder.embedding_head.weight))
        model.eval()
        expected = encode_paths(self.paths, model, self.tokenizer)
        with tempfile.TemporaryDirectory() as folder:
            save_file(model.state_dict(), str(Path(folder) / "model.safetensors"))
            self.tokenizer.save_pretrained(folder)
            tokenizer = MySPTokenizer.from_pretrained(folder)
            loaded = SLS(**self.config, ckpt_path=folder).eval()
            torch.testing.assert_close(
                expected, encode_paths(self.paths, loaded, tokenizer)
            )

    def test_style_augmentation_keeps_geometry(self):
        from unittest.mock import patch

        with patch("sls.transforms.random.random", return_value=0):
            output = SwitchStyle()(self.paths[0])
        self.assertIn('d="M0,0L5,5Z"', output)
        self.assertLess(output.index('d="'), output.index('style="'))


if __name__ == "__main__":
    unittest.main()
