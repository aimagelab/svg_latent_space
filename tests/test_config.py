"""Asset macro resolution and self-contained checkpoint config paths."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sls.runtime import read_config


class ConfigTest(unittest.TestCase):
    def test_shared_asset_macro(self):
        config = Path(__file__).resolve().parents[1] / "configs/sls.yaml"
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"SLS_ASSETS_DIR": folder}):
                cfg = read_config(config)
            self.assertEqual(
                cfg["model"]["ckpt_path"], str(Path(folder) / "weights/sls")
            )
            self.assertEqual(cfg["tokenizer"]["path"], str(Path(folder) / "tokenizer"))

    def test_missing_macro_is_reported(self):
        config = Path(__file__).resolve().parents[1] / "configs/sls.yaml"
        with patch.dict(os.environ, {"SLS_ASSETS_DIR": ""}):
            with self.assertRaisesRegex(ValueError, "SLS_ASSETS_DIR"):
                read_config(config)

    def test_final_bundle_needs_no_macro(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "release_config.yaml"
            config.write_text("model:\n  ckpt_path: .\ntokenizer:\n  path: .\n")
            with patch.dict(os.environ, {"SLS_ASSETS_DIR": ""}):
                cfg = read_config(config)
            self.assertEqual(cfg["model"]["ckpt_path"], folder)
            self.assertEqual(cfg["tokenizer"]["path"], folder)


if __name__ == "__main__":
    unittest.main()
