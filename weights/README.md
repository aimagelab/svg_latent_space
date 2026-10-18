In this version, setting `SLS_ASSETS_DIR` is required for training and inference
with the default config. Set it to the asset folder, which contains:

```text
sls_release/
  weights/sls/model.safetensors
  tokenizer/tokenizer.model
  tokenizer/tokenizer.vocab
```

The default config resolves both weights and tokenizer using `${SLS_ASSETS_DIR}`.
Export the variable in each new terminal session; see the setup command in the
repository README.
The model filename is `model.safetensors`. Its directory can also be overridden
with `inference.py --checkpoint DIRECTORY`.

The checkpoint is about 557 MiB and is excluded from Git. See
`assets_manifest.json` at the repository root for checksums and exact sizes.
