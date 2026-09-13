# Export Validation

- Tested staging ZIP: `staging-layerloop-20260912-v3.zip`
- Staging ZIP SHA256: `d721dd933ff4d68beb4c078d81d4b6f4447adc651acb15772e60fc84085ff481`
- CPU validation: 87 pytest cases passed, no skips; actual decoder completed boundary, tail and ngram routes on a random four-layer Llama. The accompanying JSON binds source, scripts, configs and tests to hashes.
- Runtime: inherited the existing otv311 environment through a separate temporary venv. Installed only pytest and its test dependencies into that venv. Every model test used `CUDA_VISIBLE_DEVICES=""`.
- Packaging: complete framework wheel built successfully with `pip wheel --no-deps`; this does not certify a fresh installation of the full dependency lock.
- Assets: all 8 inference files in `checkpoint_assets.json` matched their recorded SHA256 on the existing server.
- Decode preflight: actual GSM8K file contained 1319 rows; requested physical rows 8..391 contained 384 unique questions. Full JSONL SHA256: `5a1593e09684fa25177dcf1835e7cdbae2160bb5a1da38acfcd8e9876962fa8d`.
- Training preflight: boundary continuation with existing checkpoint-60375 and `--steps 1 --dry-run` produced `max_steps: 60376` and portable output/data registration paths. No training process was launched.
- Scope limits: no 8B GPU execution, full training, fresh CUDA installation or new paper performance measurement was performed for this export. Model weights and datasets are external assets, not embedded in the code archive.
