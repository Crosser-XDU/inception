#!/usr/bin/env bash
# Source this file on the existing experiment server. It starts nothing.
export PYTHON=/data/home/zhuangz/workplace/2026new/OTV/.mamba_root/envs/otv311/bin/python
export MODEL=/data/pretrained_model/huggingface_model/meta-llama/Meta-Llama-3-8B-Instruct
export CHECKPOINT=/data/home/zhuangz/workplace/2026new/MetaMathQA/recurft_math_exp/outputs/llama3_q3_stage1_boundary_only_continue_to_1000_20260711/checkpoint-60375
export DATA=/data/home/zhuangz/workplace/2026new/MetaMathQA/recurft_math_exp/data/gsm8k_test_full_1319.jsonl
# GPU and OUT intentionally have no defaults; choose them explicitly.
