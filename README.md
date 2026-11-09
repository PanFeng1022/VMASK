# VMASK

Official implementation for our work under review at IEEE Transactions on
Information Forensics and Security (TIFS).

This repository provides the VMASK source code, experimental configurations,
and execution scripts. VMASK mode uses the complete native Threshold SNIP path
for proof generation, verification, admission, and aggregate recovery.

## Repository Contents

- `src/vmask_exp/`: VMASK and learning implementation.
- `scripts/`: experiment, benchmark, and summarization entry points.
- `configs/`: experimental configurations.
- `tests/`: protocol and regression tests.
- `requirements.txt`: Python dependencies.

## Environment

Use Python 3.11:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Verification

```bash
python -m unittest discover -s tests -v
```
