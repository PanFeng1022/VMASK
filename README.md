
# VMASK

Implementation of VMASK, a verifiable secure aggregation protocol for federated learning. The corresponding paper is under review at IEEE Transactions on Information Forensics and Security (TIFS).

## Repository Contents

- `src/vmask_exp/`: Protocol and federated learning implementations.
- `scripts/`: Scripts for experiments, benchmarks, and result analysis.
- `configs/`: Experimental configuration files.
- `tests/`: Protocol and regression tests.
- `requirements.txt`: Python dependencies.
- `pyproject.toml`: Project configuration and package metadata.

## Installation

Requires Python 3.11.

```bash
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Tests

```bash
python -m unittest discover -s tests -v
```
