# VMASK

This repository provides the research implementation and evaluation artifact for VMASK. It includes protocol components, federated learning workflows, experiment configurations, benchmark utilities, and validation tests.

## Repository Structure

- `src/vmask_exp/`: Core protocol and federated learning components.
- `scripts/`: Experiment, benchmark, and analysis utilities.
- `configs/`: Experiment and benchmark configurations.
- `tests/`: Protocol-level and regression tests.
- `requirements.txt`: Python dependencies.
- `pyproject.toml`: Package and build configuration.

## Setup

Create an isolated Python environment and install a PyTorch build compatible with the target platform. Then install the remaining dependencies and the VMASK package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Usage

Experiment and benchmark settings are provided in `configs/`. The corresponding entry points are available in `scripts/`.

## Validation

Run the test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```
