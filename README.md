# VMASK Experiments and Results

This repository provides the source code, experimental configurations, scripts,
and results used in the evaluation of VMASK.

The evaluation covers benign utility on MNIST and CIFAR-10, AMM mitigation,
computation and communication costs, and the scalability comparison reported
in the paper.

## Repository Contents

- `src/`: implementation used in the evaluation.
- `scripts/`: experiment, benchmark, summarization, and plotting scripts.
- `configs/`: configurations for the reported experiments.
- `results.7z`: raw results, summaries, tables, and figures.
- `requirements.txt`: Python dependencies.

## External Baseline

PriRoAgg is evaluated using its
[public implementation](https://github.com/tardisblue9/PriRoAgg) at commit
`33dbcd6ec89a2412159a8404417ca2af984be508`. Place the checked-out repository
at `external/PriRoAgg` before running the PriRoAgg benchmark path.
