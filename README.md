# Python Multi-Objective Optimization Framework

A Python portfolio project for multi-objective optimization and cooperative path planning.

This repository demonstrates an optimization framework using a maritime patrol planning scenario with multiple vehicle types, including UAVs, USVs, and patrol vessels. The project focuses on building configurable simulation environments, comparing evolutionary optimization algorithms, and evaluating Pareto-front quality with standard multi-objective metrics.

## Project Highlights

- Python-based optimization framework for cooperative path planning
- Multi-objective optimization with evolutionary algorithms
- Support for multiple map environments and mission-weight settings
- Pareto-front evaluation using Hypervolume (HV), IGD+, and C-metric
- Visualization of route plans, Pareto fronts, convergence curves, and metric comparisons
- Large-scale experimental workflow across multiple scenarios, algorithms, and random seeds

## Algorithms

The repository includes implementations or experiment integrations for:

- Proposed MOGA / steady-state selection workflow
- Genetic Algorithm (GA)-based components
- NSGA-III
- SMS-EMOA
- Evolutionary Strategy (ES)
- Random and greedy baselines

## Repository Structure

```text
.
├── code/                 # Core Python implementation and experiment scripts
├── data/                 # Lightweight map and scenario arrays for demonstration
├── figures/              # Selected output figures for portfolio display
├── README.md             # Project overview
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Development dependencies
├── verify.py             # Quick verification entry point
└── LICENSE               # MIT License
```

## Example Outputs

### Optimization workflow

![Flowchart](figures/flowchart.png)

### Pareto-front example

![Pareto Front](figures/pareto_front.png)

### Route-planning examples

![Best F1 Path](figures/path_bestF1.png)

![Best F2 Path](figures/path_bestF2.png)

![Best F3 Path](figures/path_bestF3.png)

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a quick verification:

```bash
python verify.py quick
```

Run a small experiment:

```bash
python code/experiment.py --env taiwan --tier LOWER --fes 3000 --pop 100 --seeds 10
```

> Full-scale experiments can be computationally expensive. This public version keeps selected lightweight inputs and figures for demonstration. Large checkpoints, caches, logs, and full result archives are excluded.

## Evaluation Metrics

- **Hypervolume (HV)**: measures the dominated objective-space volume.
- **IGD+**: measures the distance from an approximation set to a reference Pareto set.
- **C-metric**: compares pairwise domination coverage between methods.

## Scope of This Public Version

This repository is a public portfolio version of a graduate research project. It keeps the core implementation, lightweight scenario inputs, and selected result figures. It excludes large experiment archives, checkpoints, temporary files, local logs, route-cache files, and raw data mirrors.

## Tech Stack

- Python
- NumPy
- SciPy
- Matplotlib
- Pillow

## Author

Kuan-Ting Liu

## License

This project is released under the MIT License.
