<p align="center">
  <img src="assets/github_banner.png" alt="Python Multi-Objective Optimization Framework" width="100%">
</p>

# Python Multi-Objective Optimization Framework

A Python-based framework for solving **multi-objective optimization** problems with evolutionary algorithms.

This repository demonstrates the framework through a **cooperative maritime patrol planning** problem involving UAVs, USVs, and patrol vessels. The project focuses on optimization modeling, algorithm comparison, performance evaluation, and visualization of Pareto-optimal solutions.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue">
  <img alt="Optimization" src="https://img.shields.io/badge/Topic-Multi--Objective%20Optimization-informational">
  <img alt="Algorithms" src="https://img.shields.io/badge/Algorithms-GA%20%7C%20NSGA--III%20%7C%20SMS--EMOA%20%7C%20ES-success">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

---

## Highlights

| Area | Description |
|---|---|
| Optimization | Multi-objective optimization for route planning and resource allocation |
| Algorithms | GA, NSGA-III, SMS-EMOA, Evolutionary Strategy |
| Evaluation | Hypervolume (HV), IGD+, C-metric |
| Visualization | Pareto fronts, patrol routes, convergence curves |
| Experiment Design | Multi-scenario and multi-seed simulation workflow |

---

## Framework Overview

<p align="center">
  <img src="assets/framework_overview.png" alt="Framework overview" width="85%">
</p>

The framework contains four main stages:

1. **Environment construction**: scenario maps, patrol zones, bases, and constraints  
2. **Optimization algorithms**: evolutionary multi-objective search  
3. **Performance evaluation**: Pareto front quality and algorithm comparison  
4. **Visualization**: routes, fronts, convergence, and metric analysis  

---

## Repository Structure

```text
.
├── src/                  # Core implementation and experiment scripts
├── data/                 # Example map and scenario data
├── figures/              # Selected result figures
├── public_notes/         # Public audit and excluded-file notes
├── assets/               # README banner and framework diagram
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Implemented Algorithms

| Algorithm | Purpose |
|---|---|
| Genetic Algorithm (GA) | Baseline evolutionary optimization |
| NSGA-III | Many-objective evolutionary optimization |
| SMS-EMOA | Hypervolume-oriented evolutionary search |
| Evolutionary Strategy (ES) | Baseline evolutionary strategy comparison |

All algorithms are evaluated under consistent experiment settings for fair comparison.

---

## Example Results

### Scenario Overview

<p align="center">
  <img src="figures/scenarios_preview.png" alt="Scenario preview" width="85%">
</p>

### Pareto Front

<p align="center">
  <img src="figures/pareto_front.png" alt="Pareto front" width="75%">
</p>

### Patrol Route Examples

| Objective | Example |
|---|---|
| Best F1-oriented route | <img src="figures/path_bestF1.png" width="420"> |
| Best F2-oriented route | <img src="figures/path_bestF2.png" width="420"> |
| Best F3-oriented route | <img src="figures/path_bestF3.png" width="420"> |
| Non-dominated route example | <img src="figures/path_nd.png" width="420"> |

### Performance Evaluation

<p align="center">
  <img src="figures/convergence.png" alt="Convergence" width="80%">
</p>

<p align="center">
  <img src="figures/cmp_hv_curve.png" alt="HV curve" width="80%">
</p>

---

## Performance Metrics

The framework uses common indicators for multi-objective optimization:

- **Hypervolume (HV)**: measures the dominated objective-space volume
- **IGD+**: measures distance to the reference Pareto front
- **C-metric**: compares dominance relationships between algorithms

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main experiment script:

```bash
python src/experiment.py
```

> The public repository is prepared as a portfolio version. Some large experiment outputs, cache files, and raw data mirrors are intentionally excluded to keep the repository lightweight.

---

## Research Background

This project was developed for a master's research project on **multi-objective cooperative maritime patrol planning**.

The optimization task considers multiple objectives such as patrol coverage, UAV operational cost, and cooperative efficiency among heterogeneous vehicles. The system is designed to compare multiple evolutionary algorithms under different scenario settings.

---

## Future Work

- Improve modularity of algorithm components
- Add simplified demo configuration for quick execution
- Add benchmark problems for general multi-objective optimization
- Extend visualization and reporting utilities
- Add unit tests and continuous integration workflow

---

## License

This project is released under the MIT License.
