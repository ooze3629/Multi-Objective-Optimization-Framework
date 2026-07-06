# Python Multi-Objective Optimization Framework

A Python framework for solving **multi-objective optimization** problems using evolutionary algorithms.

This project demonstrates a **cooperative maritime patrol planning** problem involving UAVs, USVs, and patrol vessels. The framework supports multiple optimization algorithms, automated experiment management, performance evaluation, and visualization of Pareto-optimal solutions.

---

## Features

- Multi-objective optimization framework
- Genetic Algorithm (GA)
- NSGA-III
- SMS-EMOA
- Evolutionary Strategy (ES)
- Cooperative maritime patrol planning
- Automated experiment management
- Pareto front evaluation
- Hypervolume (HV), IGD+, and C-metric analysis
- Result visualization

---

## Repository Structure

```text
.
├── src/                  # Core source code
├── data/                 # Example maps and scenario data
├── figures/              # Sample figures for visualization
├── public_notes/         # Public project notes
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Framework

The framework consists of four major components:

1. Environment Construction
2. Multi-objective Optimization
3. Performance Evaluation
4. Visualization

The optimization process can be summarized as:

```
Environment
      │
      ▼
Optimization Algorithms
      │
      ▼
Performance Evaluation
      │
      ▼
Visualization
```

---

## Implemented Algorithms

- Genetic Algorithm (GA)
- NSGA-III
- SMS-EMOA
- Evolutionary Strategy (ES)

The framework provides a unified experimental workflow, allowing different optimization algorithms to be evaluated under identical problem settings.

---

## Performance Evaluation

The framework includes several widely used multi-objective performance indicators:

- Hypervolume (HV)
- IGD+
- C-metric

These metrics are used to compare solution quality among different optimization algorithms.

---

## Quick Start

Install the required packages:

```bash
pip install -r requirements.txt
```

Run an experiment:

```bash
python src/experiment.py
```

---

## Example Results

The repository contains example figures demonstrating:

- Patrol routes
- Pareto fronts
- Performance comparison
- Convergence behavior

See the **figures/** directory for sample outputs.

---

## Research Background

This project was developed as part of a master's research on **multi-objective cooperative maritime patrol planning**.

The objective is to optimize patrol coverage, operational cost, and cooperative efficiency for heterogeneous vehicles using evolutionary multi-objective optimization algorithms.

---

## License

This project is released under the MIT License.