# Multi-Objective Optimization Framework

A **reusable Python framework** for **multi-objective optimization** using evolutionary algorithms.

The framework is demonstrated through a **cooperative maritime patrol planning** case study involving UAVs, USVs and patrol vessels.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Algorithms](https://img.shields.io/badge/Algorithms-GA%20%7C%20NSGA--III%20%7C%20SMS--EMOA%20%7C%20ES-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Project Highlights

- Modular multi-objective optimization framework
- Genetic Algorithm (GA)
- NSGA-III
- SMS-EMOA
- Evolutionary Strategy (ES)
- Hypervolume (HV), IGD+, and C-metric evaluation
- Real-world maritime patrol planning case study

---

## Framework Overview

![Framework](figures/flowchart.png)

The framework consists of:

1. Environment Construction
2. Evolutionary Optimization
3. Performance Evaluation
4. Result Visualization

---

## Repository Structure

```text
.
├── src/
├── data/
├── figures/
├── assets/
├── public_notes/
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Implemented Algorithms

| Algorithm | Description |
|-----------|-------------|
| GA | Genetic Algorithm |
| NSGA-III | Many-objective evolutionary optimization |
| SMS-EMOA | Hypervolume-based evolutionary optimization |
| ES | Evolutionary Strategy |

---

# Representative Experimental Results

## Representative Patrol Routes

![Representative Patrol Routes](figures/representative_patrol_routes.png)

This figure shows representative solutions optimized for different objectives (coverage, UAV cost and cooperation distance) under the Taiwan AIS scenario.

---

## Hypervolume Convergence

![Hypervolume Convergence](figures/hv_convergence.png)

The convergence curves summarize the mean and standard deviation over 30 independent runs.

---

## Pareto Front

![Pareto Front](figures/pareto_front.png)

The Pareto front illustrates the trade-offs among conflicting optimization objectives.

---

## Performance Summary

![Performance Summary](figures/metric_summary.png)

Performance is evaluated using:

- Hypervolume (HV)
- IGD+
- C-metric

---

## Quick Start

```bash
pip install -r requirements.txt
python src/experiment.py
```

---

## Research Background

This repository focuses on a reusable optimization framework. Maritime patrol planning is presented as a representative application rather than the only target problem.

---

## Future Work

- Additional evolutionary algorithms
- More benchmark optimization problems
- CI/CD support
- Unit testing
- Improved visualization

---

## License

Released under the MIT License.
