<p align="center">
  <img src="assets/github_banner.png" width="100%" alt="Project Banner">
</p>

# Multi-Objective Optimization Framework

A **reusable Python framework** for **multi-objective optimization** using evolutionary algorithms, demonstrated through a **cooperative maritime patrol planning** case study.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Algorithms](https://img.shields.io/badge/Algorithms-GA%20%7C%20NSGA--III%20%7C%20SMS--EMOA-success)
![Optimization](https://img.shields.io/badge/Multi--Objective-Optimization-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</p>

---

## Overview

This repository presents a modular optimization framework for solving complex multi-objective optimization problems.

The current implementation focuses on **cooperative maritime patrol planning** involving UAVs, USVs, and patrol vessels. Although developed for this application, the optimization workflow is designed to be reusable for other multi-objective optimization problems.

---

## Project Highlights

| Feature | Description |
|----------|-------------|
| Multi-objective Optimization | Simultaneous optimization of multiple conflicting objectives |
| Evolutionary Algorithms | GA, NSGA-III, SMS-EMOA and Evolutionary Strategy |
| Evaluation | Hypervolume (HV), IGD+, and C-metric |
| Visualization | Patrol routes, Pareto fronts and convergence curves |
| Research Scale | Multi-country, multi-scenario, multi-seed experiments |

---

## Framework Architecture

<p align="center">
<img src="assets/framework_overview.png" width="85%">
</p>

The framework consists of four major stages:

1. Environment Construction
2. Evolutionary Optimization
3. Performance Evaluation
4. Result Visualization

---

## Repository Structure

```text
.
├── src/                  Core implementation
├── data/                 Example datasets and maps
├── figures/              Result figures
├── assets/               README resources
├── public_notes/         Public project notes
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Implemented Algorithms

| Algorithm | Purpose |
|------------|---------|
| Genetic Algorithm (GA) | Baseline evolutionary optimization |
| NSGA-III | Many-objective optimization |
| SMS-EMOA | Hypervolume-based optimization |
| Evolutionary Strategy (ES) | Baseline evolutionary strategy |

---

## Example Results

### Scenario Overview

<p align="center">
<img src="figures/scenarios_preview.png" width="85%">
</p>

### Pareto Front

<p align="center">
<img src="figures/pareto_front.png" width="75%">
</p>

### Representative Patrol Routes

| Objective | Visualization |
|-----------|---------------|
| Coverage-oriented (F1) | <img src="figures/path_bestF1.png" width="360"> |
| Cost-oriented (F2) | <img src="figures/path_bestF2.png" width="360"> |
| Cooperation-oriented (F3) | <img src="figures/path_bestF3.png" width="360"> |
| Pareto Solution | <img src="figures/path_nd.png" width="360"> |

### Performance Evaluation

<p align="center">
<img src="figures/convergence.png" width="80%">
</p>

<p align="center">
<img src="figures/cmp_hv_curve.png" width="80%">
</p>

---

## Performance Metrics

The framework evaluates optimization quality using:

- **Hypervolume (HV)**
- **IGD+**
- **C-metric**

These metrics enable consistent comparison across different evolutionary algorithms.

---

## Quick Start

```bash
pip install -r requirements.txt
python src/experiment.py
```

---

## Project Status

- Active portfolio project
- Research-oriented optimization framework
- Modular algorithm implementation
- Suitable for extension to other optimization problems

---

## Future Work

- Additional optimization algorithms
- More benchmark problems
- Improved visualization modules
- Unit testing and CI/CD support

---

## License

Released under the MIT License.
