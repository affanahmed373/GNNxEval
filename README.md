# GNNxEval

**Design and Implementation of a Framework for Graph Neural Network Explainer Evaluation**

GNNxEval is a Streamlit-based interactive application for evaluating Graph Neural Network (GNN) explainers. It provides a framework for systematically assessing the quality and effectiveness of GNN explanation methods using established metrics.

---

## Overview

This project is the implementation of a master's thesis focused on evaluating explainability methods for Graph Neural Networks. Users can select a dataset, GNN architecture, and explainer algorithm, then compute evaluation metrics through an interactive web interface. The application runs without requiring GPU support.

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Conda](https://docs.conda.io/) (recommended) or pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/affanahmed373/GNNxEval.git
   cd GNNxEval
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision
   pip install torch-geometric
   pip install streamlit matplotlib numpy pandas scikit-learn
   ```

### Running the Application

```bash
streamlit run gnnxeval.py
```

The app will open in your browser where you can select a dataset, model, and explainer to evaluate.

---

## Features

- **Interactive UI**: Streamlit-based web interface for selecting datasets, models, and explainers
- **4 GNN Architectures**: GCN, GAT, GIN, GraphSAGE
- **2 Explainer Algorithms**: GNNExplainer, GraphMaskExplainer
- **3 Evaluation Metrics**: Fidelity (positive/negative), Characterization Score, Unfaithfulness
- **3 Benchmark Datasets**: Cora, CiteSeer, Pubmed (Planetoid)
- **GPU-Free Execution**: Runs on standard hardware without GPU requirements

---

## Project Structure

```
GNNxEval/
├── gnnxeval.py            # Main Streamlit application
├── gnnexp.ipynb           # GNNExplainer experiments notebook
├── graphmask.ipynb        # GraphMaskExplainer experiments notebook
├── captum.ipynb           # CaptumExplainer experiments notebook
├── comparison.ipynb       # Metric comparison visualizations
├── data/Planetoid/        # Cora, CiteSeer, Pubmed datasets
├── requirements.txt       # Conda environment export (reference)
├── geo.txt                # Pip freeze output (reference)
└── README.md
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Fidelity** | Measures how faithfully the explanation reflects the model's decision-making (positive and negative) |
| **Characterization Score** | Combined fidelity metric balancing positive and negative fidelity |
| **Unfaithfulness** | Quantifies the unreliability of the generated explanation |

---

## Manuscript Reference

The following table from the thesis manuscript is applicable to this implementation:

![Figure from Manuscript](https://github.com/affanahmed373/GNNxEval/assets/56910741/07e5d049-26fa-4b90-aa94-b45679d54919)

---

## Tech Stack

- **Streamlit** — Interactive web UI
- **PyTorch** — Deep learning framework
- **PyTorch Geometric (PyG)** — Graph neural network library and explainability tools
- **Matplotlib** — Visualization

---

## Citation

If you use this project in your research, please cite:

```
Affan Ahmed. "Design and Implementation of a Framework for Graph Neural Network Explainer Evaluation." Master's Thesis, 2023.
```

---

## Contact

For questions or feedback, please open an issue on the [GitHub repository](https://github.com/affanahmed373/GNNxEval).
