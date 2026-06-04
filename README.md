# GNNxEval

**Design and Implementation of a Framework for Graph Neural Network Explainer Evaluation**

GNNxEval is a Flask-based application for evaluating Graph Neural Network (GNN) explainers. The system provides a framework for systematically assessing the quality and effectiveness of GNN explanation methods [web:37][web:40][web:41].

---

## 📋 Overview

This project is the implementation of an academic thesis focused on evaluating explainability methods for Graph Neural Networks. The application runs without requiring GPU support and can be executed on standard hardware [web:36][web:41].

---

## 🚀 Getting Started

### Quick Start

Simply run the application:

```bash
python gnnxeval.py
```

The Flask application starts immediately and does not require a GPU to run [web:36][web:41].

---

## 🛠️ Installation & Setup

### Conda Environment (Recommended)

The project was developed and configured within a **Conda environment**, which is the recommended setup for reproducing the original runtime configuration [web:36][web:42].

1. Create and activate the Conda environment (environment file should be provided separately):
   ```bash
   conda env create -f environment.yml
   conda activate gnnxeval
   ```

2. The Conda environment includes both CPU and GPU-related dependencies, making it suitable for experiments requiring either configuration [web:36].

### Alternative: requirements.txt

A `requirements.txt` file is included for convenience and reference. However, please note:

- The project was primarily configured in a Conda environment
- `requirements.txt` is **optional** for this setup and provided mainly for reference [web:36][web:42][web:45]
- For full reproducibility, the Conda environment should be used

To install from `requirements.txt` if needed:

```bash
pip install -r requirements.txt
```

---

## 📊 Manuscript Reference

The following table from the thesis manuscript is applicable to this implementation:

![Figure from Manuscript](https://github.com/affanahmed373/GNNxEval/assets/56910741/07e5d049-26fa-4b90-aa94-b45679d54919)

This visualization demonstrates the evaluation framework architecture and methodology described in the thesis [web:40][web:43].

---

## 🧪 Features

- **GPU-Free Execution**: The application runs on standard hardware without GPU requirements [web:36][web:41]
- **GNN Explainer Evaluation**: Comprehensive framework for assessing Graph Neural Network explanation methods [web:35][web:41]
- **Flask-Based Interface**: Web application interface for interactive evaluation [web:36]
- **Academic Implementation**: Direct implementation of thesis research methodology [web:37][web:40]

---

## 📁 Project Structure

```
GNNxEval/
├── gnnxeval.py          # Main Flask application entry point
├── requirements.txt      # Optional Python dependencies (for reference)
├── environment.yml       # Conda environment configuration (recommended)
├── README.md             # This file
└── [other project files]
```

---

## 🔬 Research Context

This project implements research from the thesis:

> **"Design and Implementation of a Framework for Graph Neural Network Explainer Evaluation"**

The framework addresses the systematic evaluation of explainability methods for GNNs, building on recent work in graph neural network interpretability [web:35][web:41][web:44].

---

## 📚 Dependencies

- **Flask**: Web application framework [web:36]
- **Python**: Programming language
- **Conda**: Environment management (recommended)
- **GPU dependencies**: Included in Conda environment (optional for execution)

---

## 🤝 Citation

If you use this project in your research, consider citing the associated thesis:

```
Affan Ahmed. "Design and Implementation of a Framework for Graph Neural Network Explainer Evaluation." [Year].
```

---

## 📬 Contact

For questions or feedback related to this thesis implementation, please reach out via the GitHub repository or associated academic contact.

---



**Built as part of academic research on Graph Neural Network explainability** [web:35][web:41]
