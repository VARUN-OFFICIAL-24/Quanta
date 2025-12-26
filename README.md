# 🧠 Pattern-Aware Quantum Circuit Optimization

An **AI-driven framework for automated quantum circuit optimization**, designed to intelligently simplify and improve **5–6 qubit quantum circuits** under realistic computational constraints.

This project demonstrates how **graph-based learning**, **Bayesian optimization**, and **pattern-aware reasoning** can be combined to discover efficient quantum circuit architectures without manual design or brute-force search.

---

## 📌 Inspiration

This work is inspired by the research article:

**“Graph-Based Bayesian Optimization for Quantum Circuit Architecture Search with Uncertainty-Calibrated Surrogates”**  
**Authors:** Prashant Kumar Choudhary, Nouhaila Innan, Muhammad Shafique, Rajeev Singh

Reading this paper sparked the idea of building a **resource-aware, end-to-end system** that could automatically improve quantum circuits by learning from structure, uncertainty, and past performance.

This project is an **independent implementation and extension**, created for experimentation and deeper understanding.

---

## 🚀 Core Idea

Instead of manually engineering quantum circuits, the system:

- Treats quantum circuits as **graphs**
- Learns structure–performance relationships using **Graph Neural Networks (GNNs)**
- Applies **Bayesian Optimization** to efficiently explore the design space
- Uses **pattern-aware structural mutations** to remove redundancy
- Optimizes under a **performance–complexity tradeoff**

Given a complex and inefficient circuit as input, the system autonomously produces a **simpler, higher-performing architecture**.

---

## 🔬 Scope of the Project

- **Qubit Range:** 5–6 qubits  
- **Execution:** CPU-only (Google Colab / local machines)  
- **Optimization Focus:** Circuit structure (not parameter tuning)  
- **Evaluation:** Complexity-aware objective with uncertainty handling  

The framework is intentionally scoped to small-to-medium qubit systems where **simulation is feasible** and **structural learning is meaningful**.

---

## ⚙️ System Overview

    User-Defined Circuit
            ↓
    Graph Representation
            ↓
    GNN Surrogate Model
            ↓
    Bayesian Optimization (Uncertainty-Aware)
            ↓
    Pattern-Based Structural Mutation
            ↓
    Optimized Quantum Circuit


The optimization loop continuously learns from previous circuits, improving efficiency over time.

---

## 📊 Example Result (5-Qubit)

| Metric | Before Optimization | After Optimization |
|------|--------------------|-------------------|
| Qubits | 5 | 5 |
| Gate Count | 16 | 4 |
| Redundant Operations | Yes | No |
| Circuit Depth | High | Minimal |
| Objective Score | Lower | Higher |

The optimizer identified a **minimal effective subspace**, removing unnecessary entanglement and rotations while improving performance under a complexity-aware objective.

---

## 🛠️ Technologies Used

- **Quantum Simulation:** PennyLane  
- **Machine Learning:** PyTorch, PyTorch Geometric  
- **Optimization:** Bayesian Optimization with uncertainty  
- **Graph Processing:** NetworkX  
- **Execution Environment:** Google Colab / CPU  

---

## ▶️ Run the Project

### 🔹 Google Colab (Recommended)

Run the full project in a read-only Colab environment:

[![Open in Colab]([https://colab.research.google.com/assets/colab-badge.sv](https://colab.research.google.com/drive/1Ya4yq0Im2N9dj6K4eNZumrmTMMdRxrm7?usp=sharing))](PASTE_YOUR_COLAB_LINK_HER)

---

### 🔹 Local Setup

```bash
pip install -r requirements.txt

