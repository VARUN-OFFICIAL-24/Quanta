# 🧠 Pattern-Aware Quantum Circuit Optimization

**Graph-Based Bayesian Optimization with Fidelity Constraints**

An AI-driven framework for automated quantum circuit optimization, designed to simplify and improve small-scale (5–6 qubit) quantum circuits while provably preserving quantum functionality.

This project demonstrates how graph-based representations, uncertainty-aware Bayesian optimization, and quantum-aware pattern reasoning can be combined to optimize circuit structure without manual design or brute-force search.

---

## 📌 Inspiration

This work is inspired by the research article:

**"Graph-Based Bayesian Optimization for Quantum Circuit Architecture Search with Uncertainty-Calibrated Surrogates"**  
*Authors: Prashant Kumar Choudhary, Nouhaila Innan, Muhammad Shafique, Rajeev Singh*

Reading this paper motivated the construction of a correctness-first, end-to-end optimization system that not only searches for efficient quantum circuits, but also verifies functional equivalence rigorously.

This project is an **independent implementation and extension**, built for experimentation, validation, and deeper technical understanding.

---

## 🚀 Core Idea

Instead of manually engineering or rewriting quantum circuits, the system:

- **Represents quantum circuits as qubit-aware dependency graphs**
- **Learns structure–performance relationships** using Graph Neural Networks (GNNs)
- **Applies Bayesian Optimization** with Expected Improvement
- **Uses pattern-aware structural mutations** to remove redundancy
- **Enforces multi-sample fidelity constraints** to guarantee correctness

Given a complex or inefficient circuit as input, the system **autonomously discovers a simpler circuit** that produces the same quantum state.

---

## 🔬 Scope of the Project

| Aspect | Details |
|--------|---------|
| **Qubit Range** | 5–6 qubits |
| **Execution** | CPU-only (Google Colab / local machines) |
| **Optimization Focus** | Circuit structure (not parameter training) |
| **Correctness** | Verified via multi-sample statevector fidelity |
| **Goal** | Reduce gate count, depth, and two-qubit operations safely |

The scope is intentionally chosen where simulation is feasible and structural learning is meaningful for NISQ-era systems.

---

## ⚙️ System Overview

```
User-Defined Quantum Circuit
            ↓
Qubit-Aware Graph Representation
            ↓
Graph Neural Network Surrogate
            ↓
Bayesian Optimization (Expected Improvement)
            ↓
Pattern-Aware Structural Mutation
            ↓
Multi-Sample Fidelity Validation
            ↓
Optimized Quantum Circuit
```

Invalid candidates are explicitly rejected, ensuring correctness throughout optimization.

---

## ✅ Correctness & Reliability (Key Improvements)

This implementation fixes several critical issues commonly found in circuit optimization pipelines:

### Multi-Sample Fidelity Verification
Functional equivalence is tested across **multiple random parameter settings**, not a single point.

### Angle-Aware Rotation Merging
Rotation gates are merged by **summing angles**, preserving the exact unitary transformation.

### Independent Framework Validation
Final circuits are verified using **Qiskit statevector fidelity**, confirming correctness beyond PennyLane.

These safeguards ensure that optimization **never alters the circuit's quantum behavior**.

---

## 📊 Example Result (5-Qubit Circuit)

| Metric | Before Optimization | After Optimization |
|--------|---------------------|-------------------|
| **Qubits** | 5 | 5 |
| **Gate Count** | 16 | 12 |
| **Circuit Depth** | 6 | 4 |
| **CNOT Gates** | 6 | 4 |
| **Gate Reduction** | — | **25%** |
| **Average Fidelity** | 1.000 | ≈ 1.000 |

✔ Redundant entanglement removed  
✔ Circuit depth reduced  
✔ Quantum functionality preserved

---

## 🛠️ Technologies Used

- **Quantum Simulation:** PennyLane
- **Machine Learning:** PyTorch, PyTorch Geometric
- **Optimization:** Bayesian Optimization (Expected Improvement)
- **Graph Processing:** NetworkX
- **Validation:** Qiskit (Statevector Fidelity)
- **Execution Environment:** Google Colab (CPU)

---

## ▶️ Run this Project on Google Colab

**Latest Updated Notebook (Corrected & Verified):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Amn6WTl32qHGK6QfQ7y2hzfLlpSfgIAK?usp=sharing)

The notebook:

- ✅ Installs all dependencies automatically
- ✅ Runs a full optimization example
- ✅ Visualizes convergence and Pareto behavior
- ✅ Exports results and plots

---

## 📌 What This Project Is — and Is Not

### ✔ What it is

- A **correctness-first** quantum circuit optimizer
- A **research-faithful**, reproducible prototype
- A strong **educational and experimental** framework

### ✖ What it is not

- A hardware-noise-aware compiler
- A proof of global optimality
- A large-scale (>10 qubit) optimizer

---

## 🧭 Why This Matters

As quantum hardware remains **noisy and resource-constrained**, circuit efficiency is critical.

This project demonstrates how **AI + physics-aware reasoning** can safely automate circuit optimization without breaking correctness.

---
