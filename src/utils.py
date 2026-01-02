# src/utils.py

import numpy as np
import pennylane as qml
import networkx as nx
import torch
from torch_geometric.data import Data

# -----------------------------
# Quantum Circuit Representation
# -----------------------------

class QuantumCircuit:
    """
    Gate format: (gate, wires, angle)
    angle=None for H, CNOT
    """
    def __init__(self, gates):
        self.gates = gates

    def copy(self):
        return QuantumCircuit(self.gates.copy())

    def __len__(self):
        return len(self.gates)

    def depth(self, n_qubits=5):
        times = {q: 0 for q in range(n_qubits)}
        for gate, wires, _ in self.gates:
            t = max(times[w] for w in wires)
            for w in wires:
                times[w] = t + 1
        return max(times.values())

    def count_cnot(self):
        return sum(1 for g, _, _ in self.gates if g == "CNOT")

# -----------------------------
# Graph Conversion
# -----------------------------

def circuit_to_graph(circuit, gate_set, n_qubits):
    G = nx.DiGraph()
    last_on_qubit = {}

    for i, (gate, wires, angle) in enumerate(circuit.gates):
        G.add_node(i, gate=gate, wires=wires, angle=angle)
        for q in wires:
            if q in last_on_qubit:
                G.add_edge(last_on_qubit[q], i)
            last_on_qubit[q] = i

    x = []
    for i in G.nodes:
        gate, wires, angle = (
            G.nodes[i]["gate"],
            G.nodes[i]["wires"],
            G.nodes[i]["angle"],
        )

        gate_vec = [1.0 if gate == g else 0.0 for g in gate_set]
        qubit_vec = [1.0 if q in wires else 0.0 for q in range(n_qubits)]
        angle_vec = [0.0 if angle is None else angle / (2 * np.pi)]

        x.append(gate_vec + qubit_vec + angle_vec)

    edge_index = list(G.edges)
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index)

# -----------------------------
# Circuit Simplification
# -----------------------------

def simplify_circuit(circuit):
    gates = circuit.gates.copy()
    changed = True

    while changed:
        changed = False
        i = 0
        while i < len(gates) - 1:
            g1, w1, a1 = gates[i]
            g2, w2, a2 = gates[i + 1]

            # Cancel self-inverse
            if g1 == g2 and w1 == w2 and g1 in ["H", "CNOT"]:
                gates.pop(i)
                gates.pop(i)
                changed = True
                continue

            # Merge rotations
            if g1 == g2 and g1 in ["RX", "RY", "RZ"] and w1 == w2:
                merged = ((a1 or 0) + (a2 or 0)) % (2 * np.pi)
                if abs(merged) < 1e-6:
                    gates.pop(i)
                    gates.pop(i)
                else:
                    gates[i] = (g1, w1, merged)
                    gates.pop(i + 1)
                changed = True
                continue

            i += 1

    return QuantumCircuit(gates)

# -----------------------------
# Fidelity Evaluation
# -----------------------------

def state_fidelity(circ_a, circ_b, n_qubits, samples=5):
    dev = qml.device("default.qubit", wires=n_qubits)

    def run(circuit, params):
        @qml.qnode(dev)
        def qnode():
            idx = 0
            for g, w, a in circuit.gates:
                if g in ["RX", "RY", "RZ"]:
                    theta = a if a is not None else params[idx]
                    getattr(qml, g)(theta, wires=w[0])
                    idx += 1
                elif g == "H":
                    qml.Hadamard(wires=w[0])
                elif g == "CNOT":
                    qml.CNOT(wires=w)
            return qml.state()
        return qnode()

    fidelities = []
    param_count = sum(1 for g, _, _ in circ_a.gates if g in ["RX", "RY", "RZ"])

    for _ in range(samples):
        params = np.random.uniform(0, 2*np.pi, param_count)
        ψ1 = run(circ_a, params)
        ψ2 = run(circ_b, params)
        fidelities.append(abs(np.vdot(ψ1, ψ2)) ** 2)

    return min(fidelities), float(np.mean(fidelities))
