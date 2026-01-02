# src/optimizer.py

import random
import numpy as np
import torch
from scipy.stats import norm

from src.utils import (
    QuantumCircuit,
    simplify_circuit,
    circuit_to_graph,
    state_fidelity,
)
from src.models import CircuitGNN

# -----------------------------
# Acquisition Function
# -----------------------------

def expected_improvement(mu, sigma, best, xi=0.01):
    if sigma < 1e-6:
        return 0.0
    Z = (mu - best - xi) / sigma
    return max(0.0, (mu - best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z))

# -----------------------------
# Mutation Operator
# -----------------------------

def mutate(circuit, gate_set, n_qubits):
    new = circuit.copy()

    if random.random() < 0.3 and len(new) > 1:
        new.gates.pop(random.randrange(len(new)))
    else:
        gate = random.choice(gate_set)
        if gate == "CNOT":
            wires = random.sample(range(n_qubits), 2)
            angle = None
        elif gate == "H":
            wires = [random.randrange(n_qubits)]
            angle = None
        else:
            wires = [random.randrange(n_qubits)]
            angle = np.random.uniform(0, 2*np.pi)

        new.gates.insert(
            random.randrange(len(new) + 1),
            (gate, wires, angle)
        )

    return simplify_circuit(new)

# -----------------------------
# Optimizer
# -----------------------------

class QuantumCircuitOptimizer:
    def __init__(
        self,
        gate_set,
        n_qubits,
        fidelity_threshold=0.95,
        iterations=40,
        candidates=10,
    ):
        self.gate_set = gate_set
        self.n_qubits = n_qubits
        self.fidelity_threshold = fidelity_threshold
        self.iterations = iterations
        self.candidates = candidates

    def optimize(self, circuit):
        population = [circuit]
        scores = []

        input_dim = len(self.gate_set) + self.n_qubits + 1
        model = CircuitGNN(input_dim)

        for _ in range(self.iterations):
            graphs = []
            y = []

            for c in population:
                g = circuit_to_graph(c, self.gate_set, self.n_qubits)
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
                graphs.append(g)

                fmin, favg = state_fidelity(circuit, c, self.n_qubits)
                if fmin < self.fidelity_threshold:
                    y.append(-100.0)
                else:
                    score = 10*favg - 0.1*len(c) - 0.2*c.count_cnot()
                    y.append(score)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            loss_fn = torch.nn.MSELoss()

            for _ in range(40):
                for g, t in zip(graphs, y):
                    pred = model(g)
                    loss = loss_fn(pred, torch.tensor([[t]]))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            best = max(y)

            candidates = []
            for _ in range(self.candidates):
                base = random.choice(population)
                c = mutate(base, self.gate_set, self.n_qubits)
                g = circuit_to_graph(c, self.gate_set, self.n_qubits)
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long)

                model.train()
                preds = [model(g).item() for _ in range(6)]
                mu, sigma = np.mean(preds), np.std(preds)
                ei = expected_improvement(mu, sigma, best)
                candidates.append((ei, c))

            candidates.sort(reverse=True)
            population.append(candidates[0][1])

        return max(population, key=lambda c: len(circuit.gates) - len(c.gates))
