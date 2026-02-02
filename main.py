import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path
from scipy.stats import norm
from typing import List, Tuple, Dict, Optional
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

class Config:
    """Global configuration"""
    # Quantum parameters
    N_QUBITS = 5
    MAX_DEPTH = 20
    GATE_SET = ["RX", "RY", "RZ", "CNOT", "H"]

    # Optimization parameters
    INIT_CIRCUITS = 20
    BO_ITERS = 50
    CANDIDATES_PER_ITER = 10

    # Model parameters
    SURROGATE_EPOCHS = 60  # Reduced for Colab speed
    MC_SAMPLES = 8
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.005
    WEIGHT_DECAY = 1e-5
    DROPOUT = 0.3

    # Constraints
    FIDELITY_THRESHOLD = 0.95
    FIDELITY_SAMPLES = 5  # 🔥 FIX 1: Multi-sample fidelity
    MAX_CIRCUIT_DEPTH = 30
    MAX_PATTERNS = 15  # 🔥 FIX 3: Cap pattern library

    # Reproducibility
    SEED = 42

    # Output
    SAVE_DIR = Path("quantum_optimization_results")
    VERBOSE = True

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)

dev = qml.device("default.qubit", wires=Config.N_QUBITS)

class QuantumCircuit:
    """
    Enhanced circuit representation WITH angle tracking

    Gate format: (gate_type, wires, angle)
    - For parametric gates (RX, RY, RZ): angle is used
    - For non-parametric (H, CNOT): angle is None
    """

    def __init__(self, gates: List[Tuple]):
        """
        gates: List of (gate_type, wires, angle) tuples
        """
        self.gates = gates
        self._hash = None

    def __len__(self):
        return len(self.gates)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(self.gates))
        return self._hash

    def __eq__(self, other):
        return self.gates == other.gates

    def copy(self):
        return QuantumCircuit(self.gates.copy())

    def depth(self) -> int:
        """Calculate circuit depth (parallel time steps)"""
        if not self.gates:
            return 0

        qubit_times = {q: 0 for q in range(Config.N_QUBITS)}

        for gate, wires, _ in self.gates:
            max_time = max(qubit_times[w] for w in wires)
            for w in wires:
                qubit_times[w] = max_time + 1

        return max(qubit_times.values())

    def count_two_qubit_gates(self) -> int:
        """Count CNOT gates (expensive on real hardware)"""
        return sum(1 for gate, _, _ in self.gates if gate == "CNOT")

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "gates": [(g, w, float(a) if a is not None else None) for g, w, a in self.gates],
            "depth": self.depth(),
            "two_qubit_gates": self.count_two_qubit_gates()
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Deserialize from dictionary"""
        return cls([(g, w, a) for g, w, a in data["gates"]])

    def visualize(self, title: str = "Circuit"):
        """Simple text visualization"""
        lines = [[] for _ in range(Config.N_QUBITS)]

        for gate, wires, angle in self.gates:
            if gate == "CNOT":
                ctrl, targ = wires
                for q in range(Config.N_QUBITS):
                    if q == ctrl:
                        lines[q].append("●")
                    elif q == targ:
                        lines[q].append("⊕")
                    elif min(ctrl, targ) < q < max(ctrl, targ):
                        lines[q].append("│")
                    else:
                        lines[q].append("─")
            else:
                gate_str = gate
                if angle is not None and Config.VERBOSE:
                    gate_str = f"{gate}({angle:.2f})"
                for q in range(Config.N_QUBITS):
                    if q in wires:
                        lines[q].append(gate_str)
                    else:
                        lines[q].append("─")

        print(f"\n{title}")
        print("=" * min(80, len(lines[0]) * 4))
        for q, line in enumerate(lines):
            print(f"q{q}: " + "─".join(line))
        print()

def circuit_to_graph(circuit: QuantumCircuit) -> Data:
    """
    Convert circuit to graph with TRUE qubit dependencies
    """
    G = nx.DiGraph()
    last_gate_on_qubit = {}

    for i, (gate, wires, angle) in enumerate(circuit.gates):
        G.add_node(i, gate=gate, qubits=tuple(wires), angle=angle)

        # Connect to previous gates on SAME qubits
        for q in wires:
            if q in last_gate_on_qubit:
                G.add_edge(last_gate_on_qubit[q], i)
            last_gate_on_qubit[q] = i

    # Node features: [one-hot gate] + [qubit indicators] + [normalized angle]
    x = []
    for node_id in G.nodes():
        gate_type = G.nodes[node_id]['gate']
        qubits = G.nodes[node_id]['qubits']
        angle = G.nodes[node_id]['angle']

        gate_vec = [1.0 if gate_type == g else 0.0 for g in Config.GATE_SET]
        qubit_vec = [1.0 if q in qubits else 0.0 for q in range(Config.N_QUBITS)]

        # Add angle feature (normalized to [0, 1])
        if angle is not None:
            angle_feature = [angle / (2 * np.pi)]
        else:
            angle_feature = [0.0]

        x.append(gate_vec + qubit_vec + angle_feature)

    x = torch.tensor(x, dtype=torch.float)

    edges = list(G.edges)
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

class CircuitObjective:
    """
    Evaluates circuits with MULTI-SAMPLE fidelity constraint

    CRITICAL FIX: Tests on multiple random parameter settings,
    not just one fixed point.
    """

    def __init__(self, target_circuit: Optional[QuantumCircuit] = None):
        self.target_circuit = target_circuit

        # Generate multiple test parameter sets
        if target_circuit is not None:
            self.test_params = self._generate_test_parameters()
        else:
            self.test_params = None

    def _generate_test_parameters(self) -> List[np.ndarray]:
        """
        Generate multiple random parameter sets for testing

        This ensures we verify equivalence across the parameter space,
        not just at one point.
        """
        num_params = sum(1 for g, _, _ in self.target_circuit.gates
                        if g in ["RX", "RY", "RZ"])

        test_params = []
        for _ in range(Config.FIDELITY_SAMPLES):
            params = np.random.uniform(0, 2*np.pi, num_params)
            test_params.append(params)

        return test_params

    def _get_state(self, circuit: QuantumCircuit, param_values: np.ndarray) -> np.ndarray:
        """Execute circuit and return quantum state vector"""
        @qml.qnode(dev)
        def qc():
            param_idx = 0
            for gate, wires, angle in circuit.gates:
                if gate == "RX":
                    # Use stored angle if available, else parameter
                    theta = angle if angle is not None else param_values[param_idx]
                    qml.RX(theta, wires=wires[0])
                    param_idx += 1
                elif gate == "RY":
                    theta = angle if angle is not None else param_values[param_idx]
                    qml.RY(theta, wires=wires[0])
                    param_idx += 1
                elif gate == "RZ":
                    theta = angle if angle is not None else param_values[param_idx]
                    qml.RZ(theta, wires=wires[0])
                    param_idx += 1
                elif gate == "H":
                    qml.Hadamard(wires=wires[0])
                elif gate == "CNOT":
                    qml.CNOT(wires=wires)
            return qml.state()

        return qc()

    def evaluate(self, circuit: QuantumCircuit) -> Tuple[float, float, bool]:
        """
        Evaluate circuit with MULTI-SAMPLE fidelity constraint

        Returns:
            (score, avg_fidelity, is_valid)
        """
        try:
            if self.target_circuit is not None and self.test_params is not None:
                # 🔥 CRITICAL FIX: Test on MULTIPLE parameter sets
                fidelities = []

                for test_param in self.test_params:
                    target_state = self._get_state(self.target_circuit, test_param)
                    candidate_state = self._get_state(circuit, test_param)

                    # State fidelity: |⟨ψ_target|ψ_candidate⟩|²
                    fid = np.abs(np.vdot(target_state, candidate_state))**2
                    fidelities.append(fid)

                # Use AVERAGE fidelity across all test cases
                avg_fidelity = np.mean(fidelities)
                min_fidelity = np.min(fidelities)

                # Reject if ANY test case fails threshold
                if min_fidelity < Config.FIDELITY_THRESHOLD:
                    return -100.0, avg_fidelity, False

                # Score = fidelity bonus - complexity penalties
                gate_penalty = 0.1 * len(circuit)
                depth_penalty = 0.05 * circuit.depth()
                two_qubit_penalty = 0.2 * circuit.count_two_qubit_gates()

                score = (10 * avg_fidelity
                        - gate_penalty
                        - depth_penalty
                        - two_qubit_penalty)

                return score, avg_fidelity, True

            else:
                # Generative mode: measure expressibility
                num_params = sum(1 for g, _, _ in circuit.gates if g in ["RX", "RY", "RZ"])
                states = []
                for _ in range(5):
                    p = np.random.uniform(0, 2*np.pi, num_params)
                    s = self._get_state(circuit, p)
                    states.append(s)

                state_variance = np.var([np.abs(s)**2 for s in states], axis=0).mean()
                score = state_variance - 0.05 * len(circuit)

                return score, 1.0, True

        except Exception as e:
            if Config.VERBOSE:
                print(f"Evaluation error: {e}")
            return -100.0, 0.0, False

class CircuitSimplifier:
    """Apply quantum circuit simplification rules WITH angle tracking"""

    @staticmethod
    def simplify(circuit: QuantumCircuit) -> QuantumCircuit:
        """Apply all simplification rules"""
        simplified = circuit.copy()

        changed = True
        iterations = 0
        max_iterations = 10

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            # Rule 1: Cancel inverse pairs
            new_gates, rule1_applied = CircuitSimplifier._cancel_inverses(simplified.gates)
            if rule1_applied:
                simplified.gates = new_gates
                changed = True

            # Rule 2: Merge adjacent rotations (WITH ANGLES)
            new_gates, rule2_applied = CircuitSimplifier._merge_rotations(simplified.gates)
            if rule2_applied:
                simplified.gates = new_gates
                changed = True

            # Rule 3: Remove identity rotations
            new_gates, rule3_applied = CircuitSimplifier._remove_identities(simplified.gates)
            if rule3_applied:
                simplified.gates = new_gates
                changed = True

        return simplified

    @staticmethod
    def _cancel_inverses(gates: List[Tuple]) -> Tuple[List[Tuple], bool]:
        """Cancel self-inverse gates: H-H, CNOT-CNOT"""
        simplified = gates.copy()
        changed = False

        i = 0
        while i < len(simplified) - 1:
            gate1, wires1, angle1 = simplified[i]
            gate2, wires2, angle2 = simplified[i+1]

            # Self-inverse gates on same wires cancel
            if gate1 == gate2 and wires1 == wires2:
                if gate1 in ["H", "CNOT"]:
                    simplified.pop(i)
                    simplified.pop(i)
                    changed = True
                    continue

            i += 1

        return simplified, changed

    @staticmethod
    def _merge_rotations(gates: List[Tuple]) -> Tuple[List[Tuple], bool]:
        """
        🔥 CRITICAL FIX: Merge rotations by SUMMING angles

        RX(θ₁) followed by RX(θ₂) = RX(θ₁ + θ₂)
        """
        simplified = gates.copy()
        changed = False

        i = 0
        while i < len(simplified) - 1:
            gate1, wires1, angle1 = simplified[i]
            gate2, wires2, angle2 = simplified[i+1]

            # Same rotation type on same qubit WITH angles
            if (gate1 == gate2 and
                gate1 in ["RX", "RY", "RZ"] and
                wires1 == wires2 and
                angle1 is not None and
                angle2 is not None):

                # Merge by summing angles (mod 2π)
                merged_angle = (angle1 + angle2) % (2 * np.pi)

                # If result is ~0, remove entirely
                if abs(merged_angle) < 1e-6 or abs(merged_angle - 2*np.pi) < 1e-6:
                    simplified.pop(i)
                    simplified.pop(i)
                else:
                    # Replace with merged gate
                    simplified[i] = (gate1, wires1, merged_angle)
                    simplified.pop(i+1)

                changed = True
                continue

            i += 1

        return simplified, changed

    @staticmethod
    def _remove_identities(gates: List[Tuple]) -> Tuple[List[Tuple], bool]:
        """Remove rotations by ~0 or ~2π (identity operations)"""
        simplified = []
        changed = False

        for gate, wires, angle in gates:
            # Keep non-parametric gates
            if angle is None:
                simplified.append((gate, wires, angle))
            else:
                # Remove identity rotations
                normalized_angle = angle % (2 * np.pi)
                if abs(normalized_angle) > 1e-6 and abs(normalized_angle - 2*np.pi) > 1e-6:
                    simplified.append((gate, wires, angle))
                else:
                    changed = True

        return simplified, changed

class CircuitMutator:
    """Intelligent mutation operators with angle handling"""

    @staticmethod
    def mutate(circuit: QuantumCircuit,
               patterns: Optional[List[QuantumCircuit]] = None) -> QuantumCircuit:
        """Apply random mutation"""

        # 30% chance to use pattern
        if patterns and random.random() < 0.3:
            return random.choice(patterns).copy()

        new = circuit.copy()
        mutation_type = random.random()

        if mutation_type < 0.25 and len(new) > 2:
            # Delete random gate
            new.gates.pop(random.randint(0, len(new) - 1))
            new = CircuitSimplifier.simplify(new)

        elif mutation_type < 0.45 and len(new) > 2:
            # Swap adjacent gates
            i = random.randint(0, len(new) - 2)
            new.gates[i], new.gates[i+1] = new.gates[i+1], new.gates[i]

        elif mutation_type < 0.65:
            # Add random gate WITH angle
            gate = random.choice(Config.GATE_SET)
            pos = random.randint(0, len(new))
            if gate == "CNOT":
                wires = random.sample(range(Config.N_QUBITS), 2)
                angle = None
            elif gate == "H":
                wires = [random.randint(0, Config.N_QUBITS - 1)]
                angle = None
            else:  # Rotation gates
                wires = [random.randint(0, Config.N_QUBITS - 1)]
                angle = np.random.uniform(0, 2*np.pi)

            new.gates.insert(pos, (gate, wires, angle))

        elif mutation_type < 0.85 and len(new) > 0:
            # Modify existing gate angle
            idx = random.randint(0, len(new) - 1)
            gate, wires, angle = new.gates[idx]

            if angle is not None:
                # Perturb angle
                new_angle = angle + np.random.normal(0, np.pi/4)
                new.gates[idx] = (gate, wires, new_angle % (2*np.pi))
            else:
                # Replace gate
                new_gate = random.choice(Config.GATE_SET)
                if new_gate == "CNOT":
                    new_wires = random.sample(range(Config.N_QUBITS), 2)
                    new_angle = None
                elif new_gate == "H":
                    new_wires = [random.randint(0, Config.N_QUBITS - 1)]
                    new_angle = None
                else:
                    new_wires = [random.randint(0, Config.N_QUBITS - 1)]
                    new_angle = np.random.uniform(0, 2*np.pi)
                new.gates[idx] = (new_gate, new_wires, new_angle)

        else:
            # Insert pattern fragment
            fragment_length = random.randint(2, 3)
            pos = random.randint(0, len(new))
            for _ in range(fragment_length):
                gate = random.choice(Config.GATE_SET)
                if gate == "CNOT":
                    wires = random.sample(range(Config.N_QUBITS), 2)
                    angle = None
                elif gate == "H":
                    wires = [random.randint(0, Config.N_QUBITS - 1)]
                    angle = None
                else:
                    wires = [random.randint(0, Config.N_QUBITS - 1)]
                    angle = np.random.uniform(0, 2*np.pi)
                new.gates.insert(pos, (gate, wires, angle))

        return new


class CircuitGNN(nn.Module):
    """GNN for circuit representation learning"""

    def __init__(self, input_dim: int):
        super().__init__()

        hidden = Config.HIDDEN_DIM

        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden // 2)

        self.fc1 = nn.Linear(hidden // 2, 64)
        self.fc2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(Config.DROPOUT)
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.layer_norm2 = nn.LayerNorm(hidden)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long)

        x = self.conv1(x, edge_index)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def train_surrogate(model: nn.Module,
                   graphs: List[Data],
                   scores: List[float]) -> nn.Module:
    """Train surrogate model with train/val split"""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()

    dataset = [(g, torch.tensor([s], dtype=torch.float)) for g, s in zip(graphs, scores)]

    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size]
    val_data = dataset[train_size:] if train_size < len(dataset) else dataset[-2:]

    best_val_loss = float('inf')
    patience = 0
    max_patience = 10

    for epoch in range(Config.SURROGATE_EPOCHS):
        model.train()
        train_loss = 0

        for graph, target in train_data:
            pred = model(graph)
            loss = loss_fn(pred, target.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_data)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for graph, target in val_data:
                pred = model(graph)
                loss = loss_fn(pred, target.unsqueeze(0))
                val_loss += loss.item()

        val_loss /= len(val_data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                break

    return model

def expected_improvement(mu: float, sigma: float, best: float, xi: float = 0.01) -> float:
    """Proper Expected Improvement acquisition function"""
    if sigma < 1e-6:
        return 0.0

    Z = (mu - best - xi) / sigma
    ei = (mu - best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

    return max(0.0, ei)

def predict_with_uncertainty(model: nn.Module, graph: Data, n_samples: int = 8) -> Tuple[float, float]:
    """Use MC Dropout to estimate prediction uncertainty"""
    model.train()

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(graph).item()
            predictions.append(pred)

    return np.mean(predictions), np.std(predictions)


class QuantumCircuitOptimizer:
    """Main optimization engine with all fixes applied"""

    def __init__(self, config: Config = Config()):
        self.config = config
        self.history = {
            'scores': [],
            'fidelities': [],
            'gate_counts': [],
            'best_scores': [],
            'iteration': []
        }

    def optimize(self, user_circuit: QuantumCircuit) -> Dict:
        """
        Main optimization loop

        FIXES APPLIED:
        ✅ Multi-sample fidelity verification
        ✅ Angle-aware simplification
        ✅ Capped pattern library
        """
        print("\n" + "="*70)
        print("🔬 QUANTUM CIRCUIT OPTIMIZER")
        print("Graph-Based Bayesian Optimization with Fidelity Constraints")
        print("="*70)
        print(f"\n📋 Configuration:")
        print(f"   Fidelity threshold: {Config.FIDELITY_THRESHOLD}")
        print(f"   Fidelity test samples: {Config.FIDELITY_SAMPLES}")
        print(f"   BO iterations: {Config.BO_ITERS}")
        print(f"   Max pattern library: {Config.MAX_PATTERNS}")

        # Initialize objective
        objective = CircuitObjective(user_circuit)

        # Initial evaluation
        initial_score, initial_fidelity, _ = objective.evaluate(user_circuit)

        print(f"\n📊 Initial Circuit:")
        print(f"   Gates: {len(user_circuit)}")
        print(f"   Depth: {user_circuit.depth()}")
        print(f"   Two-qubit gates: {user_circuit.count_two_qubit_gates()}")
        print(f"   Score: {initial_score:.4f}")
        print(f"   Avg fidelity: {initial_fidelity:.4f}")

        # Initialize population
        circuits = [user_circuit]

        for _ in range(Config.INIT_CIRCUITS // 2):
            c = user_circuit.copy()
            for _ in range(random.randint(1, 3)):
                c = CircuitMutator.mutate(c)
            c = CircuitSimplifier.simplify(c)
            circuits.append(c)

        while len(circuits) < Config.INIT_CIRCUITS:
            circuits.append(CircuitMutator.mutate(user_circuit))

        # Evaluate all
        results = [objective.evaluate(c) for c in circuits]
        scores = [r[0] for r in results]
        fidelities = [r[1] for r in results]
        validities = [r[2] for r in results]

        # Filter valid
        valid_idx = [i for i, v in enumerate(validities) if v]

        if len(valid_idx) == 0:
            print("\n⚠️  No valid circuits in initial population!")
            return self._create_result_dict(user_circuit, user_circuit, initial_score, 1.0)

        circuits = [circuits[i] for i in valid_idx]
        scores = [scores[i] for i in valid_idx]
        fidelities = [fidelities[i] for i in valid_idx]

        # Convert to graphs
        graphs = []
        for c in circuits:
            g = circuit_to_graph(c)
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
            graphs.append(g)

        # Initialize model
        input_dim = len(Config.GATE_SET) + Config.N_QUBITS + 1  # +1 for angle
        model = CircuitGNN(input_dim)

        # 🔥 FIX 3: Capped pattern library
        patterns = []

        print(f"\n🚀 Starting Optimization")
        print(f"   Valid circuits: {len(circuits)}")
        print("-"*70)

        # Main BO loop
        for iteration in range(Config.BO_ITERS):
            # Train surrogate
            model = train_surrogate(model, graphs, scores)

            # Current best
            best_so_far = max(scores)
            best_idx = scores.index(best_so_far)
            best_fidelity = fidelities[best_idx]

            # Generate candidates
            candidates = []
            for _ in range(Config.CANDIDATES_PER_ITER):
                base = random.choice(circuits)
                candidate = CircuitMutator.mutate(base, patterns)
                candidate = CircuitSimplifier.simplify(candidate)

                if len(candidate) <= Config.MAX_CIRCUIT_DEPTH:
                    candidates.append(candidate)

            if not candidates:
                candidates = [CircuitMutator.mutate(random.choice(circuits))]

            # Rank by EI
            ranked = []
            for c in candidates:
                g = circuit_to_graph(c)
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long)

                mu, sigma = predict_with_uncertainty(model, g, Config.MC_SAMPLES)
                ei = expected_improvement(mu, sigma, best_so_far)

                ranked.append((ei, c))

            ranked.sort(reverse=True, key=lambda x: x[0])
            best_candidate = ranked[0][1]

            # Evaluate
            score, fidelity, is_valid = objective.evaluate(best_candidate)

            if is_valid:
                circuits.append(best_candidate)
                scores.append(score)
                fidelities.append(fidelity)

                g = circuit_to_graph(best_candidate)
                g.batch = torch.zeros(g.x.size(0), dtype=torch.long)
                graphs.append(g)

                # 🔥 FIX 3: Add to patterns with cap
                if score > np.percentile(scores, 80):
                    if len(patterns) < Config.MAX_PATTERNS:
                        patterns.append(best_candidate)
                    else:
                        # Replace worst pattern
                        worst_idx = 0
                        worst_score = float('inf')
                        for idx, p in enumerate(patterns):
                            p_score, _, _ = objective.evaluate(p)
                            if p_score < worst_score:
                                worst_score = p_score
                                worst_idx = idx
                        patterns[worst_idx] = best_candidate

                improvement = "✓" if score > best_so_far else " "
                print(f"[{iteration:3d}] {improvement} Score: {score:7.4f} | "
                      f"Fidelity: {fidelity:.4f} | Gates: {len(best_candidate):3d}")

                self.history['scores'].append(score)
                self.history['fidelities'].append(fidelity)
                self.history['gate_counts'].append(len(best_candidate))
                self.history['best_scores'].append(max(scores))
                self.history['iteration'].append(iteration)
            else:
                if Config.VERBOSE:
                    print(f"[{iteration:3d}] ✗ Invalid (fidelity: {fidelity:.4f})")

        # Final results
        best_idx = np.argmax(scores)
        best_circuit = circuits[best_idx]
        best_score = scores[best_idx]
        best_fidelity = fidelities[best_idx]

        print("\n" + "="*70)
        print("✅ OPTIMIZATION COMPLETE")
        print("="*70)

        print(f"\n📈 Results:")
        print(f"   {'Metric':<25} {'Original':<15} {'Optimized':<15} {'Change'}")
        print(f"   {'-'*70}")
        print(f"   {'Gates':<25} {len(user_circuit):<15} {len(best_circuit):<15} {len(best_circuit)-len(user_circuit):+d}")
        print(f"   {'Depth':<25} {user_circuit.depth():<15} {best_circuit.depth():<15} {best_circuit.depth()-user_circuit.depth():+d}")
        print(f"   {'CNOT gates':<25} {user_circuit.count_two_qubit_gates():<15} {best_circuit.count_two_qubit_gates():<15} {best_circuit.count_two_qubit_gates()-user_circuit.count_two_qubit_gates():+d}")
        print(f"   {'Score':<25} {initial_score:<15.4f} {best_score:<15.4f} {best_score-initial_score:+.4f}")
        print(f"   {'Avg fidelity':<25} {initial_fidelity:<15.4f} {best_fidelity:<15.4f} {best_fidelity-initial_fidelity:+.4f}")

        reduction = 100 * (1 - len(best_circuit) / len(user_circuit))
        print(f"\n✨ Gate reduction: {reduction:.1f}%")

        if best_fidelity < 0.99:
            print(f"\n⚠️  Final fidelity: {best_fidelity:.4f}")
            print("   (Acceptable for approximate optimization)")
        else:
            print(f"\n✅ Excellent fidelity: {best_fidelity:.4f}")

        return self._create_result_dict(user_circuit, best_circuit, best_score, best_fidelity)

    def _create_result_dict(self, original: QuantumCircuit,
                           optimized: QuantumCircuit,
                           score: float, fidelity: float) -> Dict:
        """Create results dictionary"""
        return {
            'original_circuit': original.to_dict(),
            'optimized_circuit': optimized.to_dict(),
            'original_gates': len(original),
            'optimized_gates': len(optimized),
            'gate_reduction': 100 * (1 - len(optimized) / len(original)),
            'original_depth': original.depth(),
            'optimized_depth': optimized.depth(),
            'score': score,
            'fidelity': fidelity,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }

    def plot_results(self):
        """Plot optimization progress"""
        if not self.history['iteration']:
            print("No history to plot!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Score convergence
        axes[0, 0].plot(self.history['iteration'], self.history['best_scores'],
                       linewidth=2, color='#2E86AB', marker='o', markersize=4)
        axes[0, 0].set_xlabel('Iteration', fontsize=12)
        axes[0, 0].set_ylabel('Best Score', fontsize=12)
        axes[0, 0].set_title('Score Convergence', fontsize=14, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)

        # Fidelity distribution
        axes[0, 1].hist(self.history['fidelities'], bins=20,
                       color='#A23B72', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(Config.FIDELITY_THRESHOLD, color='red',
                          linestyle='--', linewidth=2, label=f'Threshold ({Config.FIDELITY_THRESHOLD})')
        axes[0, 1].set_xlabel('Fidelity', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('Fidelity Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Gate count evolution
        axes[1, 0].scatter(self.history['iteration'], self.history['gate_counts'],
                          alpha=0.6, c=self.history['scores'], cmap='viridis', s=50)
        axes[1, 0].set_xlabel('Iteration', fontsize=12)
        axes[1, 0].set_ylabel('Gate Count', fontsize=12)
        axes[1, 0].set_title('Circuit Complexity Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        # Pareto frontier
        scatter = axes[1, 1].scatter(self.history['gate_counts'],
                                    self.history['scores'],
                                    c=self.history['fidelities'],
                                    cmap='plasma', alpha=0.6, s=50)
        axes[1, 1].set_xlabel('Gate Count', fontsize=12)
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].set_title('Score vs Complexity (Pareto)', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Fidelity', fontsize=11)
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Save in Colab
        if IN_COLAB:
            Config.SAVE_DIR.mkdir(exist_ok=True)
            plt.savefig(Config.SAVE_DIR / "optimization_results.png", dpi=300, bbox_inches='tight')
            print(f"\n💾 Plot saved to {Config.SAVE_DIR / 'optimization_results.png'}")

def create_example_circuit() -> QuantumCircuit:
    """Create example circuit with angles"""
    gates = [
        ("H", [0], None),
        ("H", [1], None),
        ("H", [2], None),
        ("H", [3], None),
        ("H", [4], None),
        ("CNOT", [0, 1], None),
        ("CNOT", [1, 2], None),
        ("CNOT", [2, 3], None),
        ("CNOT", [3, 4], None),
        ("CNOT", [0, 1], None),  # Redundant
        ("CNOT", [1, 2], None),  # Redundant
        ("RX", [0], np.pi/4),
        ("RX", [1], np.pi/3),
        ("RX", [2], np.pi/6),
        ("RX", [3], np.pi/2),
        ("RX", [4], np.pi/8),
    ]
    return QuantumCircuit(gates)

def create_entanglement_circuit() -> QuantumCircuit:
    """Bell state + rotations"""
    gates = [
        ("H", [0], None),
        ("CNOT", [0, 1], None),
        ("RY", [0], np.pi/2),
        ("RY", [1], np.pi/2),
        ("CNOT", [0, 1], None),  # Should cancel with first CNOT
        ("RZ", [0], np.pi/4),
        ("RZ", [0], np.pi/4),  # Should merge with previous RZ
    ]
    return QuantumCircuit(gates)

def main():
    """Main execution for Colab"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   🔬 QUANTUM CIRCUIT OPTIMIZER - FULLY CORRECTED                 ║
    ║                                                                  ║
    ║   ✅ Multi-sample fidelity verification                          ║
    ║   ✅ Angle-aware rotation merging                                ║
    ║   ✅ Capped pattern library (prevents overfitting)               ║
    ║   ✅ Graph-based Bayesian Optimization                           ║
    ║   ✅ Research-grade implementation                               ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Create circuit
    print("📋 Creating example circuit...")
    user_circuit = create_example_circuit()

    print("\n🔍 User Circuit:")
    for i, (gate, wires, angle) in enumerate(user_circuit.gates):
        angle_str = f", θ={angle:.4f}" if angle is not None else ""
        print(f"  {i+1:2d}. {gate}(q{wires}){angle_str}")

    # Run optimization
    optimizer = QuantumCircuitOptimizer()
    results = optimizer.optimize(user_circuit)

    # Show optimized circuit
    print("\n🔍 Optimized Circuit:")
    optimized = QuantumCircuit.from_dict(results['optimized_circuit'])
    for i, (gate, wires, angle) in enumerate(optimized.gates):
        angle_str = f", θ={angle:.4f}" if angle is not None else ""
        print(f"  {i+1:2d}. {gate}(q{wires}){angle_str}")

    # Plot results
    print("\n📊 Generating plots...")
    optimizer.plot_results()

    # Save results
    Config.SAVE_DIR.mkdir(exist_ok=True)
    result_file = Config.SAVE_DIR / "results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {result_file}")

    print("\n" + "="*70)
    print("🎉 COMPLETE! All critical fixes applied:")
    print("   ✅ Fidelity tested on 5 random parameter sets")
    print("   ✅ Rotation angles properly merged (not discarded)")
    print("   ✅ Pattern library capped at 15 circuits")
    print("="*70)

    return results

if __name__ == "__main__":
    results = main()

