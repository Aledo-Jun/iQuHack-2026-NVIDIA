# Recursive QAOA (RQAOA) Implementation for LABS with 2/4-Body Correlations
# Based on "Obstacles to State Preparation and Variational Optimization from Symmetry Protection"

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

try:
    import cudaq
    CUDAQ_AVAILABLE = True
except ImportError:
    print('WARNING: CUDAQ not available', flush=True)
    CUDAQ_AVAILABLE = False
    cudaq = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    print('WARNING: CUPY not available', flush=True)
    cp = None
    CUPY_AVAILABLE = False
    
from utils.labs_utils import (
    calculate_energy,
    get_labs_hamiltonian_coefficients,
    brute_force_optimal,
)

# ============================================================
# Data Structures
# ============================================================
@dataclass
class ReducedHamiltonian:
    """Hamiltonian with 2-body and 4-body terms."""
    J: Dict[Tuple[int, int], float] = field(default_factory=dict)
    K: Dict[Tuple[int, int, int, int], float] = field(default_factory=dict)
    constant: float = 0.0
    n_qubits: int = 0
    
@dataclass
class RQAOAResult:
    """Result container for RQAOA."""
    sequence: List[int]
    energy: int
    merit_factor: float
    decimation_history: List[Tuple[int, int, float]]
    n_qaoa_iterations: int
    optimized_params: Optional[Dict] = None
    
# ============================================================
# QAOA Parameter Optimizer
# ============================================================
class QAOAOptimizer:
    """
    Handles QAOA parameter optimization using classical optimizer.
    """
    
    def __init__(
        self,
        ham: ReducedHamiltonian,
        depth: int,
        shots: int,
        max_opt_iter: int = 50,
    ):
        self.ham = ham
        self.depth = depth
        self.shots = shots
        self.max_opt_iter = max_opt_iter
        self.n_qubits = ham.n_qubits
        
        # Precompute edge lists (avoid recomputation)
        self._prepare_edge_lists()
        
    def _prepare_edge_lists(self):
        """Convert Hamiltonian to edge list format for kernel."""
        self.e2_i = [int(e[0]) for e in self.ham.J.keys()]
        self.e2_j = [int(e[1]) for e in self.ham.J.keys()]
        self.e2_w = [float(w) for w in self.ham.J.values()]
        self.n_edges_2 = len(self.e2_i)
        
        self.e4_i = [int(e[0]) for e in self.ham.K.keys()]
        self.e4_j = [int(e[1]) for e in self.ham.K.keys()]
        self.e4_k = [int(e[2]) for e in self.ham.K.keys()]
        self.e4_l = [int(e[3]) for e in self.ham.K.keys()]
        self.e4_w = [float(w) for w in self.ham.K.values()]
        self.n_edges_4 = len(self.e4_i)
    
    def optimize(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Optimize QAOA parameters using COBYLA.
        
        Returns:
            (optimal_beta, optimal_gamma, best_samples)
        """
        from scipy.optimize import minimize
        
        # Initial parameters
        x0 = np.concatenate([
            np.random.uniform(0, np.pi / 4, self.depth),      # beta
            np.random.uniform(0, np.pi / 2, self.depth),      # gamma
        ])
        
        self._best_energy = float('inf')
        self._best_samples = {}
        self._eval_count = 0
        
        def objective(params):
            beta = params[:self.depth].tolist()
            gamma = params[self.depth:].tolist()
            
            samples = self._run_qaoa(beta, gamma)
            
            # Compute expected energy from samples
            total_energy = 0.0
            total_counts = 0
            n = self.n_qubits
            for bitstring, count in samples.items():
                if len(bitstring) != n: # Skip malformed bitstring
                    continue
                    
                seq = [1 if b == '0' else -1 for b in bitstring]
                e = self._compute_hamiltonian_energy(seq)
                total_energy += e * count
                total_counts += count
                
            avg_energy = total_energy / total_counts if total_counts > 0 else float('inf')
            
            # Track best
            if avg_energy < self._best_energy:
                self._best_energy = avg_energy
                self._best_samples = samples.copy()
            
            self._eval_count += 1
            return avg_energy
        
        # Run optimization
        result = minimize(
            objective,
            x0,
            method='COBYLA',
            options={'maxiter': self.max_opt_iter, 'rhobeg': 0.5}
        )
        
        optimal_beta = result.x[:self.depth]
        optimal_gamma = result.x[self.depth:]
        
        return optimal_beta, optimal_gamma, self._best_samples
    
    def _compute_hamiltonian_energy(self, seq: List[int]) -> float:
        """Compute Hamiltonian energy for a sequence."""
        energy = self.ham.constant
        
        for (i, j), w in self.ham.J.items():
            energy += w * seq[i] * seq[j]
            
        for (i, j, k, l), w in self.ham.K.items():
            energy += w * seq[i] * seq[j] * seq[k] * seq[l]
            
        return energy
    
    def _run_qaoa(self, beta: List[float], gamma: List[float]) -> Dict[str, int]:
        """Execute QAOA circuit with given parameters."""
        if not CUDAQ_AVAILABLE:
            print(f"WARNING: CUDAQ not available, using classical sim for N={self.n_qubits}")
            return self._run_qaoa_classical_sim(beta, gamma)
        
        n = self.n_qubits
        p = self.depth
        n2 = self.n_edges_2
        n4 = self.n_edges_4
        
        # Pre-extract edge data to avoid list operations in kernel
        e2_i = self.e2_i
        e2_j = self.e2_j
        e2_w = self.e2_w
        e4_i = self.e4_i
        e4_j = self.e4_j
        e4_k = self.e4_k
        e4_l = self.e4_l
        e4_w = self.e4_w
        
        @cudaq.kernel
        def qaoa_kernel(
            n_qubits: int,
            depth: int,
            n_edges_2: int,
            n_edges_4: int,
            beta: List[float],
            gamma: List[float],
            edges2_i: List[int],
            edges2_j: List[int],
            edges2_w: List[float],
            edges4_i: List[int],
            edges4_j: List[int],
            edges4_k: List[int],
            edges4_l: List[int],
            edges4_w: List[float],
        ):
            qubits = cudaq.qvector(n_qubits)
            
            # Initial superposition
            for q_idx in range(n_qubits):
                h(qubits[q_idx])
            
            # QAOA layers
            for layer in range(depth):
                # 2-body ZZ interactions
                for edge_idx in range(n_edges_2):
                    qi = edges2_i[edge_idx]
                    qj = edges2_j[edge_idx]
                    w = edges2_w[edge_idx]
                    # exp(-i * gamma * w * Z_i Z_j)
                    x.ctrl(qubits[qi], qubits[qj])
                    rz(2.0 * gamma[layer] * w, qubits[qj])
                    x.ctrl(qubits[qi], qubits[qj])
                
                # 4-body ZZZZ interactions
                for edge_idx in range(n_edges_4):
                    qi = edges4_i[edge_idx]
                    qj = edges4_j[edge_idx]
                    qk = edges4_k[edge_idx]
                    ql = edges4_l[edge_idx]
                    w = edges4_w[edge_idx]
                    # Compute parity into ql
                    x.ctrl(qubits[qi], qubits[ql])
                    x.ctrl(qubits[qj], qubits[ql])
                    x.ctrl(qubits[qk], qubits[ql])
                    rz(2.0 * gamma[layer] * w, qubits[ql])
                    # Uncompute
                    x.ctrl(qubits[qk], qubits[ql])
                    x.ctrl(qubits[qj], qubits[ql])
                    x.ctrl(qubits[qi], qubits[ql])
                
                # Mixer
                for q_idx in range(n_qubits):
                    rx(2.0 * beta[layer], qubits[q_idx])
            
            mz(qubits)
        
        results = cudaq.sample(
            qaoa_kernel,
            n, p, n2, n4,
            beta, gamma,
            e2_i, e2_j, e2_w,
            e4_i, e4_j, e4_k, e4_l, e4_w,
            shots_count=self.shots
        )
        
        samples_dict = {}
        for bitstring in results:
            samples_dict[bitstring] = results.count(bitstring)
            
        return samples_dict
    
    def _run_qaoa_classical_sim(
        self, beta: List[float], gamma: List[float]
    ) -> Dict[str, int]:
        """Classical simulation fallback using statevector."""
        # Simplified classical simulation for small systems
        n = self.n_qubits
        
        # Build statevector simulation
        dim = 2 ** n
        psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
        
        for layer in range(self.depth):
            # Problem unitary
            for idx in range(dim):
                bits = [(idx >> i) & 1 for i in range(n)]
                seq = [1 if b == 0 else -1 for b in bits]
                phase = 0.0
                
                for (i, j), w in self.ham.J.items():
                    phase += w * seq[i] * seq[j]
                for (i, j, k, l), w in self.ham.K.items():
                    phase += w * seq[i] * seq[j] * seq[k] * seq[l]
                
                psi[idx] *= np.exp(-1j * gamma[layer] * phase)
            
            # Mixer unitary (product of X rotations)
            # Apply Rx(2*beta) to each qubit
            c = np.cos(beta[layer])
            s = np.sin(beta[layer])
            for q in range(n):
                new_psi = np.zeros_like(psi)
                for idx in range(dim):
                    bit_q = (idx >> q) & 1
                    idx_flip = idx ^ (1 << q)
                    if bit_q == 0:
                        new_psi[idx] += c * psi[idx] - 1j * s * psi[idx_flip]
                    else:
                        new_psi[idx] += -1j * s * psi[idx_flip] + c * psi[idx]
                psi = new_psi
        
        # Sample from distribution
        probs = np.abs(psi) ** 2
        probs /= probs.sum()
        
        indices = np.random.choice(dim, size=self.shots, p=probs)
        samples = {}
        for idx in indices:
            bits = format(idx, f'0{n}b')[::-1]  # Reverse for qubit ordering
            samples[bits] = samples.get(bits, 0) + 1
        
        return samples
        
# ============================================================
# Improved RQAOA Class
# ============================================================
class RQAOA:
    """
    Recursive QAOA for LABS with proper parameter optimization.
    """
    
    def __init__(
        self,
        depth: int = 1,
        shots: int = 1000,
        classical_threshold: int = 6,
        use_4body_decimation: bool = True,
        max_opt_iter: int = 30,
    ):
        self.depth = depth
        self.shots = shots
        self.classical_threshold = classical_threshold
        self.use_4body_decimation = use_4body_decimation
        self.max_opt_iter = max_opt_iter
        
    def solve(self, N: int, verbose: bool = True) -> RQAOAResult:
        """Solve LABS using RQAOA with parameter optimization."""
        active_vars = list(range(N))
        fixed_vars: Dict[int, int] = {}
        constraints: List[Tuple[int, int, int]] = []
        
        # Initialize Hamiltonian
        J_orig, K_orig = get_labs_hamiltonian_coefficients(N)
        current_ham = ReducedHamiltonian(
            J=dict(J_orig),
            K=dict(K_orig),
            constant=0.0,
            n_qubits=N
        )
        
        decimation_history = []
        n_qaoa_calls = 0
        all_params = []
        
        while len(active_vars) > self.classical_threshold:
            if verbose:
                print(f"RQAOA: {len(active_vars)} vars, "
                      f"{len(current_ham.J)} 2-body, {len(current_ham.K)} 4-body terms")
            
            reduced_ham = self._map_to_reduced_indices(current_ham, active_vars)
            
            # Run QAOA with parameter optimization
            correlations_2body, correlations_4body, params = self._compute_correlations_optimized(
                reduced_ham, verbose
            )
            n_qaoa_calls += 1
            all_params.append(params)
            
            # Find best decimation
            best_pair, best_corr = self._find_best_decimation(
                correlations_2body, correlations_4body
            )
            
            if best_pair is None:
                if verbose:
                    print("No strong correlations found, stopping early")
                break
            
            reduced_i, reduced_j = best_pair
            i = active_vars[reduced_i]
            j = active_vars[reduced_j]
            sign = 1 if best_corr > 0 else -1
            
            if verbose:
                print(f"  Decimating: Z_{i} = {'+' if sign > 0 else '-'}Z_{j} "
                      f"(corr={best_corr:.3f})")
            
            current_ham = self._apply_decimation(current_ham, i, j, sign)
            constraints.append((i, j, sign))
            decimation_history.append((i, j, best_corr))
            active_vars.remove(i)
        
        # Classical solve for remaining variables
        if verbose:
            print(f"Classical phase: {len(active_vars)} remaining variables")
        
        if len(active_vars) > 0:
            best_partial, _ = brute_force_optimal(len(active_vars))
            for idx, var in enumerate(active_vars):
                fixed_vars[var] = best_partial[idx]
        
        # Propagate constraints
        for (elim, ref, sign) in reversed(constraints):
            fixed_vars[elim] = sign * fixed_vars[ref]
        
        sequence = [fixed_vars[i] for i in range(N)]
        energy = calculate_energy(sequence)
        mf = (N ** 2) / (2 * energy) if energy > 0 else float('inf')
        
        return RQAOAResult(
            sequence=sequence,
            energy=energy,
            merit_factor=mf,
            decimation_history=decimation_history,
            n_qaoa_iterations=n_qaoa_calls,
            optimized_params={'all_layers': all_params}
        )
    
    def _compute_correlations_optimized(
        self,
        ham: ReducedHamiltonian,
        verbose: bool
    ) -> Tuple[Dict, Dict, Dict]:
        """Compute correlations using optimized QAOA parameters."""
        
        optimizer = QAOAOptimizer(
            ham=ham,
            depth=self.depth,
            shots=self.shots,
            max_opt_iter=self.max_opt_iter
        )
        
        optimal_beta, optimal_gamma, samples = optimizer.optimize()
        
        if verbose:
            print(f"    QAOA optimized: beta={optimal_beta}, gamma={optimal_gamma}")
        
        # Compute correlations from optimized samples
        N = ham.n_qubits
        total_shots = sum(samples.values())
        
        correlations_2body = {}
        for (i, j) in ham.J.keys():
            corr = 0.0
            for bitstring, count in samples.items():
                zi = 1 if bitstring[N - 1 - i] == '0' else -1
                zj = 1 if bitstring[N - 1 - j] == '0' else -1
                corr += count * zi * zj
            correlations_2body[(i, j)] = corr / total_shots
        
        correlations_4body = {}
        for (i, j, k, l) in ham.K.keys():
            corr = 0.0
            for bitstring, count in samples.items():
                zi = 1 if bitstring[N - 1 - i] == '0' else -1
                zj = 1 if bitstring[N - 1 - j] == '0' else -1
                zk = 1 if bitstring[N - 1 - k] == '0' else -1
                zl = 1 if bitstring[N - 1 - l] == '0' else -1
                corr += count * zi * zj * zk * zl
            correlations_4body[(i, j, k, l)] = corr / total_shots
        
        return correlations_2body, correlations_4body, {
            'beta': optimal_beta.tolist(),
            'gamma': optimal_gamma.tolist()
        }
    
    def _map_to_reduced_indices(
        self, ham: ReducedHamiltonian, active_vars: List[int]
    ) -> ReducedHamiltonian:
        """Map Hamiltonian to reduced index space."""
        var_map = {v: idx for idx, v in enumerate(active_vars)}
        
        J_reduced = {}
        for (i, j), coeff in ham.J.items():
            if i in var_map and j in var_map:
                ni, nj = var_map[i], var_map[j]
                key = (min(ni, nj), max(ni, nj))
                J_reduced[key] = J_reduced.get(key, 0) + coeff
        
        K_reduced = {}
        for (i, j, k, l), coeff in ham.K.items():
            if all(x in var_map for x in [i, j, k, l]):
                new_idx = tuple(sorted([var_map[i], var_map[j], var_map[k], var_map[l]]))
                K_reduced[new_idx] = K_reduced.get(new_idx, 0) + coeff
        
        return ReducedHamiltonian(
            J=J_reduced, K=K_reduced, constant=ham.constant, n_qubits=len(active_vars)
        )
    
    def _apply_decimation(
        self, ham: ReducedHamiltonian, elim: int, ref: int, sign: int
    ) -> ReducedHamiltonian:
        """Apply decimation Z_elim = sign * Z_ref."""
        new_J = {}
        new_K = {}
        new_const = ham.constant
        
        # Process 2-body
        for (i, j), coeff in ham.J.items():
            if elim not in (i, j):
                key = (min(i, j), max(i, j))
                new_J[key] = new_J.get(key, 0) + coeff
            else:
                other = j if i == elim else i
                if other == ref:
                    new_const += sign * coeff
                else:
                    key = (min(ref, other), max(ref, other))
                    new_J[key] = new_J.get(key, 0) + sign * coeff
        
        # Process 4-body
        for (i, j, k, l), coeff in ham.K.items():
            indices = [i, j, k, l]
            if elim not in indices:
                new_K[tuple(sorted(indices))] = new_K.get(tuple(sorted(indices)), 0) + coeff
            else:
                new_indices = [ref if x == elim else x for x in indices]
                counts = Counter(new_indices)
                
                if counts[ref] == 2:
                    # Z_ref^2 = I -> reduces to 2-body
                    remaining = [x for x in new_indices if x != ref or counts[ref] == 1]
                    remaining = [x for x, c in counts.items() if c == 1]
                    if len(remaining) == 2:
                        key = (min(remaining), max(remaining))
                        new_J[key] = new_J.get(key, 0) + sign * coeff
                    elif len(remaining) == 0:
                        new_const += sign * coeff
                else:
                    unique = sorted(set(new_indices))
                    if len(unique) == 4:
                        new_K[tuple(unique)] = new_K.get(tuple(unique), 0) + sign * coeff
                    elif len(unique) == 3:
                        singles = [x for x, c in counts.items() if c == 1]
                        if len(singles) == 2:
                            key = (min(singles), max(singles))
                            new_J[key] = new_J.get(key, 0) + sign * coeff
        
        # Filter zeros
        new_J = {k: v for k, v in new_J.items() if abs(v) > 1e-10}
        new_K = {k: v for k, v in new_K.items() if abs(v) > 1e-10}
        
        return ReducedHamiltonian(J=new_J, K=new_K, constant=new_const, n_qubits=ham.n_qubits - 1)
    
    def _find_best_decimation(
        self, corr_2: Dict, corr_4: Dict
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """Find strongest correlation for decimation."""
        best_pair = None
        best_score = 0.0
        best_corr = 0.0
        
        for (i, j), corr in corr_2.items():
            score = abs(corr)
            
            if self.use_4body_decimation:
                for (a, b, c, d), c4 in corr_4.items():
                    if i in (a, b, c, d) and j in (a, b, c, d):
                        if np.sign(corr) == np.sign(c4):
                            score += 0.2 * abs(c4)
            
            if score > best_score:
                best_score = score
                best_pair = (i, j)
                best_corr = corr
        
        return best_pair, best_corr

def solve_rqaoa(
    N: int, 
    depth: int = 1,
    shots: int = 1000,
    classical_threshold: int = 6,
    use_4body_decimation: bool = True,
    max_opt_iter: int = 30,
    verbose: bool = False,
) -> RQAOAResult:
    """Wrapper for RQAOA solver"""
    solver = RQAOA(depth, shots, classical_threshold, use_4body_decimation, max_opt_iter)
    result = solver.solve(N, verbose=verbose)
    return result