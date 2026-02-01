# src/algorithm/naive_qaoa.py
# Naive QAOA Implementation for LABS (non-recursive, for benchmarking)
# Runs standard QAOA on full Hamiltonian without decimation

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
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
)


@dataclass
class NaiveQAOAResult:
    """Result container for Naive QAOA."""
    sequence: List[int]
    energy: int
    merit_factor: float
    n_function_evals: int
    optimized_params: Optional[Dict] = None
    all_samples: Optional[Dict[str, int]] = None


class NaiveQAOA:
    """
    Standard QAOA for LABS without recursive decimation.
    Used for benchmarking against RQAOA.
    """
    
    def __init__(
        self,
        depth: int = 1,
        shots: int = 1000,
        max_opt_iter: int = 50,
        n_restarts: int = 3,
    ):
        self.depth = depth
        self.shots = shots
        self.max_opt_iter = max_opt_iter
        self.n_restarts = n_restarts
        
    def solve(self, N: int, verbose: bool = True) -> NaiveQAOAResult:
        """
        Solve LABS using standard QAOA on full Hamiltonian.
        
        Args:
            N: Sequence length
            verbose: Print progress
            
        Returns:
            NaiveQAOAResult with best sequence found
        """
        if verbose:
            print(f"Naive QAOA: N={N}, depth={self.depth}, shots={self.shots}")
        
        # Build full LABS Hamiltonian
        J, K = get_labs_hamiltonian_coefficients(N)
        
        # Prepare edge lists for QAOA kernel
        self.e2_i = [int(e[0]) for e in J.keys()]
        self.e2_j = [int(e[1]) for e in J.keys()]
        self.e2_w = [float(w) for w in J.values()]
        self.n_edges_2 = len(self.e2_i)
        
        self.e4_i = [int(e[0]) for e in K.keys()]
        self.e4_j = [int(e[1]) for e in K.keys()]
        self.e4_k = [int(e[2]) for e in K.keys()]
        self.e4_l = [int(e[3]) for e in K.keys()]
        self.e4_w = [float(w) for w in K.values()]
        self.n_edges_4 = len(self.e4_i)
        
        self.N = N
        self.J = J
        self.K = K
        
        # Multi-start optimization
        best_energy = float('inf')
        best_sequence = None
        best_params = None
        best_samples = None
        total_evals = 0
        
        for restart in range(self.n_restarts):
            if verbose:
                print(f"  Restart {restart + 1}/{self.n_restarts}")
            
            beta, gamma, samples, n_evals = self._optimize_parameters(verbose)
            total_evals += n_evals
            
            # Find best sequence from samples
            for bitstring, count in samples.items():
                if len(bitstring) != N:
                    continue
                seq = [1 if b == '0' else -1 for b in bitstring]
                energy = calculate_energy(seq)
                
                if energy < best_energy:
                    best_energy = energy
                    best_sequence = seq
                    best_params = {'beta': beta.tolist(), 'gamma': gamma.tolist()}
                    best_samples = samples
                    
                    if verbose:
                        mf = (N ** 2) / (2 * energy) if energy > 0 else float('inf')
                        print(f"    New best: E={energy}, MF={mf:.2f}")
        
        if best_sequence is None:
            # Fallback to random if optimization failed
            best_sequence = np.random.choice([-1, 1], size=N).tolist()
            best_energy = calculate_energy(best_sequence)
        
        mf = (N ** 2) / (2 * best_energy) if best_energy > 0 else float('inf')
        
        return NaiveQAOAResult(
            sequence=best_sequence,
            energy=best_energy,
            merit_factor=mf,
            n_function_evals=total_evals,
            optimized_params=best_params,
            all_samples=best_samples,
        )
    
    def _optimize_parameters(
        self, verbose: bool
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], int]:
        """Optimize QAOA parameters using COBYLA."""
        from scipy.optimize import minimize
        
        # Initial parameters (random)
        x0 = np.concatenate([
            np.random.uniform(0, np.pi / 4, self.depth),      # beta
            np.random.uniform(0, np.pi / 2, self.depth),      # gamma
        ])
        
        self._best_samples = {}
        self._eval_count = 0
        
        def objective(params):
            beta = params[:self.depth].tolist()
            gamma = params[self.depth:].tolist()
            
            samples = self._run_qaoa(beta, gamma)
            
            # Compute expected LABS energy from samples
            total_energy = 0.0
            total_counts = 0
            
            for bitstring, count in samples.items():
                if len(bitstring) != self.N:
                    continue
                seq = [1 if b == '0' else -1 for b in bitstring]
                e = calculate_energy(seq)
                total_energy += e * count
                total_counts += count
            
            avg_energy = total_energy / total_counts if total_counts > 0 else float('inf')
            
            self._best_samples = samples
            self._eval_count += 1
            
            return avg_energy
        
        result = minimize(
            objective,
            x0,
            method='COBYLA',
            options={'maxiter': self.max_opt_iter, 'rhobeg': 0.5}
        )
        
        optimal_beta = result.x[:self.depth]
        optimal_gamma = result.x[self.depth:]
        
        return optimal_beta, optimal_gamma, self._best_samples, self._eval_count
    
    def _run_qaoa(self, beta: List[float], gamma: List[float]) -> Dict[str, int]:
        """Execute QAOA circuit."""
        if not CUDAQ_AVAILABLE:
            print(f"WARNING: CUDAQ not available, using classical sim for N={self.N}")
            return self._run_qaoa_classical_sim(beta, gamma)
        
        n = self.N
        p = self.depth
        n2 = self.n_edges_2
        n4 = self.n_edges_4
        
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
                # Problem Hamiltonian: 2-body ZZ
                for edge_idx in range(n_edges_2):
                    qi = edges2_i[edge_idx]
                    qj = edges2_j[edge_idx]
                    w = edges2_w[edge_idx]
                    x.ctrl(qubits[qi], qubits[qj])
                    rz(2.0 * gamma[layer] * w, qubits[qj])
                    x.ctrl(qubits[qi], qubits[qj])
                
                # Problem Hamiltonian: 4-body ZZZZ
                for edge_idx in range(n_edges_4):
                    qi = edges4_i[edge_idx]
                    qj = edges4_j[edge_idx]
                    qk = edges4_k[edge_idx]
                    ql = edges4_l[edge_idx]
                    w = edges4_w[edge_idx]
                    # Parity computation into ql
                    x.ctrl(qubits[qi], qubits[ql])
                    x.ctrl(qubits[qj], qubits[ql])
                    x.ctrl(qubits[qk], qubits[ql])
                    rz(2.0 * gamma[layer] * w, qubits[ql])
                    x.ctrl(qubits[qk], qubits[ql])
                    x.ctrl(qubits[qj], qubits[ql])
                    x.ctrl(qubits[qi], qubits[ql])
                
                # Mixer Hamiltonian
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
        """Classical statevector simulation fallback."""
        n = self.N

        dim = 2 ** n
        psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
        
        for layer in range(self.depth):
            # Problem unitary exp(-i * gamma * H_C)
            for idx in range(dim):
                bits = [(idx >> i) & 1 for i in range(n)]
                seq = [1 if b == 0 else -1 for b in bits]
                
                phase = 0.0
                for (i, j), w in self.J.items():
                    phase += w * seq[i] * seq[j]
                for (i, j, k, l), w in self.K.items():
                    phase += w * seq[i] * seq[j] * seq[k] * seq[l]
                
                psi[idx] *= np.exp(-1j * gamma[layer] * phase)
            
            # Mixer unitary: product of Rx(2*beta)
            c = np.cos(beta[layer])
            s = np.sin(beta[layer])
            for q in range(n):
                new_psi = np.zeros_like(psi)
                for idx in range(dim):
                    idx_flip = idx ^ (1 << q)
                    new_psi[idx] += c * psi[idx] - 1j * s * psi[idx_flip]
                psi = new_psi
        
        # Sample from probability distribution
        probs = np.abs(psi) ** 2
        probs /= probs.sum()
        
        indices = np.random.choice(dim, size=self.shots, p=probs)
        samples = {}
        for idx in indices:
            bits = format(idx, f'0{n}b')[::-1]
            samples[bits] = samples.get(bits, 0) + 1
        
        return samples


def solve_naive_qaoa(
    N: int,
    depth: int = 1,
    shots: int = 1000,
    max_opt_iter: int = 50,
    n_restarts: int = 3,
    verbose: bool = False,
) -> NaiveQAOAResult:
    """Wrapper function for NaiveQAOA solver."""
    solver = NaiveQAOA(depth, shots, max_opt_iter, n_restarts)
    return solver.solve(N, verbose=verbose)


def get_naive_qaoa_seeds(
    N: int,
    n_seeds: int = 10,
    depth: int = 1,
    shots: int = 2000,
    verbose: bool = False,
) -> List[List[int]]:
    """
    Generate multiple seed sequences from naive QAOA for population seeding.
    
    Args:
        N: Sequence length
        n_seeds: Number of seeds to generate
        depth: QAOA depth
        shots: Number of shots
        verbose: Print progress
        
    Returns:
        List of seed sequences (each is List[int] of +1/-1)
    """
    solver = NaiveQAOA(depth=depth, shots=shots, max_opt_iter=30, n_restarts=1)
    
    # Build Hamiltonian
    J, K = get_labs_hamiltonian_coefficients(N)
    solver.e2_i = [int(e[0]) for e in J.keys()]
    solver.e2_j = [int(e[1]) for e in J.keys()]
    solver.e2_w = [float(w) for w in J.values()]
    solver.n_edges_2 = len(solver.e2_i)
    solver.e4_i = [int(e[0]) for e in K.keys()]
    solver.e4_j = [int(e[1]) for e in K.keys()]
    solver.e4_k = [int(e[2]) for e in K.keys()]
    solver.e4_l = [int(e[3]) for e in K.keys()]
    solver.e4_w = [float(w) for w in K.values()]
    solver.n_edges_4 = len(solver.e4_i)
    solver.N = N
    solver.J = J
    solver.K = K
    
    # Quick optimization
    beta, gamma, samples, _ = solver._optimize_parameters(verbose)
    
    if verbose:
        print(f"Naive QAOA seeding: optimized beta={beta}, gamma={gamma}")
    
    # Extract top sequences by frequency
    sorted_samples = sorted(samples.items(), key=lambda x: -x[1])
    
    seeds = []
    seen = set()
    
    for bitstring, count in sorted_samples:
        if len(bitstring) != N:
            continue
        if bitstring in seen:
            continue
        
        seq = [1 if b == '0' else -1 for b in bitstring]
        seeds.append(seq)
        seen.add(bitstring)
        
        if len(seeds) >= n_seeds:
            break
    
    # Fill remaining with perturbations if needed
    while len(seeds) < n_seeds and len(seeds) > 0:
        base = seeds[np.random.randint(len(seeds))]
        new_seq = base.copy()
        n_flips = np.random.randint(1, 4)
        flip_idx = np.random.choice(N, n_flips, replace=False)
        for idx in flip_idx:
            new_seq[idx] *= -1
        seeds.append(new_seq)
    
    return seeds
