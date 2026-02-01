# src/benchmarks/benchmark_suite.py
"""
Comprehensive Benchmarking Suite for LABS Solvers
Compares: Classical MTS (CPU/GPU), Naive QAOA+MTS, Hybrid RQAOA+MTS
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from time import time
import json
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from utils.labs_utils import calculate_energy, calculate_merit_factor
from solvers.mts import run_mts, GPUAcceleratedMTS, tabu_search_optimized
from algorithms.rqaoa import solve_rqaoa, RQAOA
from algorithms.naive_qaoa import solve_naive_qaoa, NaiveQAOA


# ============================================================
# Best Known Solutions Reference
# ============================================================
# From literature: Packebusch & Mertens (2016)
BEST_KNOWN_ENERGIES = {
    3: 1, 4: 2, 5: 2, 6: 7, 7: 3, 8: 8, 9: 12, 10: 13,
    11: 5, 12: 10, 13: 6, 14: 19, 15: 15, 16: 24, 17: 32, 18: 25,
    19: 29, 20: 26, 21: 26, 22: 39, 23: 47, 24: 36, 25: 36, 26: 45,
    27: 37, 28: 50, 29: 62, 30: 59, 31: 67, 32: 64, 33: 64, 34: 65,
    35: 73, 36: 82, 37: 86, 38: 87, 39: 99, 40: 108, 41: 108,
    42: 101, 43: 109, 44: 122, 45: 118, 46: 131, 47: 135, 
}

def get_best_known_mf(N: int) -> float:
    """Get best known merit factor for given N."""
    if N in BEST_KNOWN_ENERGIES:
        E = BEST_KNOWN_ENERGIES[N]
        return (N ** 2) / (2 * E)
    return None


# ============================================================
# Result Data Structures
# ============================================================
@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    method: str
    N: int
    energy: int
    merit_factor: float
    time_seconds: float
    sequence: List[int]
    best_known_mf: Optional[float] = None
    mf_ratio: Optional[float] = None  # achieved_mf / best_known_mf
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.best_known_mf is None:
            self.best_known_mf = get_best_known_mf(self.N)
        if self.best_known_mf and self.mf_ratio is None:
            self.mf_ratio = self.merit_factor / self.best_known_mf


@dataclass
class SpeedupResult:
    """GPU vs CPU speedup measurement."""
    N: int
    cpu_time_ms: float
    gpu_time_ms: float
    speedup: float
    n_iterations: int


@dataclass 
class TrajectoryPoint:
    """Single point in optimization trajectory."""
    step: int
    n_active_vars: int
    energy: float
    merit_factor: float
    decimated_pair: Optional[Tuple[int, int]] = None
    correlation: Optional[float] = None


@dataclass
class CorrelationSnapshot:
    """Correlation matrix snapshot."""
    step: int
    n_qubits: int
    correlations_2body: Dict[Tuple[int, int], float]
    correlations_4body: Dict[Tuple[int, int, int, int], float]
    
    def to_matrix(self) -> np.ndarray:
        """Convert to dense correlation matrix."""
        matrix = np.zeros((self.n_qubits, self.n_qubits))
        for (i, j), corr in self.correlations_2body.items():
            matrix[i, j] = corr
            matrix[j, i] = corr
        np.fill_diagonal(matrix, 1.0)
        return matrix


# ============================================================
# Benchmark Runners
# ============================================================
class LABSBenchmarkSuite:
    """
    Comprehensive benchmark suite for LABS solvers.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
        self.speedup_results: List[SpeedupResult] = []
        self.trajectories: Dict[str, List[TrajectoryPoint]] = {}
        self.correlation_snapshots: List[CorrelationSnapshot] = []
        
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    # --------------------------------------------------------
    # Individual Solver Benchmarks
    # --------------------------------------------------------
    def run_classical_mts_cpu(
        self,
        N: int,
        population_size: int = 100,
        generations: int = 100,
    ) -> BenchmarkResult:
        """Run classical MTS on CPU (NumPy only)."""
        self.log(f"[CPU MTS] N={N}, pop={population_size}, gen={generations}")
        
        start = time()
        best_seq, best_energy, history = run_mts(
            N,
            population_size=population_size,
            generations=generations,
            verbose=False
        )
        elapsed = time() - start
        
        mf = calculate_merit_factor(best_seq)
        result = BenchmarkResult(
            method="MTS_CPU",
            N=N,
            energy=best_energy,
            merit_factor=mf,
            time_seconds=elapsed,
            sequence=best_seq,
            extra_data={"history": history, "generations": generations}
        )
        self.results.append(result)
        self.log(f"  -> E={best_energy}, MF={mf:.3f}, t={elapsed:.2f}s")
        return result
    
    def run_classical_mts_gpu(
        self,
        N: int,
        population_size: int = 100,
        generations: int = 100,
    ) -> BenchmarkResult:
        """Run classical MTS with GPU acceleration."""
        self.log(f"[GPU MTS] N={N}, pop={population_size}, gen={generations}")

        start = time()
        best_seq, best_energy, history = run_mts(
            N,
            population_size=population_size,
            generations=generations,
            use_gpu=True,
            batch_size=64,
            verbose=False
        )
        elapsed = time() - start
        
        mf = calculate_merit_factor(best_seq)
        result = BenchmarkResult(
            method="MTS_GPU",
            N=N,
            energy=best_energy,
            merit_factor=mf,
            time_seconds=elapsed,
            sequence=best_seq,
            extra_data={"history": history, "generations": generations}
        )
        self.results.append(result)
        self.log(f"  -> E={best_energy}, MF={mf:.3f}, t={elapsed:.2f}s")
        return result
    
    def run_naive_qaoa_mts(
        self,
        N: int,
        qaoa_depth: int = 1,
        qaoa_shots: int = 1000,
        mts_generations: int = 100,
        mts_population_size: int = 100,
    ) -> BenchmarkResult:
        """Run Naive QAOA + MTS hybrid."""
        self.log(f"[Naive QAOA+MTS] N={N}, depth={qaoa_depth}")
        
        start = time()
        
        # Quantum seeding phase
        qaoa_result = solve_naive_qaoa(
            N, depth=qaoa_depth, shots=qaoa_shots,
            max_opt_iter=30, n_restarts=2, verbose=False
        )
        quantum_time = time() - start
        seed = qaoa_result.sequence
        quantum_energy = qaoa_result.energy
        
        # Build population from seed
        population = [seed.copy()]
        n_perturbed = mts_population_size // 2
        for _ in range(n_perturbed):
            new_seq = seed.copy()
            n_flips = np.random.randint(1, 4)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            for idx in flip_idx:
                new_seq[idx] *= -1
            population.append(new_seq)
        
        while len(population) < mts_population_size:
            population.append(np.random.choice([-1, 1], size=N).tolist())
        
        # MTS phase
        start = time()
        best_seq, best_energy, history = run_mts(
            N,
            population_size=mts_population_size,
            generations=mts_generations,
            use_gpu=True,
            batch_size=64,
            verbose=False
        )
        elapsed = time() - start
        
        mf = calculate_merit_factor(best_seq)
        result = BenchmarkResult(
            method="NaiveQAOA_MTS",
            N=N,
            energy=best_energy,
            merit_factor=mf,
            time_seconds=elapsed,
            sequence=best_seq,
            extra_data={
                "quantum_energy": quantum_energy,
                "quantum_time": quantum_time,
                "qaoa_depth": qaoa_depth,
            }
        )
        self.results.append(result)
        self.log(f"  -> E={best_energy}, MF={mf:.3f}, t={elapsed:.2f}s (quantum: {quantum_time:.2f}s)")
        return result
    
    def run_rqaoa_mts(
        self,
        N: int,
        rqaoa_depth: int = 1,
        rqaoa_shots: int = 1000,
        mts_generations: int = 100,
        mts_population_size: int = 100,
        record_trajectory: bool = False,
        record_correlations: bool = False,
    ) -> BenchmarkResult:
        """Run RQAOA + MTS hybrid with optional trajectory recording."""
        self.log(f"[RQAOA+MTS] N={N}, depth={rqaoa_depth}")
        
        start = time()
        
        # Run RQAOA with trajectory capture if requested
        if record_trajectory or record_correlations:
            rqaoa_result, trajectory, correlations = self._run_rqaoa_with_tracking(
                N, rqaoa_depth, rqaoa_shots, record_correlations
            )
            if record_trajectory:
                self.trajectories[f"RQAOA_N{N}"] = trajectory
            if record_correlations:
                self.correlation_snapshots.extend(correlations)
        else:
            rqaoa_result = solve_rqaoa(
                N, depth=rqaoa_depth, shots=rqaoa_shots,
                classical_threshold=min(6, N - 2),
                verbose=False
            )
        
        quantum_time = time() - start
        seed = rqaoa_result.sequence
        quantum_energy = rqaoa_result.energy
        
        # Build population from seed
        population = [seed.copy()]
        n_perturbed = mts_population_size // 2
        for _ in range(n_perturbed):
            new_seq = seed.copy()
            n_flips = np.random.randint(1, 4)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            for idx in flip_idx:
                new_seq[idx] *= -1
            population.append(new_seq)
        
        while len(population) < mts_population_size:
            population.append(np.random.choice([-1, 1], size=N).tolist())
        
        # MTS phase
        start = time()
        best_seq, best_energy, history = run_mts(
            N,
            population_size=mts_population_size,
            generations=mts_generations,
            use_gpu=True,
            batch_size=64,
            verbose=False
        )
        elapsed = time() - start
        
        mf = calculate_merit_factor(best_seq)
        result = BenchmarkResult(
            method="RQAOA_MTS",
            N=N,
            energy=best_energy,
            merit_factor=mf,
            time_seconds=elapsed,
            sequence=best_seq,
            extra_data={
                "quantum_energy": quantum_energy,
                "quantum_time": quantum_time,
                "rqaoa_depth": rqaoa_depth,
                "n_decimations": len(rqaoa_result.decimation_history),
                "decimation_history": rqaoa_result.decimation_history,
            }
        )
        self.results.append(result)
        self.log(f"  -> E={best_energy}, MF={mf:.3f}, t={elapsed:.2f}s (quantum: {quantum_time:.2f}s)")
        return result
    
    def _run_rqaoa_with_tracking(
        self,
        N: int,
        depth: int,
        shots: int,
        record_correlations: bool,
    ) -> Tuple[Any, List[TrajectoryPoint], List[CorrelationSnapshot]]:
        """Run RQAOA with trajectory and correlation tracking."""
        from algorithms.rqaoa import RQAOA, ReducedHamiltonian, QAOAOptimizer
        from utils.labs_utils import get_labs_hamiltonian_coefficients, brute_force_optimal
        
        solver = RQAOA(
            depth=depth,
            shots=shots,
            classical_threshold=min(6, N - 2),
            use_4body_decimation=True,
            max_opt_iter=30,
        )
        
        active_vars = list(range(N))
        fixed_vars = {}
        constraints = []
        
        J_orig, K_orig = get_labs_hamiltonian_coefficients(N)
        current_ham = ReducedHamiltonian(
            J=dict(J_orig),
            K=dict(K_orig),
            constant=0.0,
            n_qubits=N
        )
        
        trajectory = []
        correlations = []
        step = 0
        
        # Initial trajectory point
        trajectory.append(TrajectoryPoint(
            step=0,
            n_active_vars=N,
            energy=float('inf'),
            merit_factor=0.0,
        ))
        
        while len(active_vars) > solver.classical_threshold:
            reduced_ham = solver._map_to_reduced_indices(current_ham, active_vars)
            
            # Compute correlations
            optimizer = QAOAOptimizer(
                ham=reduced_ham,
                depth=solver.depth,
                shots=solver.shots,
                max_opt_iter=solver.max_opt_iter
            )
            optimal_beta, optimal_gamma, samples = optimizer.optimize()
            
            # Compute correlations from samples
            n_reduced = reduced_ham.n_qubits
            total_shots = sum(samples.values())
            
            correlations_2body = {}
            for (i, j) in reduced_ham.J.keys():
                corr = 0.0
                for bitstring, count in samples.items():
                    if len(bitstring) != n_reduced:
                        continue
                    zi = 1 if bitstring[n_reduced - 1 - i] == '0' else -1
                    zj = 1 if bitstring[n_reduced - 1 - j] == '0' else -1
                    corr += count * zi * zj
                correlations_2body[(i, j)] = corr / total_shots

            correlations_4body = {}
            for (i, j, k, l) in reduced_ham.K.keys():
                corr = 0.0
                for bitstring, count in samples.items():
                    if len(bitstring) != n_reduced:
                        continue
                    zi = 1 if bitstring[n_reduced - 1 - i] == '0' else -1
                    zj = 1 if bitstring[n_reduced - 1 - j] == '0' else -1
                    zk = 1 if bitstring[n_reduced - 1 - k] == '0' else -1
                    zl = 1 if bitstring[n_reduced - 1 - l] == '0' else -1
                    corr += count * zi * zj * zk * zl
                correlations_4body[(i, j, k, l)] = corr / total_shots

            if record_correlations:
                correlations.append(CorrelationSnapshot(
                    step=step,
                    n_qubits=N,
                    correlations_2body=correlations_2body,
                    correlations_4body=correlations_4body
                ))
            
            # Find best decimation
            best_pair, best_corr = solver._find_best_decimation(
                correlations_2body, correlations_4body
            )
            
            if best_pair is None:
                break
                
            reduced_i, reduced_j = best_pair
            i = active_vars[reduced_i]
            j = active_vars[reduced_j]
            sign = 1 if best_corr > 0 else -1
            
            if len(trajectory) > 0:
                prev_point = trajectory[-1]
                trajectory.append(TrajectoryPoint(
                    step=step + 1,
                    n_active_vars=len(active_vars) - 1,
                    energy=prev_point.energy, # Carry forward or 0.0
                    merit_factor=prev_point.merit_factor,
                    decimated_pair=(i, j),
                    correlation=best_corr
                ))

            current_ham = solver._apply_decimation(current_ham, i, j, sign)
            constraints.append((i, j, sign))
            active_vars.remove(i)
            step += 1
            
            
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
        
        # Update final trajectory point
        trajectory.append(TrajectoryPoint(
            step=step + 1,
            n_active_vars=0,
            energy=energy,
            merit_factor=mf,
        ))
        
        # Create result object
        from algorithms.rqaoa import RQAOAResult
        result = RQAOAResult(
            sequence=sequence,
            energy=energy,
            merit_factor=mf,
            decimation_history=[(c[0], c[1], 0.0) for c in constraints],
            n_qaoa_iterations=step,
        )
        
        return result, trajectory, correlations
    
    # --------------------------------------------------------
    # Speedup Measurement
    # --------------------------------------------------------
    def measure_gpu_speedup(
        self,
        N_values: List[int] = [10, 20, 30, 40, 50],
        n_iterations: int = 100,
        population_size: int = 1000,
        generations: int = 500,
    ) -> List[SpeedupResult]:
        """Measure GPU vs CPU speedup for neighborhood evaluation."""
        self.log("\n=== GPU Speedup Measurement ===")
        
        results = []
        
        for N in N_values:
            self.log(f"N={N}...")
            
            seq = np.random.choice([-1, 1], size=N).tolist()

            # CPU timing
            cpu_times = []
            for _ in range(n_iterations):
                t0 = time()
                best_seq, best_energy, history = run_mts(
                    N,
                    population_size=population_size,
                    generations=generations,
                    use_gpu=False,
                    batch_size=1,
                    verbose=False
                )
                cpu_times.append((time() - t0) * 1000)
            cpu_avg = np.mean(cpu_times)
            
            # GPU timing 
            gpu_times = []
            for _ in range(n_iterations):
                t0 = time()
                best_seq, best_energy, history = run_mts(
                    N,
                    population_size=population_size,
                    generations=generations,
                    use_gpu=True,
                    batch_size=64,
                    verbose=False
                )
                gpu_times.append((time() - t0) * 1000)
            gpu_avg = np.mean(gpu_times)
 
            speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 1.0
            
            result = SpeedupResult(
                N=N,
                cpu_time_ms=cpu_avg,
                gpu_time_ms=gpu_avg,
                speedup=speedup,
                n_iterations=n_iterations,
            )
            results.append(result)
            self.speedup_results.append(result)
            
            self.log(f"  CPU: {cpu_avg:.3f}ms, GPU: {gpu_avg:.3f}ms, Speedup: {speedup:.1f}x")
        
        return results
    
    # --------------------------------------------------------
    # Full Benchmark Suite
    # --------------------------------------------------------
    def run_full_benchmark(
        self,
        small_N: List[int] = [8, 10],
        medium_N: List[int] = [15, 20, 25],
        large_N: List[int] = [30, 35, 40],
        scale_N: int = 50,
        n_trials: int = 3,
        mts_generations: int = 100,
        timeout_hours: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Run complete benchmark suite with all metrics.
        
        Returns comprehensive results dictionary.
        """
        start_time = time()
        timeout_seconds = timeout_hours * 3600
        
        self.log("=" * 60)
        self.log("LABS Benchmark Suite - Full Run")
        self.log("=" * 60)
        
        all_results = {
            "small_N": {},
            "medium_N": {},
            "large_N": {},
            "scale_test": {},
            "speedup": [],
            "metrics": {},
        }
        
        # ---- Small N: All methods including CPU MTS ----
        self.log("\n### Small N Benchmark ###")
        for N in small_N:
            if time() - start_time > timeout_seconds:
                self.log("TIMEOUT reached!")
                break
                
            all_results["small_N"][N] = {"trials": []}
            
            for trial in range(n_trials):
                self.log(f"\n--- N={N}, Trial {trial+1}/{n_trials} ---")
                trial_results = {}
                
                # CPU MTS
                trial_results["MTS_CPU"] = self.run_classical_mts_cpu(
                    N, generations=mts_generations
                )
                
                # GPU MTS
                trial_results["MTS_GPU"] = self.run_classical_mts_gpu(
                    N, generations=mts_generations
                )
                
                # Naive QAOA + MTS
                trial_results["NaiveQAOA_MTS"] = self.run_naive_qaoa_mts(
                    N, mts_generations=mts_generations
                )
                
                # RQAOA + MTS
                trial_results["RQAOA_MTS"] = self.run_rqaoa_mts(
                    N, mts_generations=mts_generations,
                    record_trajectory=(trial == 0),
                    record_correlations=(trial == 0),
                )
                
                all_results["small_N"][N]["trials"].append(trial_results)
        
        # ---- Medium N ----
        self.log("\n### Medium N Benchmark ###")
        for N in medium_N:
            if time() - start_time > timeout_seconds:
                self.log("TIMEOUT reached!")
                break
                
            all_results["medium_N"][N] = {"trials": []}
            
            for trial in range(n_trials):
                self.log(f"\n--- N={N}, Trial {trial+1}/{n_trials} ---")
                trial_results = {}
                
                trial_results["MTS_GPU"] = self.run_classical_mts_gpu(
                    N, generations=mts_generations
                )
                
                trial_results["NaiveQAOA_MTS"] = self.run_naive_qaoa_mts(
                    N, mts_generations=mts_generations
                )
                
                trial_results["RQAOA_MTS"] = self.run_rqaoa_mts(
                    N, mts_generations=mts_generations,
                    record_trajectory=(trial == 0),
                )
                
                all_results["medium_N"][N]["trials"].append(trial_results)
        
        # ---- Large N: GPU MTS, RQAOA, Naive QAOA ----
        self.log("\n### Large N Benchmark ###")
        for N in large_N:
            if time() - start_time > timeout_seconds:
                self.log("TIMEOUT reached!")
                break
                
            all_results["large_N"][N] = {"trials": []}
            
            for trial in range(n_trials):
                self.log(f"\n--- N={N}, Trial {trial+1}/{n_trials} ---")
                trial_results = {}
                
                trial_results["MTS_GPU"] = self.run_classical_mts_gpu(
                    N, generations=mts_generations
                )
                
                trial_results["NaiveQAOA_MTS"] = self.run_naive_qaoa_mts(
                    N, mts_generations=mts_generations
                )
                
                trial_results["RQAOA_MTS"] = self.run_rqaoa_mts(
                    N, mts_generations=mts_generations
                )
                
                all_results["large_N"][N]["trials"].append(trial_results)
        
        # ---- Scale Test: N >= 30 ----
        self.log(f"\n### Scale Test (N={scale_N}) ###")
        if time() - start_time < timeout_seconds:
            self.log(f"Running N={scale_N} scale test...")
            scale_result = self.run_rqaoa_mts(
                scale_N,
                mts_generations=mts_generations * 2,
                mts_population_size=100,
            )
            all_results["scale_test"] = {
                "N": scale_N,
                "result": scale_result,
                "success": scale_result.energy > 0,
            }
        
        # ---- GPU Speedup Measurement ----
        self.log("\n### GPU Speedup Measurement ###")
        speedup_results = self.measure_gpu_speedup()
        all_results["speedup"] = speedup_results
        
        # ---- Compute Metrics ----
        total_time = time() - start_time
        all_results["metrics"] = self._compute_metrics(all_results)
        all_results["total_time_seconds"] = total_time
        
        self.log("\n" + "=" * 60)
        self.log("Benchmark Complete!")
        self.log(f"Total time: {total_time/60:.1f} minutes")
        self.log("=" * 60)
        
        return all_results
    
    def _compute_metrics(self, results: Dict) -> Dict[str, Any]:
        """Compute success metrics from benchmark results."""
        metrics = {
            "metric1_approximation": {},
            "metric2_superiority": {},
            "metric3_speedup": {},
            "metric4_scale": {},
        }
        
        # Metric 1: MF within 90% of best known for N <= 40
        for size_category in ["small_N", "medium_N", "large_N"]:
            for N, data in results.get(size_category, {}).items():
                if N > 40:
                    continue
                best_known = get_best_known_mf(N)
                if best_known is None:
                    continue
                
                for trial_data in data.get("trials", []):
                    for method, result in trial_data.items():
                        if method not in metrics["metric1_approximation"]:
                            metrics["metric1_approximation"][method] = []
                        
                        ratio = result.merit_factor / best_known
                        metrics["metric1_approximation"][method].append({
                            "N": N,
                            "ratio": ratio,
                            "passes": ratio >= 0.90,
                        })
        
        # Metric 2: RQAOA > Naive QAOA for N > 20
        for size_category in ["medium_N", "large_N"]:
            for N, data in results.get(size_category, {}).items():
                if N <= 20:
                    continue
                
                for trial_data in data.get("trials", []):
                    rqaoa = trial_data.get("RQAOA_MTS")
                    naive = trial_data.get("NaiveQAOA_MTS")
                    
                    if rqaoa and naive:
                        metrics["metric2_superiority"][N] = metrics["metric2_superiority"].get(N, [])
                        metrics["metric2_superiority"][N].append({
                            "rqaoa_mf": rqaoa.merit_factor,
                            "naive_mf": naive.merit_factor,
                            "passes": rqaoa.merit_factor > naive.merit_factor,
                        })
        
        # Metric 3: GPU speedup >= 20x
        speedups = results.get("speedup", [])
        if speedups:
            avg_speedup = np.mean([s.speedup for s in speedups])
            max_speedup = max(s.speedup for s in speedups)
            metrics["metric3_speedup"] = {
                "avg_speedup": avg_speedup,
                "max_speedup": max_speedup,
                "passes": max_speedup >= 20.0,
                "details": [(s.N, s.speedup) for s in speedups],
            }
        
        # Metric 4: N=50 success
        scale_test = results.get("scale_test", {})
        if scale_test:
            metrics["metric4_scale"] = {
                "N": scale_test.get("N"),
                "success": scale_test.get("success", False),
                "energy": scale_test.get("result", {}).energy if scale_test.get("result") else None,
                "time": scale_test.get("result", {}).time_seconds if scale_test.get("result") else None,
            }
        
        return metrics
    
    def save_results(self, filepath: str):
        """Save benchmark results to JSON file."""
        # Convert results to serializable format
        serializable = []
        for r in self.results:
            serializable.append({
                "method": r.method,
                "N": r.N,
                "energy": r.energy,
                "merit_factor": r.merit_factor,
                "time_seconds": r.time_seconds,
                "mf_ratio": r.mf_ratio,
                "extra_data": {k: v for k, v in r.extra_data.items() 
                              if not isinstance(v, (np.ndarray, list)) or len(str(v)) < 1000},
            })
        
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        self.log(f"Results saved to {filepath}")
