# Hybrid Quantum-Classical Solver for LABS
# Combines RQAOA (Quantum initialization) with MTS (Classical local search)

from typing import List, Tuple, Dict, Any, Literal
import numpy as np
from time import time

from algorithms.rqaoa import *
from solvers.mts import *

# ============================================================
# Improved Hybrid Solver
# ============================================================
def solve_hybrid(
    N: int,
    quantum_method: Literal['rqaoa', 'qaoa'] = 'rqaoa',
    depth: int = 2,
    shots: int = 2000,
    mts_generations: int = 100,
    mts_population_size: int = 50,
    n_qaoa_samples: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Hybrid solver with:
    - QAOA parameter optimization
    - Multiple QAOA samples for seeding
    - GPU-accelerated MTS with persistent memory
    """
    from time import time
    start = time()
    
    # Quantum phase
    if quantum_method == 'rqaoa':
        if verbose:
            print(f"=== Quantum Phase (RQAOA depth={depth}) ===")

        result = solve_rqaoa(N, depth=depth, shots=shots, verbose=verbose)

        if verbose:
            print(f"RQAOA: E={result.energy}, MF={result.merit_factor:.2f}")
        
    elif quantum_method == 'qaoa':
        if verbose:
            print(f"=== Quantum Phase (QAOA depth={depth}) ===")

        result = solve_naive_qaoa(N, depth=depth, shots=shots, verbose=verbose)
        
        if verbose:
            print(f"QAOA: E={result.energy}, MF={result.merit_factor:.2f}")
        
    quantum_time = time() - start
    
    # Build diverse initial population
    seed = result.sequence
    population = [seed.copy()]
    
    # Add perturbations (1-3 bit flips)
    n_perturbed = mts_population_size // 2
    for _ in range(n_perturbed):
        new_seq = seed.copy()
        n_flips = np.random.randint(1, 4)
        flip_idx = np.random.choice(N, n_flips, replace=False)
        for idx in flip_idx:
            new_seq[idx] *= -1
        population.append(new_seq)
    
    # Fill rest with random for diversity
    while len(population) < mts_population_size:
        population.append(np.random.choice([-1, 1], size=N).tolist())
    
    # Classical phase with GPU
    if verbose:
        print(f"=== Classical Phase (MTS gen={mts_generations}) ===")
    
    gpu_solver = None
    if CUPY_AVAILABLE:
        try:
            gpu_solver = GPUAcceleratedMTS(N)
        except ImportError:
            gpu_solver = None
    
    energies = [calculate_energy(p) for p in population]
    
    best_idx = int(np.argmin(energies))
    best_seq = population[best_idx].copy()
    best_energy = energies[best_idx]
    
    for gen in range(mts_generations):
        # Tournament selection
        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
        p1 = population[idx1] if energies[idx1] < energies[idx2] else population[idx2]
        
        idx3, idx4 = np.random.choice(len(population), 2, replace=False)
        p2 = population[idx3] if energies[idx3] < energies[idx4] else population[idx4]
        
        # Crossover
        xover_pt = np.random.randint(0, N)
        child = p1[:xover_pt] + p2[xover_pt:]
        
        # Mutation
        for i in range(N):
            if np.random.random() < 1.0 / N:
                child[i] *= -1
        
        # Local search
        improved, imp_energy = tabu_search_optimized(child, gpu_solver)
        
        if imp_energy < best_energy:
            best_energy = imp_energy
            best_seq = improved.copy()
            if verbose:
                print(f"  Gen {gen}: E={best_energy}, MF={(N**2)/(2*best_energy):.2f}")
        
        # Replace random individual
        replace_idx = np.random.randint(len(population))
        population[replace_idx] = improved
        energies[replace_idx] = imp_energy
    
    total_time = time() - start
    final_mf = (N ** 2) / (2 * best_energy) if best_energy > 0 else float('inf')
    
    return {
        "N": N,
        "quantum_method": quantum_method,
        "final_sequence": best_seq,
        "final_energy": best_energy,
        "final_merit_factor": final_mf,
        "quantum_seed_energy": result.energy,
        "quantum_time": quantum_time,
        "total_time": total_time,
    }