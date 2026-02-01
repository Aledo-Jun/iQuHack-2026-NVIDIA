# src/solvers/mts.py
# Classical Solvers: Memetic Tabu Search (MTS)
# Fully Optimized for NVIDIA L4 (24GB VRAM)

from typing import List, Tuple, Optional, Union
import numpy as np
import os

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from utils.labs_utils import calculate_energy

# ============================================================
# Hardware Awareness & Auto-Tuning (Ad-hoc; might not needed)
# ============================================================

def select_best_gpu(verbose: bool = False):
    """
    Automatically selects the GPU with the most free memory.
    Useful since you have 2x L4s and one might be busy.
    """
    if not CUPY_AVAILABLE or cp is None:
        return
        
    try:
        n_devices = cp.cuda.runtime.getDeviceCount()
        if n_devices <= 1:
            return # Only one GPU, nothing to choose

        # Check free memory on all devices
        max_free = -1
        best_dev = 0
        
        for i in range(n_devices):
            with cp.cuda.Device(i):
                free_mem, _ = cp.cuda.runtime.memGetInfo()
                if free_mem > max_free:
                    max_free = free_mem
                    best_dev = i
        
        # Set the device
        cp.cuda.Device(best_dev).use()
        if verbose:
            print(f"MTS: Auto-selected GPU {best_dev} (Free Memory: {max_free / 1024**3:.2f} GB)")
            
    except Exception as e:
        if verbose:
            print(f"MTS: GPU auto-selection failed: {e}")

def get_optimal_config(N: int, user_batch_size: Optional[int] = None) -> Tuple[bool, int]:
    """
    Determines optimal backend (CPU/GPU) and batch size based on N.
    Tuned specifically for NVIDIA L4 (24GB VRAM, 7424 Cores).
    """
    # L4 is fast, but kernel launch latency dominates for tiny N.
    # N < 15: CPU is faster in this regime(verified via quick benchmark)
    GPU_MIN_N = 15
    
    if user_batch_size is None:
        if N < GPU_MIN_N:
            # CPU for trivial problems
            return False, 1
            
        # --- L4 TUNING ---
        # We need ~100k+ threads to hide latency and saturate 7424 cores.
        # Batch Size logic: Target ~100k - 200k total threads (Batch * N)
        if N < 25:
            return True, 8192
            
        elif N < 30:
            return True, 4096
            
        else:
            return True, 2048
            
    # Respect user force, but enforce GPU min threshold unless forced externally
    return (N >= GPU_MIN_N), user_batch_size

# ============================================================
# CPU Helper Functions (Legacy / Fallback)
# ============================================================

def combine(parent1: List[int], parent2: List[int]) -> List[int]:
    """Single-point crossover (CPU)."""
    idx = np.random.randint(0, len(parent1))
    return parent1[:idx] + parent2[idx:]

def mutate(sequence: List[int], mutation_rate: float) -> List[int]:
    """Bit-flip mutation (CPU)."""
    new_seq = sequence.copy()
    for i in range(len(new_seq)):
        if np.random.random() < mutation_rate:
            new_seq[i] *= -1
    return new_seq

def tournament_selection(population: List[List[int]], energies: List[int], k: int = 3) -> List[int]:
    """Tournament selection (CPU)."""
    selected_indices = np.random.choice(len(population), k, replace=False)
    tournament_energies = [energies[i] for i in selected_indices]
    winner_idx_in_tournament = np.argmin(tournament_energies)
    return population[selected_indices[winner_idx_in_tournament]]

# ============================================================
# Main Runner
# ============================================================

def run_mts(
    N: int,
    population_size: int = 100,
    generations: int = 100,
    initial_population: Optional[List[List[int]]] = None,
    recombination_rate: float = 0.9,
    verbose: bool = True,
    use_gpu: bool = False,
    batch_size: Optional[int] = None
) -> Tuple[List[int], int, List[int]]:
    """
    Memetic Tabu Search runner.
    """
    
    # --- DYNAMIC CONFIGURATION ---
    # Auto-select best GPU if we are about to use one
    if use_gpu and CUPY_AVAILABLE and cp is not None:
        # Only run selection once per process effectively
        if not hasattr(run_mts, '_gpu_selected'):
            select_best_gpu(verbose)
            run_mts._gpu_selected = True

    # Calculate optimal settings based on N and L4 specs
    should_use_gpu, optimal_batch = get_optimal_config(N, batch_size)
    
    # Apply configuration
    # Only override use_gpu if user didn't force a batch size (implying they want auto-config)
    if batch_size is None:
        batch_size = optimal_batch
        # Note: We rely on the caller's 'use_gpu' preference, but we use the optimal batch
        # if they *do* use GPU. 
    
    # --------------------------------------------------------
    # GPU Branch
    # --------------------------------------------------------
    if use_gpu and CUPY_AVAILABLE and cp is not None:
        if should_use_gpu or batch_size > 1: # Run if efficient OR if user forced large batch
            if verbose:
                print(f"MTS: GPU Optimization (N={N}, Batch={batch_size})")
            
            # Initialize GPU Solver
            gpu_solver = GPUAcceleratedMTS(N)
            
            return _run_mts_gpu(
                gpu_solver, 
                N, 
                population_size, 
                generations, 
                initial_population, 
                recombination_rate, 
                batch_size, 
                verbose
            )
        elif verbose:
            print(f"MTS: N={N} is too small for GPU overhead. Switching to CPU.")

    # --------------------------------------------------------
    # CPU Branch (Fallback)
    # --------------------------------------------------------
    if use_gpu and (not CUPY_AVAILABLE or cp is None) and verbose:
        print("MTS: GPU requested but CuPy not found. Falling back to CPU.")
            
    # Normalize batch size for CPU (always 1)
    if batch_size is None:
        batch_size = 1
        
    mutation_rate = 1.0 / N
    
    # Initialize Population
    if initial_population is None:
        population = [
            np.random.choice([-1, 1], size=N).tolist() 
            for _ in range(population_size)
        ]
    else:
        population = [seq.copy() for seq in initial_population]
    
    # Evaluate initial population
    energies = [calculate_energy(p) for p in population]
    
    # Track Global Best
    min_idx = int(np.argmin(energies))
    best_global_seq = population[min_idx].copy()
    best_global_energy = energies[min_idx]
    
    history = [best_global_energy]
    
    for gen in range(generations):
        # 1. Generate Batch of Children
        offspring_batch = []
        
        for _ in range(batch_size):
            p1 = tournament_selection(population, energies)
            p2 = tournament_selection(population, energies)
            
            if np.random.random() < recombination_rate:
                child = combine(p1, p2)
            else:
                child = population[np.random.randint(0, len(population))].copy()
            
            child = mutate(child, mutation_rate=mutation_rate)
            offspring_batch.append(child)
            
        # 2. Optimize Batch
        opt_children = []
        opt_energies = []
        for child in offspring_batch:
            c, e = tabu_search(child)
            opt_children.append(c)
            opt_energies.append(e)
        
        # 3. Update Population & Global Best
        for child, energy in zip(opt_children, opt_energies):
            if energy < best_global_energy:
                best_global_energy = energy
                best_global_seq = child
                if verbose:
                    print(f"Gen {gen}: New Best Energy = {best_global_energy}")
            
            # Random Replacement
            replace_idx = np.random.randint(0, population_size)
            population[replace_idx] = child
            energies[replace_idx] = energy
        
        history.append(best_global_energy)
            
    return best_global_seq, best_global_energy, history

# ============================================================
# Internal GPU Runner (Resident Memory)
# ============================================================

def _run_mts_gpu(
    solver: 'GPUAcceleratedMTS',
    N: int,
    pop_size: int,
    generations: int,
    initial_pop: Optional[List[List[int]]],
    rec_rate: float,
    batch_size: int,
    verbose: bool
) -> Tuple[List[int], int, List[int]]:
    """
    Executes the MTS loop entirely on the GPU to avoid PCIe bottleneck.
    """
    # 1. Initialize Population on GPU
    if initial_pop is None:
        # Random initialization directly on GPU
        pop_gpu = cp.random.choice(cp.asarray([-1, 1], dtype=cp.int8), size=(pop_size, N))
    else:
        # Pad or slice provided population
        if len(initial_pop) < pop_size:
            # Fill remainder with random
            provided = cp.asarray(initial_pop, dtype=cp.int8)
            needed = pop_size - len(initial_pop)
            random_fill = cp.random.choice(cp.asarray([-1, 1], dtype=cp.int8), size=(needed, N))
            pop_gpu = cp.vstack([provided, random_fill])
        else:
            pop_gpu = cp.asarray(initial_pop[:pop_size], dtype=cp.int8)
            
    # Calculate initial energies
    energies_gpu = solver.compute_energies(pop_gpu)
    
    # Track Global Best
    best_idx = cp.argmin(energies_gpu)
    best_global_energy = float(energies_gpu[best_idx])
    best_global_seq = pop_gpu[best_idx].copy()
    
    history = [best_global_energy]
    mutation_rate = 1.0 / N
    
    for gen in range(generations):
        # --- Batch Generation (Vectorized) ---
        
        # 1. Tournament Selection for parents (Batch x 2)
        k = 3
        candidates_idx = cp.random.randint(0, pop_size, size=(batch_size, 2, k))
        
        # Gather energies: (Batch, 2, k)
        cand_energies = energies_gpu[candidates_idx]
        
        # Find winner index in k-dimension: (Batch, 2)
        winners_local_idx = cp.argmin(cand_energies, axis=2)
        
        # Convert to global population indices (Flattened indexing)
        flat_cand_idx = candidates_idx.reshape(-1, k)
        flat_win_local = winners_local_idx.reshape(-1)
        flat_batch_range = cp.arange(len(flat_win_local))
        
        parent_indices_flat = flat_cand_idx[flat_batch_range, flat_win_local]
        parent_indices = parent_indices_flat.reshape(batch_size, 2)
        
        p1 = pop_gpu[parent_indices[:, 0]]
        p2 = pop_gpu[parent_indices[:, 1]]
        
        # 2. Crossover (Vectorized)
        do_crossover = cp.random.random(batch_size) < rec_rate
        x_points = cp.random.randint(1, N, size=batch_size)
        
        col_indices = cp.arange(N)[cp.newaxis, :]
        mask_cross = col_indices < x_points[:, cp.newaxis]
        
        children = cp.where(
            do_crossover[:, cp.newaxis], 
            cp.where(mask_cross, p1, p2), 
            p1 
        )
        
        # 3. Mutation (Vectorized)
        mut_mask = cp.random.random((batch_size, N)) < mutation_rate
        children = cp.where(mut_mask, children * -1, children)
        
        # --- GPU Tabu Search ---
        opt_children, opt_energies = solver.run_tabu_search_batch_resident(children)
        
        # --- Population Update ---
        
        batch_best_idx = cp.argmin(opt_energies)
        batch_best_energy = float(opt_energies[batch_best_idx])
        
        if batch_best_energy < best_global_energy:
            best_global_energy = batch_best_energy
            best_global_seq = opt_children[batch_best_idx].copy()
            if verbose:
                print(f"  Gen {gen}: New Best Energy = {best_global_energy}")
        
        history.append(best_global_energy)
        
        # Random Replacement in Population
        replace_indices = cp.random.randint(0, pop_size, size=batch_size)
        pop_gpu[replace_indices] = opt_children
        energies_gpu[replace_indices] = opt_energies

    if hasattr(best_global_seq, 'get'):
        final_seq = best_global_seq.get().tolist()
    else:
        final_seq = best_global_seq.tolist()
        
    return final_seq, int(best_global_energy), history


# ============================================================
# Classical Local Search (CPU)
# ============================================================

def tabu_search(
    initial_sequence: List[int],
    max_iter: int = 1000
) -> Tuple[List[int], int]:
    """Standard Tabu Search (CPU)."""
    N = len(initial_sequence)
    current_seq = initial_sequence.copy()
    current_energy = calculate_energy(current_seq)
    
    best_seq = current_seq.copy()
    best_energy = current_energy
    
    tabu_list = np.zeros(N, dtype=np.int32)
    
    for t in range(1, max_iter + 1):
        best_neighbor_energy = float('inf')
        best_move = -1
        
        for i in range(N):
            current_seq[i] *= -1
            e = calculate_energy(current_seq)
            is_tabu = tabu_list[i] >= t
            is_better = e < best_energy 
            
            if (not is_tabu) or is_better:
                if e < best_neighbor_energy:
                    best_neighbor_energy = e
                    best_move = i
            current_seq[i] *= -1
            
        if best_move != -1:
            current_seq[best_move] *= -1
            current_energy = best_neighbor_energy
            tenure = np.random.randint(2, 9)
            tabu_list[best_move] = t + tenure
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_seq = current_seq[:]
                
    return best_seq, best_energy

def tabu_search_optimized(
    initial_sequence: List[int],
    gpu_solver = None
) -> Tuple[List[int], int]:
    """Wrapper to run optimized tabu search (legacy hook)."""
    if gpu_solver and CUPY_AVAILABLE and cp is not None:
        res, eng = gpu_solver.run_tabu_search_batch([initial_sequence], n_iter=200)
        return res[0], eng[0]
        
    return tabu_search(initial_sequence)


# ============================================================
# GPU Accelerated Solver
# ============================================================

class GPUAcceleratedMTS:
    """
    High-performance GPU solver for LABS using batched operations and RawKernels.
    """
    
    def __init__(self, N: int):
        if not CUPY_AVAILABLE or cp is None:
             raise ImportError("CuPy is required for GPUAcceleratedMTS")
            
        self.N = N
        self._flip_mask = 1 - 2 * cp.eye(N, dtype=cp.int8)
        
        if hasattr(cp, 'RawKernel'):
            self._energy_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void labs_energy(const signed char* seqs, float* energies, int B, int N) {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                if (idx >= B) return;

                const signed char* my_seq = seqs + (idx * N);
                float total_energy = 0.0f;

                for (int k = 1; k < N; ++k) {
                    int correlation = 0;
                    for (int i = 0; i < N - k; ++i) {
                        correlation += my_seq[i] * my_seq[i + k];
                    }
                    total_energy += (float)(correlation * correlation);
                }
                energies[idx] = total_energy;
            }
            ''', 'labs_energy')
        else:
            self._energy_kernel = None
        
    def compute_energies(self, sequences_gpu: 'cp.ndarray') -> 'cp.ndarray':
        """
        Fast GPU Energy Calculation using custom CUDA kernel.
        """
        B, N = sequences_gpu.shape
        energies = cp.zeros(B, dtype=cp.float32)
        
        # Fallback for mocking/tests
        if isinstance(sequences_gpu, np.ndarray):
            for idx in range(B):
                seq = sequences_gpu[idx]
                total_energy = 0.0
                for k in range(1, N):
                    correlation = np.sum(seq[:N-k] * seq[k:])
                    total_energy += correlation ** 2
                energies[idx] = total_energy
            return energies

        # Real GPU Path
        threads_per_block = 256
        blocks = (B + threads_per_block - 1) // threads_per_block
        
        if self._energy_kernel:
            self._energy_kernel(
                (blocks,), (threads_per_block,),
                (sequences_gpu, energies, B, N)
            )
        
        return energies

    def run_tabu_search_batch_resident(
        self,
        current_seqs: 'cp.ndarray',
        n_iter: int = 200,
        tenure_min: int = 2,
        tenure_max: int = 8
    ) -> Tuple['cp.ndarray', 'cp.ndarray']:
        """GPU-Resident Tabu Search."""
        B, N = current_seqs.shape
        
        best_seqs = current_seqs.copy()
        current_energies = self.compute_energies(current_seqs)
        best_energies = current_energies.copy()
        
        tabu_list = cp.zeros((B, N), dtype=cp.int32)
        
        for t in range(1, n_iter + 1):
            neighbors = current_seqs[:, cp.newaxis, :] * self._flip_mask[cp.newaxis, :, :]
            neighbors_flat = neighbors.reshape(B * N, N)
            
            neighbor_energies_flat = self.compute_energies(neighbors_flat)
            neighbor_energies = neighbor_energies_flat.reshape(B, N)
            
            best_energies_broad = best_energies[:, cp.newaxis]
            aspiration_mask = neighbor_energies < best_energies_broad
            is_tabu = tabu_list >= t
            
            valid_move_mask = (~is_tabu) | aspiration_mask
            
            penalized_energies = cp.where(valid_move_mask, neighbor_energies, cp.inf)
            
            best_move_indices = cp.argmin(penalized_energies, axis=1) 
            best_move_energies = cp.min(penalized_energies, axis=1)   
            
            batch_indices = cp.arange(B)
            
            current_seqs[batch_indices, best_move_indices] *= -1
            current_energies = best_move_energies
            
            tenures = cp.random.randint(tenure_min, tenure_max + 1, size=B, dtype=cp.int32)
            tabu_list[batch_indices, best_move_indices] = t + tenures
            
            improved_mask = current_energies < best_energies
            if cp.any(improved_mask):
                mask_broad = improved_mask[:, cp.newaxis]
                best_seqs = cp.where(mask_broad, current_seqs, best_seqs)
                best_energies = cp.where(improved_mask, current_energies, best_energies)

        return best_seqs, best_energies

    def run_tabu_search_batch(
        self, 
        initial_sequences: List[List[int]], 
        n_iter: int = 200
    ) -> Tuple[List[List[int]], List[int]]:
        """Legacy Wrapper for CPU-list input."""
        seqs_gpu = cp.asarray(initial_sequences, dtype=cp.int8)
        final_seqs_gpu, final_energies_gpu = self.run_tabu_search_batch_resident(seqs_gpu, n_iter=n_iter)
        
        if hasattr(final_seqs_gpu, 'get'):
            final_seqs = final_seqs_gpu.get().tolist()
            final_energies = final_energies_gpu.get().tolist()
        else:
            final_seqs = final_seqs_gpu.tolist()
            final_energies = final_energies_gpu.tolist()
            
        return final_seqs, final_energies