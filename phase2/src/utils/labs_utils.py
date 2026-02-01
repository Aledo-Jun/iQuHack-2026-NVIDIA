# LABS Utilities for RQAOA Implementation
# Based on the formulation in the iQuHack 2026 NVIDIA challenge

from typing import List, Tuple
import numpy as np


def calculate_energy(sequence: List[int]) -> int:
    """
    Compute LABS energy E(s) = sum_{k=1}^{N-1} C_k^2
    where C_k = sum_{i=1}^{N-k} s_i * s_{i+k}
    
    Args:
        sequence: Binary sequence of +1/-1 values
        
    Returns:
        Energy value (lower is better)
    """
    N = len(sequence)
    E = 0
    for k in range(1, N):
        Ck = 0
        for i in range(N - k):
            Ck += sequence[i] * sequence[i + k]
        E += Ck ** 2
    return E


def calculate_merit_factor(sequence: List[int]) -> float:
    """
    Compute Merit Factor MF = N^2 / (2 * E)
    Higher is better.
    """
    N = len(sequence)
    E = calculate_energy(sequence)
    if E == 0:
        return float('inf')
    return (N ** 2) / (2 * E)


def sequence_to_binary(sequence: List[int]) -> str:
    """Convert +1/-1 sequence to binary string (1/0)."""
    return ''.join(['1' if s == 1 else '0' for s in sequence])


def binary_to_sequence(bitstring: str) -> List[int]:
    """Convert binary string (1/0) to +1/-1 sequence."""
    return [1 if c == '1' else -1 for c in bitstring]


def get_labs_hamiltonian_coefficients(N: int) -> Tuple[dict, dict]:
    """
    Generate the LABS Hamiltonian coefficients for use in QAOA/RQAOA.
    
    H_LABS = sum_{i,j} J_ij Z_i Z_j + sum_{ijkl} K_ijkl Z_i Z_j Z_k Z_l
    
    For RQAOA, we primarily need the 2-body (J) and 4-body (K) terms.
    
    Returns:
        J: Dict of (i, j) -> coefficient for 2-body ZZ terms
        K: Dict of (i, j, k, l) -> coefficient for 4-body ZZZZ terms
    """
    J = {}  # 2-body terms
    K = {}  # 4-body terms
    
    # The LABS Hamiltonian from the paper:
    # H_f = 2 * sum_{i=0}^{N-3} Z_i * sum_{k=1}^{floor((N-i-1)/2)} Z_{i+k}
    #     + 4 * sum_{i=0}^{N-4} Z_i * sum_{t=1}^{floor((N-i-2)/2)} sum_{k=t+1}^{N-i-t-1} Z_{i+t} Z_{i+k} Z_{i+k+t}
    
    # 2-body terms (ZZ interactions)
    for i in range(N - 2):
        limit_k = (N - i - 1) // 2
        for k in range(1, limit_k + 1):
            j = i + k
            key = (min(i, j), max(i, j))
            J[key] = J.get(key, 0) + 2.0
    
    # 4-body terms (ZZZZ interactions)
    for i in range(N - 3):
        limit_t = (N - i - 2) // 2
        for t in range(1, limit_t + 1):
            limit_k = N - i - t - 1
            for k in range(t + 1, limit_k + 1):
                # Indices: i, i+t, i+k, i+k+t
                indices = tuple(sorted([i, i + t, i + k, i + k + t]))
                K[indices] = K.get(indices, 0) + 4.0
    
    return J, K


def verify_symmetries(sequence: List[int]) -> bool:
    """
    Verify LABS symmetries:
    - E(S) == E(reversed(S))
    - E(S) == E(-S)
    
    Returns True if all symmetries hold.
    """
    E_original = calculate_energy(sequence)
    
    # Check reversal symmetry
    reversed_seq = sequence[::-1]
    E_reversed = calculate_energy(reversed_seq)
    if E_original != E_reversed:
        return False
    
    # Check negation symmetry
    negated_seq = [-s for s in sequence]
    E_negated = calculate_energy(negated_seq)
    if E_original != E_negated:
        return False
    
    return True


def brute_force_optimal(N: int) -> Tuple[List[int], int]:
    """
    Brute-force search for optimal LABS sequence of length N.
    Only practical for N <= 10.
    
    Returns:
        (optimal_sequence, min_energy)
    """
    if N > 15:
        raise ValueError(f"Brute force not practical for N={N} > 15")
    
    best_seq = None
    best_energy = float('inf')
    
    for i in range(2 ** N):
        # Generate sequence from integer
        seq = [1 if (i >> j) & 1 else -1 for j in range(N)]
        energy = calculate_energy(seq)
        if energy < best_energy:
            best_energy = energy
            best_seq = seq
    
    return best_seq, best_energy


# Known optimal energies for small N (for validation)
KNOWN_OPTIMAL_ENERGIES = {
    3: 1,
    4: 2,
    5: 2,
    6: 7,
    7: 3,
    8: 8,
    9: 12,
    10: 13,
}
