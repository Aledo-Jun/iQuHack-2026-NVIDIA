# tests/test_rqaoa.py 

import pytest
import numpy as np
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.labs_utils import calculate_energy, get_labs_hamiltonian_coefficients
from algorithms.rqaoa import (
    RQAOA,
    RQAOAResult,
    ReducedHamiltonian,
    QAOAOptimizer,
    solve_rqaoa,
)


class TestReducedHamiltonian:
    """Test ReducedHamiltonian dataclass."""
    
    def test_creation(self):
        """Test basic creation of ReducedHamiltonian."""
        ham = ReducedHamiltonian(
            J={(0, 1): 1.0, (1, 2): -0.5},
            K={(0, 1, 2, 3): 0.25},
            constant=0.0,
            n_qubits=4
        )
        assert len(ham.J) == 2
        assert len(ham.K) == 1
        assert ham.n_qubits == 4
    
    def test_empty_hamiltonian(self):
        """Test creation with empty terms."""
        ham = ReducedHamiltonian(
            J={},
            K={},
            constant=1.5,
            n_qubits=2
        )
        assert len(ham.J) == 0
        assert len(ham.K) == 0
        assert ham.constant == 1.5


class TestRQAOADecimation:
    """Test RQAOA decimation logic."""
    
    @pytest.fixture
    def rqaoa_solver(self):
        return RQAOA(depth=1, shots=100, classical_threshold=4)
    
    def test_map_to_reduced_indices(self, rqaoa_solver):
        """Test index remapping for active variables."""
        ham = ReducedHamiltonian(
            J={(0, 1): 1.0, (1, 2): 0.5, (0, 2): -0.3, (2, 3): 0.7},
            K={(0, 1, 2, 3): 0.1},
            constant=0.0,
            n_qubits=4
        )
        active_vars = [0, 2, 3]
        
        reduced = rqaoa_solver._map_to_reduced_indices(ham, active_vars)
        
        assert reduced.n_qubits == 3
        assert (0, 1) in reduced.J
        assert (1, 2) in reduced.J
    
    def test_apply_decimation_2body(self, rqaoa_solver):
        """Test decimation of 2-body terms."""
        ham = ReducedHamiltonian(
            J={(0, 1): 1.0, (1, 2): 0.5, (0, 2): -0.3},
            K={},
            constant=0.0,
            n_qubits=3
        )
        
        result = rqaoa_solver._apply_decimation(ham, elim=0, ref=1, sign=1)
        
        assert result.constant == 1.0
        assert (1, 2) in result.J
        assert result.n_qubits == 2
    
    def test_apply_decimation_2body_negative_sign(self, rqaoa_solver):
        """Test decimation with negative sign."""
        ham = ReducedHamiltonian(
            J={(0, 1): 1.0, (0, 2): 0.5},
            K={},
            constant=0.0,
            n_qubits=3
        )
        
        result = rqaoa_solver._apply_decimation(ham, elim=0, ref=1, sign=-1)
        
        assert result.constant == -1.0
        assert (1, 2) in result.J
        assert result.J[(1, 2)] == -0.5
    
    def test_apply_decimation_4body_to_2body(self, rqaoa_solver):
        """Test that 4-body terms reduce to 2-body when ref appears."""
        ham = ReducedHamiltonian(
            J={},
            K={(0, 1, 2, 3): 1.0},
            constant=0.0,
            n_qubits=4
        )
        
        result = rqaoa_solver._apply_decimation(ham, elim=0, ref=1, sign=1)
        
        assert len(result.K) == 0
        assert (2, 3) in result.J
        assert result.J[(2, 3)] == 1.0
    
    def test_apply_decimation_4body_remains(self, rqaoa_solver):
        """Test that 4-body terms stay 4-body when ref doesn't appear."""
        ham = ReducedHamiltonian(
            J={},
            K={(0, 2, 3, 4): 1.0},
            constant=0.0,
            n_qubits=5
        )
        
        result = rqaoa_solver._apply_decimation(ham, elim=0, ref=1, sign=1)
        
        assert (1, 2, 3, 4) in result.K
        assert result.K[(1, 2, 3, 4)] == 1.0
    
    def test_apply_decimation_removes_zeros(self, rqaoa_solver):
        """Test that zero coefficients are removed."""
        ham = ReducedHamiltonian(
            J={(0, 1): 1.0, (0, 2): -1.0, (1, 2): 1.0},
            K={},
            constant=0.0,
            n_qubits=3
        )
        
        result = rqaoa_solver._apply_decimation(ham, elim=0, ref=1, sign=1)
        
        for coeff in result.J.values():
            assert abs(coeff) > 1e-10
        for coeff in result.K.values():
            assert abs(coeff) > 1e-10


class TestQAOAOptimizer:
    """Test QAOA parameter optimization."""
    
    def test_optimizer_creation(self):
        """Test QAOAOptimizer initialization."""
        ham = ReducedHamiltonian(
            J={(0, 1): 1.0},
            K={},
            constant=0.0,
            n_qubits=2
        )
        optimizer = QAOAOptimizer(ham, depth=1, shots=100, max_opt_iter=10)
        
        assert optimizer.depth == 1
        assert optimizer.shots == 100
        assert optimizer.n_qubits == 2
    
    def test_edge_list_preparation(self):
        """Test that edge lists are correctly prepared."""
        ham = ReducedHamiltonian(
            J={(0, 1): 1.0, (1, 2): 0.5},
            K={(0, 1, 2, 3): 0.25},
            constant=0.0,
            n_qubits=4
        )
        optimizer = QAOAOptimizer(ham, depth=1, shots=100)
        
        assert optimizer.n_edges_2 == 2
        assert optimizer.n_edges_4 == 1
        assert len(optimizer.e2_i) == 2
        assert len(optimizer.e4_i) == 1
    
    def test_hamiltonian_energy_computation(self):
        """Test energy computation from Hamiltonian."""
        ham = ReducedHamiltonian(
            J={(0, 1): 1.0},
            K={},
            constant=0.5,
            n_qubits=2
        )
        optimizer = QAOAOptimizer(ham, depth=1, shots=100)
        
        energy = optimizer._compute_hamiltonian_energy([1, 1])
        assert energy == 1.5
        
        energy = optimizer._compute_hamiltonian_energy([1, -1])
        assert energy == -0.5
    
    def test_classical_simulation_fallback(self):
        """Test classical simulation produces valid samples."""
        ham = ReducedHamiltonian(
            J={(0, 1): 0.5},
            K={},
            constant=0.0,
            n_qubits=2
        )
        optimizer = QAOAOptimizer(ham, depth=1, shots=100)
        
        samples = optimizer._run_qaoa_classical_sim([0.1], [0.2])
        
        assert isinstance(samples, dict)
        total_counts = sum(samples.values())
        assert total_counts == 100
        for bitstring in samples.keys():
            assert len(bitstring) == 2
            assert all(b in '01' for b in bitstring)
    
    def test_optimization_runs_classical(self):
        """Test that optimization completes without error (classical fallback)."""
        ham = ReducedHamiltonian(
            J={(0, 1): 0.5},
            K={},
            constant=0.0,
            n_qubits=2
        )
        # Force classical simulation by mocking CUDAQ_AVAILABLE
        optimizer = QAOAOptimizer(ham, depth=1, shots=50, max_opt_iter=5)
        
        # Temporarily disable CUDA-Q to use classical sim
        import algorithms.rqaoa as rqaoa_module
        original_cudaq = rqaoa_module.CUDAQ_AVAILABLE
        rqaoa_module.CUDAQ_AVAILABLE = False
        
        try:
            beta, gamma, samples = optimizer.optimize()
            
            assert len(beta) == 1
            assert len(gamma) == 1
            assert isinstance(samples, dict)
        finally:
            rqaoa_module.CUDAQ_AVAILABLE = original_cudaq


class TestRQAOASolver:
    """Test full RQAOA solver."""
    
    def test_solve_small_n_classical_only(self):
        """For N <= classical_threshold, should solve classically."""
        result = solve_rqaoa(N=4, depth=1, classical_threshold=6, verbose=False)
        
        assert len(result.sequence) == 4
        assert result.energy == calculate_energy(result.sequence)
        assert all(s in [-1, 1] for s in result.sequence)
        assert result.n_qaoa_iterations == 0
    
    def test_solve_with_decimation_classical(self):
        """Test RQAOA with decimation steps (classical fallback)."""
        import algorithms.rqaoa as rqaoa_module
        original_cudaq = rqaoa_module.CUDAQ_AVAILABLE
        rqaoa_module.CUDAQ_AVAILABLE = False
        
        try:
            result = solve_rqaoa(N=8, depth=1, classical_threshold=4, verbose=False)
            
            assert len(result.sequence) == 8
            assert result.energy == calculate_energy(result.sequence)
            assert result.n_qaoa_iterations >= 1
            assert len(result.decimation_history) >= 1
        finally:
            rqaoa_module.CUDAQ_AVAILABLE = original_cudaq
    
    def test_decimation_history_format_classical(self):
        """Test that decimation history has correct format (classical fallback)."""
        import algorithms.rqaoa as rqaoa_module
        original_cudaq = rqaoa_module.CUDAQ_AVAILABLE
        rqaoa_module.CUDAQ_AVAILABLE = False
        
        try:
            result = solve_rqaoa(N=10, depth=1, classical_threshold=5, verbose=False)
            
            for (i, j, corr) in result.decimation_history:
                assert isinstance(i, int)
                assert isinstance(j, int)
                assert isinstance(corr, float)
                assert -1.0 <= corr <= 1.0
        finally:
            rqaoa_module.CUDAQ_AVAILABLE = original_cudaq
    
    def test_merit_factor_consistency(self):
        """Test merit factor is computed correctly."""
        # Use small N that doesn't require decimation
        result = solve_rqaoa(N=5, depth=1, classical_threshold=6, verbose=False)
        
        expected_mf = (5 ** 2) / (2 * result.energy) if result.energy > 0 else float('inf')
        assert abs(result.merit_factor - expected_mf) < 1e-6
    
    def test_4body_decimation_flag_classical(self):
        """Test that 4-body decimation flag works (classical fallback)."""
        import algorithms.rqaoa as rqaoa_module
        original_cudaq = rqaoa_module.CUDAQ_AVAILABLE
        rqaoa_module.CUDAQ_AVAILABLE = False
        
        try:
            result_without = solve_rqaoa(
                N=8, depth=1, classical_threshold=4,
                use_4body_decimation=False, verbose=False
            )
            result_with = solve_rqaoa(
                N=8, depth=1, classical_threshold=4,
                use_4body_decimation=True, verbose=False
            )
            
            assert len(result_without.sequence) == 8
            assert len(result_with.sequence) == 8
        finally:
            rqaoa_module.CUDAQ_AVAILABLE = original_cudaq
    
    def test_deeper_qaoa_classical(self):
        """Test RQAOA with depth > 1 (classical fallback)."""
        import algorithms.rqaoa as rqaoa_module
        original_cudaq = rqaoa_module.CUDAQ_AVAILABLE
        rqaoa_module.CUDAQ_AVAILABLE = False
        
        try:
            result = solve_rqaoa(N=6, depth=2, classical_threshold=3, verbose=False)
            
            assert len(result.sequence) == 6
            assert result.energy == calculate_energy(result.sequence)
        finally:
            rqaoa_module.CUDAQ_AVAILABLE = original_cudaq


class TestCorrelationComputation:
    """Test correlation computation from samples."""
    
    def test_find_best_decimation(self):
        """Test finding strongest correlation."""
        solver = RQAOA(depth=1, shots=100, classical_threshold=4)
        
        corr_2 = {(0, 1): 0.8, (1, 2): -0.3, (0, 2): 0.5}
        corr_4 = {}
        
        pair, corr = solver._find_best_decimation(corr_2, corr_4)
        
        assert pair == (0, 1)
        assert corr == 0.8
    
    def test_find_best_decimation_negative(self):
        """Test finding strongest negative correlation."""
        solver = RQAOA(depth=1, shots=100, classical_threshold=4)
        
        corr_2 = {(0, 1): 0.3, (1, 2): -0.9, (0, 2): 0.5}
        corr_4 = {}
        
        pair, corr = solver._find_best_decimation(corr_2, corr_4)
        
        assert pair == (1, 2)
        assert corr == -0.9
    
    def test_find_best_decimation_with_4body_boost(self):
        """Test 4-body correlation boosting."""
        solver = RQAOA(depth=1, shots=100, classical_threshold=4, use_4body_decimation=True)
        
        # Without 4-body boost, (0,1) would win with 0.5
        # With 4-body boost for (0,2), it should win
        corr_2 = {(0, 1): 0.5, (0, 2): 0.45}
        corr_4 = {(0, 2, 3, 4): 0.8}  # Boosts (0,2) since both are in this term
        
        pair, corr = solver._find_best_decimation(corr_2, corr_4)
        
        # (0,2) gets boosted: 0.45 + 0.2*0.8 = 0.61 > 0.5
        assert pair == (0, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
