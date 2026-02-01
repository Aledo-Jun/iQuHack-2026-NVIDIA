# tests/test_hybrid.py

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.labs_utils import calculate_energy


class TestHybridSolver:
    """Test hybrid RQAOA + MTS solver."""
    
    # @pytest.fixture(autouse=True)
    # def disable_cudaq(self):
    #     """Disable CUDA-Q for all tests to use classical fallback."""
    #     import algorithms.rqaoa as rqaoa_module
    #     original = rqaoa_module.CUDAQ_AVAILABLE
    #     rqaoa_module.CUDAQ_AVAILABLE = False
    #     yield
    #     rqaoa_module.CUDAQ_AVAILABLE = original
    
    def test_hybrid_returns_valid_result(self):
        """Hybrid solver should return valid result dictionary."""
        from algorithms.hybrid import solve_hybrid
        
        result = solve_hybrid(
            N=4,
            depth=1,
            shots=100,
            mts_generations=5,
            mts_population_size=10,
            verbose=False
        )
        
        assert "final_sequence" in result
        assert "final_energy" in result
        assert "final_merit_factor" in result
        assert "quantum_seed_energy" in result
        assert "quantum_time" in result
        assert "total_time" in result
    
    def test_hybrid_sequence_valid(self):
        """Hybrid solver should return valid binary sequence."""
        from algorithms.hybrid import solve_hybrid
        
        N = 4
        result = solve_hybrid(
            N=N,
            depth=1,
            shots=100,
            mts_generations=5,
            mts_population_size=10,
            verbose=False
        )
        
        assert len(result["final_sequence"]) == N
        assert all(s in [-1, 1] for s in result["final_sequence"])
        assert result["final_energy"] == calculate_energy(result["final_sequence"])
    
    def test_hybrid_energy_consistency(self):
        """Energy and merit factor should be consistent."""
        from algorithms.hybrid import solve_hybrid
        
        N = 4
        result = solve_hybrid(
            N=N,
            depth=1,
            shots=100,
            mts_generations=5,
            mts_population_size=10,
            verbose=False
        )
        
        expected_mf = (N ** 2) / (2 * result["final_energy"])
        assert abs(result["final_merit_factor"] - expected_mf) < 1e-6
    
    def test_hybrid_timing(self):
        """Timing values should be positive."""
        from algorithms.hybrid import solve_hybrid
        
        result = solve_hybrid(
            N=4,
            depth=1,
            shots=50,
            mts_generations=3,
            mts_population_size=5,
            verbose=False
        )
        
        assert result["quantum_time"] > 0
        assert result["total_time"] > 0
        assert result["total_time"] >= result["quantum_time"]
    
    def test_hybrid_provides_seed(self):
        """RQAOA energy should be recorded."""
        from algorithms.hybrid import solve_hybrid
        
        result = solve_hybrid(
            N=4,
            depth=1,
            shots=100,
            mts_generations=10,
            mts_population_size=15,
            verbose=False
        )
        
        assert result["quantum_seed_energy"] is not None
        assert result["quantum_seed_energy"] > 0
    
    def test_hybrid_mts_improves_or_maintains(self):
        """Final energy should be <= RQAOA energy."""
        from algorithms.hybrid import solve_hybrid
        
        result = solve_hybrid(
            N=4,
            depth=1,
            shots=200,
            mts_generations=20,
            mts_population_size=20,
            verbose=False
        )
        
        assert result["final_energy"] <= result["quantum_seed_energy"]
    
    def test_hybrid_different_depths(self):
        """Hybrid should work with different QAOA depths."""
        from algorithms.hybrid import solve_hybrid
        
        for depth in [1, 2]:
            result = solve_hybrid(
                N=4,
                depth=depth,
                shots=100,
                mts_generations=5,
                mts_population_size=10,
                verbose=False
            )
            
            assert len(result["final_sequence"]) == 4
            assert result["final_energy"] == calculate_energy(result["final_sequence"])
    
    def test_hybrid_larger_n(self):
        """Test hybrid solver with larger N."""
        from algorithms.hybrid import solve_hybrid
        
        N = 12
        result = solve_hybrid(
            N=N,
            depth=1,
            shots=200,
            mts_generations=10,
            mts_population_size=15,
            verbose=False
        )
        
        assert len(result["final_sequence"]) == N
        assert result["final_energy"] == calculate_energy(result["final_sequence"])


class TestHybridPopulationSeeding:
    """Test population seeding strategy."""
    
    def test_seed_perturbations(self):
        """Test that perturbation logic works correctly."""
        N = 4
        seed = np.random.choice([-1, 1], size=N).tolist()
        
        new_seq = seed.copy()
        n_flips = 2
        flip_idx = np.random.choice(N, n_flips, replace=False)
        for idx in flip_idx:
            new_seq[idx] *= -1
        
        diff_count = sum(1 for a, b in zip(seed, new_seq) if a != b)
        assert diff_count == n_flips
    
    def test_population_diversity(self):
        """Population should have some diversity."""
        N = 4
        pop_size = 20
        
        seed = np.random.choice([-1, 1], size=N).tolist()
        population = [seed.copy()]
        
        for _ in range(pop_size // 2):
            new_seq = seed.copy()
            n_flips = np.random.randint(1, 4)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            for idx in flip_idx:
                new_seq[idx] *= -1
            population.append(new_seq)
        
        while len(population) < pop_size:
            population.append(np.random.choice([-1, 1], size=N).tolist())
        
        unique_seqs = set(tuple(s) for s in population)
        assert len(unique_seqs) > 1


class TestHybridIntegration:
    """Integration tests for hybrid solver."""
    
    # @pytest.fixture(autouse=True)
    # def disable_cudaq(self):
    #     """Disable CUDA-Q for all tests."""
    #     import algorithms.rqaoa as rqaoa_module
    #     original = rqaoa_module.CUDAQ_AVAILABLE
    #     rqaoa_module.CUDAQ_AVAILABLE = False
    #     yield
    #     rqaoa_module.CUDAQ_AVAILABLE = original
    
    def test_stress_multiple_runs(self):
        """Solver should work reliably across multiple runs."""
        from algorithms.hybrid import solve_hybrid
        
        N = 10
        for _ in range(3):
            result = solve_hybrid(
                N=N,
                depth=1,
                shots=50,
                mts_generations=3,
                mts_population_size=5,
                verbose=False
            )
            
            assert len(result["final_sequence"]) == N
            assert result["final_energy"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
