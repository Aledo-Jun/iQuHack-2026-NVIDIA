# tests/test_mts.py
# Tests for Memetic Tabu Search with GPU acceleration

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.labs_utils import calculate_energy
from solvers.mts import (
    run_mts,
    tabu_search,
    tabu_search_optimized,
    combine,
    mutate,
    tournament_selection,
    GPUAcceleratedMTS,
)


class TestGeneticOperators:
    """Test genetic algorithm operators."""
    
    def test_combine_crossover(self):
        """Test single-point crossover."""
        p1 = [1, 1, 1, 1, 1]
        p2 = [-1, -1, -1, -1, -1]
        
        np.random.seed(42)
        child = combine(p1, p2)
        
        assert len(child) == 5
        assert all(s in [-1, 1] for s in child)
        # Child should be a mix (unless crossover point is at 0 or N)
    
    def test_combine_preserves_length(self):
        """Crossover should preserve sequence length."""
        for N in [5, 10, 20]:
            p1 = np.random.choice([-1, 1], size=N).tolist()
            p2 = np.random.choice([-1, 1], size=N).tolist()
            child = combine(p1, p2)
            assert len(child) == N
    
    def test_mutate_rate_zero(self):
        """No mutation with rate 0."""
        seq = [1, 1, 1, 1, 1]
        mutated = mutate(seq, mutation_rate=0.0)
        assert mutated == seq
    
    def test_mutate_rate_one(self):
        """All bits flip with rate 1."""
        seq = [1, 1, 1, 1, 1]
        mutated = mutate(seq, mutation_rate=1.0)
        assert mutated == [-1, -1, -1, -1, -1]
    
    def test_mutate_preserves_values(self):
        """Mutation should only produce -1 or 1."""
        seq = np.random.choice([-1, 1], size=20).tolist()
        mutated = mutate(seq, mutation_rate=0.5)
        assert all(s in [-1, 1] for s in mutated)
    
    def test_tournament_selection(self):
        """Tournament selection should return valid individual."""
        population = [
            [1, 1, 1],
            [-1, -1, -1],
            [1, -1, 1],
        ]
        energies = [5, 3, 2]  # Third is best
        
        # With k=3 (all), should always select best
        selected = tournament_selection(population, energies, k=3)
        assert selected == [1, -1, 1]


class TestTabuSearch:
    """Test Tabu Search algorithm."""
    
    def test_tabu_search_returns_valid(self):
        """Tabu search should return valid sequence."""
        seq = np.random.choice([-1, 1], size=10).tolist()
        result_seq, result_energy = tabu_search(seq, max_iter=50)
        
        assert len(result_seq) == 10
        assert all(s in [-1, 1] for s in result_seq)
        assert result_energy == calculate_energy(result_seq)
    
    def test_tabu_search_non_worsening(self):
        """Tabu search should not make solution worse."""
        np.random.seed(42)
        seq = np.random.choice([-1, 1], size=12).tolist()
        initial_energy = calculate_energy(seq)
        
        result_seq, result_energy = tabu_search(seq, max_iter=100)
        
        assert result_energy <= initial_energy
    
    def test_tabu_search_different_starts(self):
        """Tabu search should work with different starting points."""
        for _ in range(5):
            seq = np.random.choice([-1, 1], size=8).tolist()
            result_seq, result_energy = tabu_search(seq, max_iter=30)
            assert result_energy == calculate_energy(result_seq)


class TestTabuSearchOptimized:
    """Test optimized tabu search."""
    
    def test_optimized_returns_valid(self):
        """Optimized tabu search returns valid solution."""
        seq = np.random.choice([-1, 1], size=8).tolist()
        
        result_seq, result_energy = tabu_search_optimized(seq)
        
        assert len(result_seq) == 8
        assert all(s in [-1, 1] for s in result_seq)
        assert result_energy == calculate_energy(result_seq)
    
    def test_optimized_with_gpu_solver(self):
        """Test with explicit GPU solver instance."""
        N = 10
        
        # Mock GPU solver
        mock_solver = MagicMock()
        # Mock run_tabu_search_batch return value (list of lists, list of energies)
        mock_solver.run_tabu_search_batch.return_value = (
            [[1] * N], # best_seqs
            [0]        # best_energies
        )
        
        seq = np.random.choice([-1, 1], size=N).tolist()
        result_seq, result_energy = tabu_search_optimized(seq, mock_solver)
        
        # Should have called batch method
        mock_solver.run_tabu_search_batch.assert_called_once()
        args, kwargs = mock_solver.run_tabu_search_batch.call_args
        # First arg should be list containing seq
        assert args[0] == [seq]
        
        assert result_seq == [1] * N
        assert result_energy == 0
    
    def test_optimized_non_worsening(self):
        """Optimized search should not worsen solution."""
        np.random.seed(123)
        seq = np.random.choice([-1, 1], size=12).tolist()
        initial_energy = calculate_energy(seq)
        
        result_seq, result_energy = tabu_search_optimized(seq)
        
        assert result_energy <= initial_energy


class TestMTSRunner:
    """Test full MTS runner."""
    
    def test_mts_returns_valid(self):
        """MTS should return valid solution and history."""
        N = 6
        seq, energy, history = run_mts(
            N, population_size=10, generations=5, verbose=False
        )
        
        assert len(seq) == N
        assert energy == calculate_energy(seq)
        assert len(history) == 6  # Initial + 5 generations
    
    def test_mts_history_non_increasing(self):
        """Best energy in history should never increase."""
        N = 8
        _, _, history = run_mts(
            N, population_size=15, generations=10, verbose=False
        )
        
        for i in range(1, len(history)):
            assert history[i] <= history[i-1]
    
    def test_mts_with_initial_population(self):
        """MTS should accept initial population."""
        N = 6
        initial_pop = [
            np.random.choice([-1, 1], size=N).tolist()
            for _ in range(10)
        ]
        
        seq, energy, history = run_mts(
            N,
            population_size=10,
            generations=5,
            initial_population=initial_pop,
            verbose=False
        )
        
        assert len(seq) == N
        assert energy == calculate_energy(seq)

    def test_mts_runs_with_batch_size(self):
        """MTS should run correctly with batch_size > 1."""
        N = 5
        batch_size = 5
        generations = 2
        
        # We can mock tabu_search to count calls
        with patch('solvers.mts.tabu_search') as mock_tabu:
            # Mock return
            mock_tabu.return_value = ([1]*N, 0)
            
            run_mts(
                N, 
                population_size=10, 
                generations=generations, 
                batch_size=batch_size, 
                verbose=False
            )
            
            # Expected calls: generations * batch_size
            assert mock_tabu.call_count == generations * batch_size
    
    def test_mts_improves_random_start(self):
        """MTS should generally improve from random start."""
        np.random.seed(42)
        N = 10
        
        # Generate initial random population and find worst
        initial_pop = [
            np.random.choice([-1, 1], size=N).tolist()
            for _ in range(20)
        ]
        initial_best = min(calculate_energy(s) for s in initial_pop)
        
        seq, energy, _ = run_mts(
            N,
            population_size=20,
            generations=20,
            initial_population=initial_pop,
            verbose=False
        )
        
        # Should be at least as good as initial best
        assert energy <= initial_best


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
