# tests/test_benchmark_suite.py
"""Tests for the benchmark suite."""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBestKnownValues:
    """Test best known values reference."""
    
    def test_best_known_energies_exist(self):
        from benchmarks.benchmark_suite import BEST_KNOWN_ENERGIES
        
        assert len(BEST_KNOWN_ENERGIES) >= 40
        assert BEST_KNOWN_ENERGIES[10] == 13
        assert BEST_KNOWN_ENERGIES[20] == 26
    
    def test_get_best_known_mf(self):
        from benchmarks.benchmark_suite import get_best_known_mf
        
        mf_10 = get_best_known_mf(10)
        assert mf_10 == (10 ** 2) / (2 * 13)
        
        # Unknown N should return None
        assert get_best_known_mf(1000) is None


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""
    
    def test_mf_ratio_computed(self):
        from benchmarks.benchmark_suite import BenchmarkResult
        
        result = BenchmarkResult(
            method="test",
            N=10,
            energy=15,
            merit_factor=3.33,
            time_seconds=1.0,
            sequence=[1] * 10,
        )
        
        assert result.best_known_mf is not None
        assert result.mf_ratio is not None
        assert result.mf_ratio < 1.0  # Our result is worse than optimal


class TestCorrelationSnapshot:
    """Test CorrelationSnapshot functionality."""
    
    def test_to_matrix(self):
        from benchmarks.benchmark_suite import CorrelationSnapshot
        
        snapshot = CorrelationSnapshot(
            step=0,
            n_qubits=3,
            correlations_2body={(0, 1): 0.8, (1, 2): -0.5, (0, 2): 0.3},
            correlations_4body={},
        )
        
        matrix = snapshot.to_matrix()
        
        assert matrix.shape == (3, 3)
        assert matrix[0, 1] == 0.8
        assert matrix[1, 0] == 0.8  # Symmetric
        assert matrix[1, 2] == -0.5
        assert np.diag(matrix).tolist() == [1.0, 1.0, 1.0]


class TestLABSBenchmarkSuite:
    """Test benchmark suite runner."""
    
    @pytest.fixture(autouse=True)
    def disable_cudaq(self):
        """Disable CUDA-Q for tests."""
        import algorithms.rqaoa as rqaoa_module
        import algorithms.naive_qaoa as naive_qaoa_module
        
        orig_rqaoa = rqaoa_module.CUDAQ_AVAILABLE
        orig_naive = naive_qaoa_module.CUDAQ_AVAILABLE
        
        rqaoa_module.CUDAQ_AVAILABLE = False
        naive_qaoa_module.CUDAQ_AVAILABLE = False
        
        yield
        
        rqaoa_module.CUDAQ_AVAILABLE = orig_rqaoa
        naive_qaoa_module.CUDAQ_AVAILABLE = orig_naive
    
    def test_run_classical_mts_cpu(self):
        from benchmarks.benchmark_suite import LABSBenchmarkSuite
        
        suite = LABSBenchmarkSuite(verbose=False)
        result = suite.run_classical_mts_cpu(N=4, generations=10, population_size=5)
        
        assert result.method == "MTS_CPU"
        assert result.N == 4
        assert len(result.sequence) == 4
        assert result.energy > 0
        assert result.time_seconds > 0
    
    def test_run_classical_mts_gpu(self):
        from benchmarks.benchmark_suite import LABSBenchmarkSuite
        
        suite = LABSBenchmarkSuite(verbose=False)
        result = suite.run_classical_mts_gpu(N=4, generations=10, population_size=5)
        
        assert result.method == "MTS_GPU"
        assert result.N == 4
        assert len(result.sequence) == 4
    
    def test_run_naive_qaoa_mts(self):
        from benchmarks.benchmark_suite import LABSBenchmarkSuite
        
        suite = LABSBenchmarkSuite(verbose=False)
        result = suite.run_naive_qaoa_mts(
            N=4, qaoa_depth=1, qaoa_shots=50,
            mts_generations=5, mts_population_size=5
        )
        
        assert result.method == "NaiveQAOA_MTS"
        assert "quantum_energy" in result.extra_data
        assert "quantum_time" in result.extra_data
    
    def test_run_rqaoa_mts(self):
        from benchmarks.benchmark_suite import LABSBenchmarkSuite
        
        suite = LABSBenchmarkSuite(verbose=False)
        result = suite.run_rqaoa_mts(
            N=4, rqaoa_depth=1, rqaoa_shots=50,
            mts_generations=5, mts_population_size=5
        )
        
        assert result.method == "RQAOA_MTS"
        assert len(result.sequence) == 4
    
    def test_run_rqaoa_mts_with_trajectory(self):
        from benchmarks.benchmark_suite import LABSBenchmarkSuite
        
        suite = LABSBenchmarkSuite(verbose=False)
        result = suite.run_rqaoa_mts(
            N=4, rqaoa_depth=1, rqaoa_shots=50,
            mts_generations=5, mts_population_size=5,
            record_trajectory=True,
            record_correlations=True,
        )
        
        assert len(suite.trajectories) > 0
        assert len(suite.correlation_snapshots) > 0
    
    def test_measure_gpu_speedup(self):
        from benchmarks.benchmark_suite import LABSBenchmarkSuite
        
        suite = LABSBenchmarkSuite(verbose=False)
        results = suite.measure_gpu_speedup(
            N_values=[4, 6], 
            n_iterations=3,
            population_size=20, 
            generations=10
        )
        
        assert len(results) == 2
        for r in results:
            assert r.cpu_time_ms > 0
            assert r.gpu_time_ms > 0
            assert r.speedup > 0


class TestMetricsComputation:
    """Test metrics computation."""
    
    def test_metric1_approximation(self):
        from benchmarks.benchmark_suite import LABSBenchmarkSuite, BenchmarkResult
        
        suite = LABSBenchmarkSuite(verbose=False)
        
        # Create mock results
        suite.results = [
            BenchmarkResult(
                method="RQAOA_MTS", N=10, energy=14,
                merit_factor=3.57, time_seconds=1.0, sequence=[1]*10
            ),
        ]
        
        # The ratio should be computed
        assert suite.results[0].mf_ratio is not None


class TestVisualizationHelpers:
    """Test visualization utilities without actual plotting."""
    
    def test_setup_plotting_style(self):
        from benchmarks.visualization import setup_plotting_style
        
        # Should not raise
        setup_plotting_style()
    
    def test_generate_summary_report(self):
        from benchmarks.visualization import generate_summary_report
        
        mock_results = {
            "metrics": {
                "metric1_approximation": {
                    "RQAOA_MTS": [{"N": 10, "ratio": 0.95, "passes": True}]
                },
                "metric2_superiority": {
                    25: [{"rqaoa_mf": 5.0, "naive_mf": 4.5, "passes": True}]
                },
                "metric3_speedup": {
                    "avg_speedup": 25.0,
                    "max_speedup": 30.0,
                    "passes": True,
                    "details": [(10, 20.0), (20, 30.0)],
                },
                "metric4_scale": {
                    "N": 50,
                    "success": True,
                    "energy": 329,
                    "time": 100.0,
                },
            },
            "total_time_seconds": 3600.0,
        }
        
        report = generate_summary_report(mock_results)
        
        assert "METRIC 1" in report
        assert "METRIC 2" in report
        assert "METRIC 3" in report
        assert "METRIC 4" in report
        assert "RQAOA_MTS" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
