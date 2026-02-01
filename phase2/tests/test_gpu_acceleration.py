# tests/test_gpu_acceleration.py
# Tests specifically for GPU acceleration functionality

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.labs_utils import calculate_energy
from solvers.mts import GPUAcceleratedMTS

class TestCuPyMocking:
    """Test GPU functions with mocked CuPy."""
    
    def test_mock_cupy_available(self):
        """Test behavior when CuPy is available."""
        mock_cp = MagicMock()
        mock_cp.asarray = MagicMock(side_effect=lambda x, dtype=None: np.array(x, dtype=dtype))
        mock_cp.zeros = MagicMock(side_effect=lambda shape, dtype=None: np.zeros(shape, dtype=dtype))
        mock_cp.asnumpy = MagicMock(side_effect=lambda x: np.array(x))
        mock_cp.sum = np.sum
        mock_cp.eye = np.eye
        mock_cp.arange = np.arange
        mock_cp.newaxis = np.newaxis
        
        with patch.dict('sys.modules', {'cupy': mock_cp}):
            import cupy as cp
            assert cp.asarray([1, 2, 3], dtype=np.float32) is not None

class TestGPUAcceleratedMTS:
    """Test GPU-accelerated MTS components with mocks."""
    
    def setup_method(self):
        """Setup mock environment."""
        self.mock_cp = MagicMock()
        # Basic array operations
        self.mock_cp.asarray = lambda x, dtype=None: np.array(x, dtype=dtype)
        self.mock_cp.zeros = lambda shape, dtype=None: np.zeros(shape, dtype=dtype)
        # Handle astype calls on mocked arrays if they return normal numpy arrays?
        # Actually our mock returns numpy arrays, which have .astype
        
        self.mock_cp.sum = np.sum
        self.mock_cp.eye = np.eye
        self.mock_cp.arange = np.arange
        self.mock_cp.newaxis = np.newaxis
        self.mock_cp.float32 = np.float32
        self.mock_cp.int8 = np.int8
        self.mock_cp.int32 = np.int32
        self.mock_cp.inf = float('inf')
        self.mock_cp.where = np.where
        self.mock_cp.argmin = np.argmin
        self.mock_cp.min = np.min
        self.mock_cp.random.randint = np.random.randint
        self.mock_cp.any = np.any
        
    def test_init_requires_cupy(self):
        """Should raise ImportError if CuPy is missing."""
        with patch.dict('sys.modules', {'cupy': None}):
            # We also need to reload or patch the module level 'cp' in mts
            with patch('solvers.mts.cp', None):
                with pytest.raises(ImportError):
                    GPUAcceleratedMTS(N=10)

    def test_compute_energies_correctness(self):
        """Test vectorized energy calculation matches CPU."""
        with patch.dict('sys.modules', {'cupy': self.mock_cp}):
            with patch('solvers.mts.cp', self.mock_cp):
                N = 6
                solver = GPUAcceleratedMTS(N)
                
                # Test batch of 2
                seq1 = [1, 1, -1, 1, -1, 1]
                seq2 = [1, 1, 1, 1, 1, 1]
                batch = np.array([seq1, seq2], dtype=np.int8)
                
                energies = solver.compute_energies(batch)
                
                assert len(energies) == 2
                assert int(energies[0]) == calculate_energy(seq1)
                assert int(energies[1]) == calculate_energy(seq2)

    def test_run_tabu_search_batch_logic(self):
        """Test the batch tabu search logic."""
        with patch.dict('sys.modules', {'cupy': self.mock_cp}):
            with patch('solvers.mts.cp', self.mock_cp):
                N = 5
                solver = GPUAcceleratedMTS(N)
                
                # Mock flip mask since our mock eye works
                # Mock everything else required for the loop
                
                initial_seq = [1, -1, 1, -1, 1]
                batch = [initial_seq]
                
                # Run for small iter
                final_seqs, final_energies = solver.run_tabu_search_batch(batch, n_iter=5)
                
                assert len(final_seqs) == 1
                assert len(final_energies) == 1
                assert len(final_seqs[0]) == N
                
                # Result should be at least as good as initial
                initial_energy = calculate_energy(initial_seq)
                assert final_energies[0] <= initial_energy

    def test_batch_scaling(self):
        """Test functioning with larger batch."""
        with patch.dict('sys.modules', {'cupy': self.mock_cp}):
            with patch('solvers.mts.cp', self.mock_cp):
                N = 8
                solver = GPUAcceleratedMTS(N)
                
                batch_size = 10
                initial_seqs = [
                    np.random.choice([-1, 1], size=N).tolist()
                    for _ in range(batch_size)
                ]
                
                final_seqs, final_energies = solver.run_tabu_search_batch(initial_seqs, n_iter=2)
                
                assert len(final_seqs) == batch_size
                assert len(final_energies) == batch_size

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
