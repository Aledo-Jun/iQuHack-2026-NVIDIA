# Recursive QAOA for LABS (Phase 2)

**iQuHack 2026 NVIDIA Challenge**
**Team:** June

## Overview

This project implements a **Recursive Quantum Approximate Optimization Algorithm (RQAOA)** hybridized with a **GPU-Accelerated Memetic Tabu Search (MTS)** to solve the **Low Autocorrelation of Binary Sequences (LABS)** problem.

The solution leverages **NVIDIA's CUDA-Q** for dynamic quantum circuit simulation and **CuPy** for massively parallel classical neighborhood exploration on the GPU.

### Key Features

*   **Recursive QAOA (RQAOA)**: Iterative decimation of variables based on $\langle Z_i Z_j \rangle$ correlations to handle "frustrated" energy landscapes.
*   **CUDA-Q Acceleration**: Uses `nvidia-mgpu` backend with **JIT compilation** to handle dynamic ansatz generation.
*   **GPU-MTS**: A custom Memetic Tabu Search written in **CuPy** that evaluates thousands of neighbor flips in a single kernel call, achieving >20x speedup over CPU baselines.
*   **Hybrid Architecture**: Seamless integration of quantum seeding and massive classical local search.

## Prerequisites

*   **Python**: >= 3.10
*   **Package Manager**: `uv` (Recommended)
*   **Hardware**: NVIDIA GPU (L4, A100, or similar) with CUDA 12.x drivers.

## Installation

This project uses `uv` for dependency management.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Aledo-Jun/iQuHack-2026-NVIDIA.git
    cd iQuHack-2026-NVIDIA/NVIDIA/phase2
    ```

2.  **Install Dependencies**:
    Sync the environment including GPU dependencies.
    ```bash
    uv sync --extra gpu
    ```
    *This will install `cudaq`, `cupy-cuda12x`, `numpy`, `matplotlib`, and testing tools.*

## Usage

### 1. Running Tests
Verify the installation and core logic (Quantum Environemnt, MTS, RQAOA integrity).
```bash
uv run pytest
```

### 2. Running Benchmarks
Execute the full hybrid benchmark suite to measure Time to Solution (TTS) and Approximation Ratio.
```bash
uv run python benchmarks/benchmark_suite.py
```

### 3. Training / Experiments
To run specific RQAOA experiments:
```bash
uv run python src/algorithms/rqaoa.py
```

## Project Structure

```
phase2/
├── benchmarks/          # Reproducible metrics scripts
├── src/
│   ├── algorithms/     # RQAOA and QAOA implementations
│   ├── solvers/        # GPU-Accelerated MTS (Memetic Tabu Search)
│   ├── utils/          # Helper functions and noise models
│   └── hybrid.py       # Main interaction loop
├── tests/              # Unit and property-based tests
├── presentation.md     # Phase 2 Presentation
├── pyproject.toml      # Dependency configuration
└── README.md           # This file
```

## License
MIT
