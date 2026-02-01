# Product Requirements Document (PRD)

**Project Name:** Recursive QAOA for LABS
**Team Name:** June
**GitHub Repository:** [GitHub](https://github.com/Aledo-Jun/iQuHack-2026-NVIDIA)

---

## 1. Team Roles & Responsibilities [Solely me]
Since I am the only member of my team, I will be responsible for all roles and AI Agent(Antigravity) will be my assistant.

---

## 2. The Architecture

### Choice of Quantum Algorithm
* **Algorithm:** **Recursive Quantum Approximate Optimization Algorithm (RQAOA)**.
* **Motivation:** 
    * **Metric-driven (Performance):** The LABS problem features a highly "frustrated" energy landscape (like a spin glass) where standard local optimizers and even standard QAOA often get trapped in local minima. RQAOA overcomes this by not trying to find the solution in one shot. Instead, it iteratively calculates two-body correlations ($\langle Z_i Z_j \rangle$) using a shallow QAOA circuit, "fixes" the most strongly correlated variable (spin decimation), and reduces the problem size by 1. This "divide-and-conquer" approach has been proven to yield higher approximation ratios on dense, frustrated graphs compared to standard QAOA.
    * **Skills-driven (Advanced Hybrid Logic):** Implementing RQAOA requires a complex, dynamic hybrid loop where the *Hamiltonian itself* changes at every step. This demonstrates mastery of CUDA-Q's ability to re-compile or parameterize circuits on the fly, a significant step up from static circuit execution.

### Literature Review
* **Reference:** [Obstacles to State Preparation and Variational Optimization from Symmetry Protection (Original RQAOA)](https://arxiv.org/abs/1910.08980)
* **Relevance:** Introduces the recursive decimation strategy to overcome the "locality" limitations of standard QAOA, directly applicable to the all-to-all connectivity of the LABS Hamiltonian.
* **Reference:** [The binary sequence problem with quantum algorithms](https://arxiv.org/abs/2403.04706)
* **Relevance:** Benchmarks quantum approaches for LABS, highlighting the need for advanced ansatzes or strategies beyond vanilla VQE/QAOA to beat classical solvers.

---

## 3. The Acceleration Strategy

### Quantum Acceleration (CUDA-Q)
* **Strategy:**
    * **Dynamic Graph Reduction:** RQAOA requires $N$ iterations of QAOA. We will use CUDA-Q's `nvidia-mgpu` backend to massively parallelize the expectation value calculations ($\langle Z_i Z_j \rangle$) for each iteration.
    * **JIT Compilation:** Because the circuit structure changes (qubits are removed) and the Hamiltonian changes (couplings are updated) at every recursive step, we will leverage CUDA-Q's Just-In-Time (JIT) compilation to handle these dynamic circuit generations efficiently without efficient python-loop overhead.

### Classical Acceleration (MTS)
* **Strategy:**
    * **Parallel Neighborhood Evaluation:** The bottleneck in Memetic Tabu Search is evaluating the energy of all neighbors (single bit flips). We will use `cupy` to re-implement the energy function (Merit Factor calculation) to evaluate the entire neighborhood (batch size $N$) in a single GPU kernel call, rather than iterating sequentially on the CPU.
    * **Data Transfer Minimization:** We will keep the population on the GPU memory as much as possible to avoid costly Host-to-Device transfers.

### Hardware Targets
* **Dev Environment:** Qbraid (CPU) for initial logic implementation and unit testing.
* **Production Environment:** Brev L4 for intermediate scaling (N=20-30) and debugging GPU kernels. Brev A100-80GB (single or multi-GPU) for final benchmarking at large N (N=40-50).

---

## 4. The Verification Plan

### Unit Testing Strategy
* **Framework:** `pytest` for all unit tests.
* **AI Hallucination Guardrails:**
    * **Property-Based Testing:** We will use `hypothesis` to generate random binary sequences and verify that our accelerated energy functions match the reference CPU implementation exactly.
    * **Physical Validity Checks:** Assertions will verify that valid quantum states have norm 1.0 (within tolerance) and probabilities sum to 1.0.

## 4. The Verification Plan

### Unit Testing Strategy
* **Framework:** `pytest` for all unit tests.
* **AI Hallucination Guardrails:**
    * **Property-Based Testing:** Use `hypothesis` to generate random graphs and verify:
        *   **Correlation Bounds:** $|\langle Z_i Z_j \rangle| \le 1.0 + \epsilon$ for all pairs.
        *   **Energy Conservation:** Energy of the system should remain bounded during recursion.
    * **Recursive Logic Check:** For a small $N=4$ system, manually perform one step of decimation (fix $Z_1 = Z_2$) and verify the reduced $N=3$ Hamiltonian matches manual derivation.

### Core Correctness Checks
* **Check 1 (Symmetry - Reflection & Negation):** As described previously, $Energy(S) == Energy(S_{reversed}) == Energy(-S)$.
* **Check 2 (Ground Truth - Small N):** Verify exact solutions for $N \le 6$ against brute-force baselines.
* **Check 3 (Recursive Step Integrity):**
    *   **Action:** Before and after every variable elimination (decimation), calculating the energy of a probe state should yield consistent results (accounting for the constant energy shift $C$).
    *   **Assert:** `Energy_Full(State_Full) == Energy_Reduced(State_Reduced) + Constant_Shift`.
* **Check 4 (Zero-Correlation Alert):** If RQAOA calculates $\langle Z_i Z_j \rangle \approx 0$ for all pairs, the algorithm is failing to capture structure (likely depth $p$ is too low or noise is too high). Raises a warning.

---

## 5. Execution Strategy & Success Metrics

### Agentic Workflow
* **Plan:**
    * **Architect (Agent 1):** Designs the **Recursive Decimation Logic** and the base QAOA circuit.
    * **Engineer (Agent 2):** Implements the "Outer Loop" (Classical Recursion) in Python/CuPy and the "Inner Loop" (Quantum Expectation Values) in CUDA-Q.
    * **Verifier (Agent 3):** Writes tests to verify that the "Reduced Hamiltonian" energy matches the "Full Hamiltonian" energy for fixed variables (Consistency Check).

### Success Metrics
* **Metric 1 (Approximation):** Achieve a Merit Factor (MF) within 90% of the best known solutions for $N \le 40$.
* **Metric 2 (Superiority):** RQAOA must achieve a higher Merit Factor than a standard depth-$p=1$ QAOA baseline for $N > 20$.
* **Metric 3 (Speedup):** Achieve at least a **20x speedup** in neighborhood evaluation (MTS) on the A100 GPU compared to the single-threaded CPU numpy baseline.
* **Metric 4 (Scale):** Successfully run the full Hybrid pipeline (Quantum Seed + Accelerated MTS) for $N=50$ within the 2-hour benchmark window.

### Visualization Plan
* **Plot 1:** **RQAOA vs. Standard QAOA vs. Classical BFS:**
    *   **X-axis:** Problem Size ($N$).
    *   **Y-axis:** Final Energy / Merit Factor.
    *   **Goal:** Demonstrate that RQAOA avoids the "plateau" that standard QAOA hits for frustrated systems at $N > 20$.
* **Plot 2:** **Optimization Trajectory:** Energy vs. Recursion Step. Show the energy lowering as variables are decimated.
* **Plot 3:** **Correlation Heatmap:** Visualize the $\langle Z_i Z_j \rangle$ matrix at Step 1 to show which qubits are being fixed first (interpretable AI).

---

## 6. Resource Management Plan

* **Plan:**
    * **Development (Phase 1):** All initial coding, unit testing, and small-scale validation ($N \le 15$) will be done on the **Qbraid CPU environment** (Free).
    * **Testing & Porting (Phase 2):** We will spin up a **Brev L4 instance** ($0.85/hr approx) to verify CUDA-Q installation and debug the CuPy kernels. We expect to use ~3-4 hours here.
    * **Production Benchmarking (Phase 3):** We will reserve the **A100-80GB instance** only for the final data collection runs for the presentation. We estimate a maximum of 2-3 hours of runtime.
    * **Zombie Killer Protocol:** The GPU PIC will assume the role of "Instance Watchdog", manually verifying that instances are halted via the Brev CLI/Console immediately after scripts complete. Automated timeout scripts will be investigated if manual management proves unreliable.
