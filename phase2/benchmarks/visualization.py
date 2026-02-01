# src/benchmarks/visualizations.py
"""
Visualization utilities for LABS benchmark results.
"""
import sys
from pathlib import Path

import numpy as np
from typing import List, Dict, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from benchmarks.benchmark_suite import (
    BenchmarkResult,
    TrajectoryPoint,
    CorrelationSnapshot,
    get_best_known_mf,
    BEST_KNOWN_ENERGIES,
)


def setup_plotting_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


# ============================================================
# Plot 1: Method Comparison (MF vs N)
# ============================================================
def plot_method_comparison(
    results: List[BenchmarkResult],
    title: str = "LABS Solver Comparison: Merit Factor vs Problem Size",
    save_path: Optional[str] = None,
    show_best_known: bool = True,
):
    """
    Plot Merit Factor vs N for different methods.
    
    Args:
        results: List of BenchmarkResult objects
        title: Plot title
        save_path: If provided, save figure to this path
        show_best_known: Whether to show best known MF line
    """
    setup_plotting_style()
    
    # Group results by method
    methods = {}
    for r in results:
        if r.method not in methods:
            methods[r.method] = {"N": [], "mf": [], "energy": []}
        methods[r.method]["N"].append(r.N)
        methods[r.method]["mf"].append(r.merit_factor)
        methods[r.method]["energy"].append(r.energy)
    
    # Color scheme
    colors = {
        "MTS_CPU": "gray",
        "MTS_GPU": "blue",
        "NaiveQAOA_MTS": "orange",
        "RQAOA_MTS": "green",
    }
    markers = {
        "MTS_CPU": "s",
        "MTS_GPU": "^",
        "NaiveQAOA_MTS": "o",
        "RQAOA_MTS": "D",
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Merit Factor
    for method, data in methods.items():
        # Aggregate by N (mean ± std)
        N_unique = sorted(set(data["N"]))
        mf_means = []
        mf_stds = []
        for n in N_unique:
            mf_vals = [mf for N, mf in zip(data["N"], data["mf"]) if N == n]
            mf_means.append(np.mean(mf_vals))
            mf_stds.append(np.std(mf_vals))
        
        color = colors.get(method, "black")
        marker = markers.get(method, "x")
        
        ax1.errorbar(
            N_unique, mf_means, yerr=mf_stds,
            label=method, color=color, marker=marker,
            capsize=3, linewidth=2, markersize=8,
        )
    
    # Best known MF line
    if show_best_known:
        N_range = sorted(set(r.N for r in results))
        best_mf = [get_best_known_mf(n) for n in N_range]
        best_mf = [mf if mf else np.nan for mf in best_mf]
        ax1.plot(N_range, best_mf, 'k--', label="Best Known", linewidth=2, alpha=0.7)
    
    ax1.set_xlabel("Problem Size (N)")
    ax1.set_ylabel("Merit Factor")
    ax1.set_title("Merit Factor Comparison")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Energy
    for method, data in methods.items():
        N_unique = sorted(set(data["N"]))
        e_means = []
        e_stds = []
        for n in N_unique:
            e_vals = [e for N, e in zip(data["N"], data["energy"]) if N == n]
            e_means.append(np.mean(e_vals))
            e_stds.append(np.std(e_vals))
        
        color = colors.get(method, "black")
        marker = markers.get(method, "x")
        
        ax2.errorbar(
            N_unique, e_means, yerr=e_stds,
            label=method, color=color, marker=marker,
            capsize=3, linewidth=2, markersize=8,
        )
    
    # Best known energy line
    if show_best_known:
        N_range = sorted(set(r.N for r in results))
        best_e = [BEST_KNOWN_ENERGIES.get(n, np.nan) for n in N_range]
        ax2.plot(N_range, best_e, 'k--', label="Best Known", linewidth=2, alpha=0.7)
    
    ax2.set_xlabel("Problem Size (N)")
    ax2.set_ylabel("Energy")
    ax2.set_title("Energy Comparison")
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig


# ============================================================
# Plot 2: Optimization Trajectory
# ============================================================
def plot_optimization_trajectory(
    trajectories: Dict[str, List[TrajectoryPoint]],
    title: str = "RQAOA Optimization Trajectory",
    save_path: Optional[str] = None,
):
    """
    Plot energy/merit factor vs decimation step.
    
    Args:
        trajectories: Dict mapping run name to list of TrajectoryPoints
        title: Plot title
        save_path: If provided, save figure to this path
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(trajectories)))
    
    for idx, (name, points) in enumerate(trajectories.items()):
        steps = [p.step for p in points]
        n_vars = [p.n_active_vars for p in points]
        correlations = [p.correlation if p.correlation else 0 for p in points]
        
        color = colors[idx]
        
        # Plot 1: Active variables vs step
        axes[0].plot(steps, n_vars, 'o-', color=color, label=name, linewidth=2, markersize=6)
        
        # Plot 2: Correlation strength at each decimation
        valid_corrs = [(s, c) for s, c in zip(steps, correlations) if c != 0]
        if valid_corrs:
            corr_steps, corr_vals = zip(*valid_corrs)
            axes[1].bar(
                [s + idx * 0.2 for s in corr_steps],
                [abs(c) for c in corr_vals],
                width=0.2, color=color, label=name, alpha=0.7
            )
        
        # Plot 3: Decimation pairs (text annotations)
        for p in points:
            if p.decimated_pair:
                axes[2].annotate(
                    f"Z{p.decimated_pair[0]}↔Z{p.decimated_pair[1]}",
                    xy=(p.step, idx),
                    fontsize=9,
                )
    
    axes[0].set_xlabel("Decimation Step")
    axes[0].set_ylabel("Active Variables")
    axes[0].set_title("Variable Reduction")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Decimation Step")
    axes[1].set_ylabel("|Correlation|")
    axes[1].set_title("Decimation Correlation Strength")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel("Decimation Step")
    axes[2].set_ylabel("Run")
    axes[2].set_title("Decimated Variable Pairs")
    axes[2].set_yticks(range(len(trajectories)))
    axes[2].set_yticklabels(list(trajectories.keys()))
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig


# ============================================================
# Plot 3: Correlation Heatmap
# ============================================================
def plot_correlation_heatmap(
    snapshot: CorrelationSnapshot,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_values: bool = True,
):
    """
    Plot correlation matrix heatmap.
    
    Args:
        snapshot: CorrelationSnapshot object
        title: Plot title (auto-generated if None)
        save_path: If provided, save figure to this path
        show_values: Whether to show correlation values in cells
    """
    setup_plotting_style()
    
    matrix = snapshot.to_matrix()
    n = snapshot.n_qubits
    
    if title is None:
        title = f"⟨ZᵢZⱼ⟩ Correlation Matrix (Step {snapshot.step}, {n} qubits)"
    
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(6, n * 0.4)))
    
    # Custom colormap: blue (negative) -> white (zero) -> red (positive)
    cmap = plt.cm.RdBu_r
    
    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("⟨ZᵢZⱼ⟩ Correlation")
    
    # Add text annotations
    if show_values and n <= 15:
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if abs(val) > 0.01:
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           color=color, fontsize=8)
    
    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Z{i}" for i in range(n)])
    ax.set_yticklabels([f"Z{i}" for i in range(n)])
    ax.set_xlabel("Qubit j")
    ax.set_ylabel("Qubit i")
    ax.set_title(title)
    
    # Highlight strongest correlations
    strongest = []
    for (i, j), corr in snapshot.correlations_2body.items():
        if abs(corr) > 0.7:
            strongest.append((i, j, corr))
    
    if strongest:
        for i, j, corr in strongest:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                 fill=False, edgecolor='gold', linewidth=3)
            ax.add_patch(rect)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig


# ============================================================
# Plot 4: GPU Speedup
# ============================================================
def plot_gpu_speedup(
    speedup_results: List[Any],
    title: str = "GPU Acceleration Speedup",
    save_path: Optional[str] = None,
    target_speedup: float = 20.0,
):
    """
    Plot GPU vs CPU speedup.
    
    Args:
        speedup_results: List of SpeedupResult objects
        title: Plot title
        save_path: If provided, save figure to this path
        target_speedup: Target speedup to highlight (Metric 3)
    """
    setup_plotting_style()
    
    N_values = [r.N for r in speedup_results]
    speedups = [r.speedup for r in speedup_results]
    cpu_times = [r.cpu_time_ms for r in speedup_results]
    gpu_times = [r.gpu_time_ms for r in speedup_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Speedup factor
    bars = axes[0].bar(N_values, speedups, color='steelblue', alpha=0.8, edgecolor='black')
    axes[0].axhline(y=target_speedup, color='red', linestyle='--', linewidth=2, 
                    label=f'Target: {target_speedup}x')
    
    # Color bars that meet target
    for bar, speedup in zip(bars, speedups):
        if speedup >= target_speedup:
            bar.set_color('green')
    
    axes[0].set_xlabel("Problem Size (N)")
    axes[0].set_ylabel("Speedup Factor (CPU/GPU)")
    axes[0].set_title("GPU Speedup vs CPU Baseline")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Right: Absolute times
    width = 0.35
    x = np.arange(len(N_values))
    
    axes[1].bar(x - width/2, cpu_times, width, label='CPU', color='gray', alpha=0.8)
    axes[1].bar(x + width/2, gpu_times, width, label='GPU', color='steelblue', alpha=0.8)
    
    axes[1].set_xlabel("Problem Size (N)")
    axes[1].set_ylabel("Time per Evaluation (ms)")
    axes[1].set_title("Neighborhood Evaluation Time")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(N_values)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_yscale('log')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    
    plt.show()
    return fig


# ============================================================
# Summary Report
# ============================================================
def generate_summary_report(
    benchmark_results: Dict[str, Any],
    save_path: Optional[str] = None,
) -> str:
    """
    Generate a text summary report of benchmark results.
    
    Args:
        benchmark_results: Results from LABSBenchmarkSuite.run_full_benchmark()
        save_path: If provided, save report to this file
        
    Returns:
        Report as string
    """
    metrics = benchmark_results.get("metrics", {})
    
    lines = [
        "=" * 60,
        "LABS BENCHMARK SUMMARY REPORT",
        "=" * 60,
        "",
    ]
    
    # Metric 1: Approximation Quality
    lines.append("### METRIC 1: Approximation Quality (MF ≥ 90% of best known)")
    m1 = metrics.get("metric1_approximation", {})
    for method, results in m1.items():
        passes = sum(1 for r in results if r["passes"])
        total = len(results)
        avg_ratio = np.mean([r["ratio"] for r in results]) if results else 0
        lines.append(f"  {method}: {passes}/{total} passed, avg ratio = {avg_ratio:.3f}")
    lines.append("")
    
    # Metric 2: RQAOA Superiority
    lines.append("### METRIC 2: RQAOA > Naive QAOA for N > 20")
    m2 = metrics.get("metric2_superiority", {})
    for N, results in sorted(m2.items()):
        passes = sum(1 for r in results if r["passes"])
        total = len(results)
        lines.append(f"  N={N}: {passes}/{total} trials RQAOA superior")
    lines.append("")
    
    # Metric 3: GPU Speedup
    lines.append("### METRIC 3: GPU Speedup ≥ 20x")
    m3 = metrics.get("metric3_speedup", {})
    if m3:
        lines.append(f"  Average Speedup: {m3.get('avg_speedup', 0):.1f}x")
        lines.append(f"  Maximum Speedup: {m3.get('max_speedup', 0):.1f}x")
        lines.append(f"  PASSES: {'YES' if m3.get('passes') else 'NO'}")
        lines.append("  Details by N:")
        for N, speedup in m3.get("details", []):
            status = "✓" if speedup >= 20 else "✗"
            lines.append(f"    N={N}: {speedup:.1f}x {status}")
    lines.append("")
    
    # Metric 4: Scale Test
    lines.append("### METRIC 4: N=50 Scale Test")
    m4 = metrics.get("metric4_scale", {})
    if m4:
        lines.append(f"  N={m4.get('N')}")
        lines.append(f"  Success: {'YES' if m4.get('success') else 'NO'}")
        if m4.get('energy'):
            lines.append(f"  Energy: {m4.get('energy')}")
            mf = (50 ** 2) / (2 * m4.get('energy'))
            lines.append(f"  Merit Factor: {mf:.3f}")
        if m4.get('time'):
            lines.append(f"  Time: {m4.get('time'):.1f}s")
    lines.append("")
    
    # Overall
    lines.append("=" * 60)
    lines.append(f"Total Benchmark Time: {benchmark_results.get('total_time_seconds', 0)/60:.1f} minutes")
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {save_path}")
    
    return report
