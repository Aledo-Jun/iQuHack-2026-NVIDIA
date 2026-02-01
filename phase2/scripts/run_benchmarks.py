# scripts/run_benchmarks.py
"""
Main benchmark runner script.
"""

import sys
from pathlib import Path
from time import time
import argparse

from benchmarks.benchmark_suite import LABSBenchmarkSuite
from benchmarks.visualization import (
    plot_method_comparison,
    plot_optimization_trajectory,
    plot_correlation_heatmap,
    plot_gpu_speedup,
    generate_summary_report,
)


def main():
    parser = argparse.ArgumentParser(description="LABS Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (smaller N)")
    parser.add_argument("--full", action="store_true", help="Full benchmark (all N)")
    parser.add_argument("--speedup-only", action="store_true", help="Only measure GPU speedup")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--timeout", type=float, default=2.0, help="Timeout in hours")
    parser.add_argument("--n_trials", type=int, default=3, help="The number of trials to be averaged")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    suite = LABSBenchmarkSuite(verbose=True)
    
    if args.speedup_only:
        print("=== GPU Speedup Measurement Only ===")
        speedup_results = suite.measure_gpu_speedup(
            N_values=[10, 20, 30, 40, 50],
            n_iterations=100
        )
        plot_gpu_speedup(
            speedup_results,
            save_path=str(output_dir / "speedup.png")
        )
        return
    
    if args.quick:
        print("=== Quick Benchmark ===")
        results = suite.run_full_benchmark(
            small_N=[4],
            medium_N=[6, 10, 14],
            large_N=[20, 24],
            scale_N=28,
            n_trials=1,
            mts_generations=50,
            timeout_hours=args.timeout,
        )
    else:
        print("=== Full Benchmark ===")
        results = suite.run_full_benchmark(
            small_N=[4, 8],
            medium_N=[12, 16, 20],
            large_N=[24, 28],
            scale_N=32,
            n_trials=args.n_trials,
            mts_generations=50,
            timeout_hours=args.timeout,
        )
    
    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    
    # Plot 1: Method comparison
    plot_method_comparison(
        suite.results,
        save_path=str(output_dir / "method_comparison.png")
    )
    
    # Plot 2: Optimization trajectory
    if suite.trajectories:
        plot_optimization_trajectory(
            suite.trajectories,
            save_path=str(output_dir / "trajectory.png")
        )
    
    # Plot 3: Correlation heatmaps
    for i, snapshot in enumerate(suite.correlation_snapshots[:3]):  # First 3 steps
        plot_correlation_heatmap(
            snapshot,
            save_path=str(output_dir / f"correlation_step{snapshot.step}.png")
        )
    
    # Plot 4: GPU speedup
    if suite.speedup_results:
        plot_gpu_speedup(
            suite.speedup_results,
            save_path=str(output_dir / "speedup.png")
        )
    
    # Generate report
    report = generate_summary_report(
        results,
        save_path=str(output_dir / "report.txt")
    )
    print("\n" + report)
    
    # Save raw results
    suite.save_results(str(output_dir / "raw_results.json"))
    
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
