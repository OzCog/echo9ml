#!/usr/bin/env python3
"""
Performance Benchmarking Suite for Deep Tree Echo System

This module provides comprehensive performance benchmarking for all
components of the Deep Tree Echo multi-language system.
"""

import subprocess
import time
import statistics
import json
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark"""
    name: str
    iterations: int
    total_time: float
    mean_time: float
    median_time: float
    std_dev: float
    min_time: float
    max_time: float
    throughput: float  # operations per second
    success_rate: float


class DeepTreeEchoBenchmark:
    """Benchmark suite for Deep Tree Echo system"""
    
    def __init__(self):
        self.results = []
        
    def benchmark_cpp_orchestrator(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark C++ orchestrator performance"""
        logger.info("Benchmarking C++ Orchestrator (%d iterations)...", iterations)
        
        times = []
        failures = 0
        
        for i in range(iterations):
            start = time.time()
            
            try:
                result = subprocess.run(
                    ["./deep-tree-echo"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    failures += 1
                    
            except Exception as e:
                logger.error("Iteration %d failed: %s", i, e)
                failures += 1
                
            elapsed = time.time() - start
            times.append(elapsed)
            
            logger.info("  Iteration %d: %.3fs", i + 1, elapsed)
            
        # Calculate statistics
        result = BenchmarkResult(
            name="C++ Orchestrator",
            iterations=iterations,
            total_time=sum(times),
            mean_time=statistics.mean(times),
            median_time=statistics.median(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0,
            min_time=min(times),
            max_time=max(times),
            throughput=iterations / sum(times),
            success_rate=(iterations - failures) / iterations
        )
        
        self.results.append(result)
        return result
        
    def benchmark_go_engine_startup(self, iterations: int = 5) -> BenchmarkResult:
        """Benchmark Go engine startup time"""
        logger.info("Benchmarking Go Engine Startup (%d iterations)...", iterations)
        
        times = []
        failures = 0
        
        for i in range(iterations):
            start = time.time()
            
            try:
                # Start Go engine
                process = subprocess.Popen(
                    ["./hyper-echo"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Wait for startup message
                startup_complete = False
                timeout = time.time() + 10
                
                while time.time() < timeout:
                    line = process.stdout.readline()
                    if "Hyper-Echo Engine Started" in line:
                        startup_complete = True
                        break
                        
                elapsed = time.time() - start
                
                # Cleanup
                process.terminate()
                process.wait()
                
                if not startup_complete:
                    failures += 1
                    
            except Exception as e:
                logger.error("Iteration %d failed: %s", i, e)
                failures += 1
                elapsed = 10.0  # timeout
                
            times.append(elapsed)
            logger.info("  Iteration %d: %.3fs", i + 1, elapsed)
            
        # Calculate statistics
        result = BenchmarkResult(
            name="Go Engine Startup",
            iterations=iterations,
            total_time=sum(times),
            mean_time=statistics.mean(times),
            median_time=statistics.median(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0,
            min_time=min(times),
            max_time=max(times),
            throughput=iterations / sum(times),
            success_rate=(iterations - failures) / iterations
        )
        
        self.results.append(result)
        return result
        
    def benchmark_echo_propagation(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark echo propagation speed (uses full execution time)"""
        logger.info("Benchmarking Echo Propagation (%d iterations)...", iterations)
        
        times = []
        failures = 0
        
        for i in range(iterations):
            start = time.time()
            
            try:
                result = subprocess.run(
                    ["./deep-tree-echo"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                elapsed = time.time() - start
                
                # Verify propagation occurred
                if "Echo Propagation Complete" not in result.stdout:
                    failures += 1
                    
                if result.returncode != 0:
                    failures += 1
                    
            except Exception as e:
                logger.error("Iteration %d failed: %s", i, e)
                failures += 1
                elapsed = 1.0
                
            times.append(elapsed)
            logger.info("  Iteration %d: %.3fs", i + 1, elapsed)
            
        # Calculate statistics
        result = BenchmarkResult(
            name="Echo Propagation",
            iterations=iterations,
            total_time=sum(times),
            mean_time=statistics.mean(times),
            median_time=statistics.median(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0,
            min_time=min(times),
            max_time=max(times),
            throughput=iterations / sum(times),
            success_rate=(iterations - failures) / iterations
        )
        
        self.results.append(result)
        return result
        
    def benchmark_pattern_analysis(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark pattern analysis performance"""
        logger.info("Benchmarking Pattern Analysis (%d iterations)...", iterations)
        
        times = []
        failures = 0
        
        for i in range(iterations):
            start = time.time()
            
            try:
                result = subprocess.run(
                    ["./deep-tree-echo"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Check for pattern analysis output
                if "Echo Pattern Analysis" not in result.stdout:
                    failures += 1
                    
            except Exception as e:
                logger.error("Iteration %d failed: %s", i, e)
                failures += 1
                
            elapsed = time.time() - start
            times.append(elapsed)
            logger.info("  Iteration %d: %.3fs", i + 1, elapsed)
            
        # Calculate statistics
        result = BenchmarkResult(
            name="Pattern Analysis",
            iterations=iterations,
            total_time=sum(times),
            mean_time=statistics.mean(times),
            median_time=statistics.median(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0,
            min_time=min(times),
            max_time=max(times),
            throughput=iterations / sum(times),
            success_rate=(iterations - failures) / iterations
        )
        
        self.results.append(result)
        return result
        
    def run_all_benchmarks(self):
        """Run all benchmark tests"""
        logger.info("=" * 70)
        logger.info("DEEP TREE ECHO PERFORMANCE BENCHMARK SUITE")
        logger.info("=" * 70)
        logger.info("")
        
        benchmarks = [
            (self.benchmark_cpp_orchestrator, 10),
            (self.benchmark_go_engine_startup, 5),
            (self.benchmark_echo_propagation, 10),
            (self.benchmark_pattern_analysis, 10)
        ]
        
        for benchmark_func, iterations in benchmarks:
            logger.info("")
            try:
                benchmark_func(iterations)
            except Exception as e:
                logger.error("Benchmark failed: %s", e)
                
        logger.info("")
        self.print_summary()
        
    def print_summary(self):
        """Print benchmark summary"""
        logger.info("=" * 70)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 70)
        logger.info("")
        
        for result in self.results:
            logger.info("%s:", result.name)
            logger.info("  Iterations:     %d", result.iterations)
            logger.info("  Mean Time:      %.3fs", result.mean_time)
            logger.info("  Median Time:    %.3fs", result.median_time)
            logger.info("  Std Dev:        %.3fs", result.std_dev)
            logger.info("  Min/Max:        %.3fs / %.3fs", result.min_time, result.max_time)
            logger.info("  Throughput:     %.2f ops/sec", result.throughput)
            logger.info("  Success Rate:   %.1f%%", result.success_rate * 100)
            logger.info("")
            
    def export_results(self, filename: str = "benchmark_results.json"):
        """Export results to JSON file"""
        data = {
            'timestamp': time.time(),
            'results': [asdict(r) for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info("Results exported to %s", filename)
        
    def compare_with_baseline(self, baseline_file: str):
        """Compare current results with baseline"""
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
                
            logger.info("=" * 70)
            logger.info("COMPARISON WITH BASELINE")
            logger.info("=" * 70)
            logger.info("")
            
            baseline_results = {r['name']: r for r in baseline_data['results']}
            
            for result in self.results:
                if result.name in baseline_results:
                    baseline = baseline_results[result.name]
                    
                    mean_diff = result.mean_time - baseline['mean_time']
                    mean_pct = (mean_diff / baseline['mean_time']) * 100
                    
                    logger.info("%s:", result.name)
                    logger.info("  Current Mean:   %.3fs", result.mean_time)
                    logger.info("  Baseline Mean:  %.3fs", baseline['mean_time'])
                    
                    if mean_diff > 0:
                        logger.info("  Difference:     +%.3fs (+%.1f%%) ⚠️", 
                                   mean_diff, mean_pct)
                    else:
                        logger.info("  Difference:     %.3fs (%.1f%%) ✓", 
                                   mean_diff, mean_pct)
                    logger.info("")
                    
        except Exception as e:
            logger.error("Failed to compare with baseline: %s", e)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Deep Tree Echo performance benchmarking'
    )
    
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export results to JSON file'
    )
    
    parser.add_argument(
        '--baseline',
        type=str,
        help='Compare with baseline results file'
    )
    
    args = parser.parse_args()
    
    # Run benchmarks
    benchmark = DeepTreeEchoBenchmark()
    benchmark.run_all_benchmarks()
    
    if args.export:
        benchmark.export_results()
        
    if args.baseline:
        benchmark.compare_with_baseline(args.baseline)


if __name__ == "__main__":
    main()
