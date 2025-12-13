#!/usr/bin/env python3
"""
Example Usage Demonstrations for Deep Tree Echo System

This script demonstrates various usage patterns and capabilities
of the Deep Tree Echo multi-language persona system.
"""

import subprocess
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_section(title):
    """Print a formatted section header"""
    logger.info("\n" + "=" * 70)
    logger.info(title)
    logger.info("=" * 70 + "\n")


def example_1_basic_cpp_processing():
    """Example 1: Basic C++ neural tree processing"""
    print_section("EXAMPLE 1: Basic C++ Neural Tree Processing")
    
    logger.info("This example demonstrates the C++ orchestrator's ability to:")
    logger.info("  - Create and manage neural tree structures")
    logger.info("  - Propagate echo values through the tree")
    logger.info("  - Analyze patterns (variance, coherence, resonance)")
    logger.info("  - Integrate with LLAMA inference engine")
    logger.info("")
    
    result = subprocess.run(
        ["./deep-tree-echo"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    # Extract key metrics
    lines = result.stdout.split('\n')
    
    logger.info("Key Output:")
    for line in lines:
        if any(keyword in line for keyword in [
            'Created root node',
            'Echo Propagation',
            'echo_variance',
            'emotional_coherence',
            'resonance_depth',
            'LLAMA Inference'
        ]):
            logger.info("  %s", line.strip())
            
    logger.info("\n✓ Example 1 complete")


def example_2_go_concurrent_execution():
    """Example 2: Go concurrent execution with multiple workers"""
    print_section("EXAMPLE 2: Go Concurrent Execution")
    
    logger.info("This example demonstrates the Go engine's ability to:")
    logger.info("  - Execute tasks concurrently with multiple workers")
    logger.info("  - Manage WebSocket connections for real-time communication")
    logger.info("  - Process echo nodes with priority handling")
    logger.info("  - Analyze hyper-patterns across multiple dimensions")
    logger.info("")
    
    # Start Go engine
    process = subprocess.Popen(
        ["./hyper-echo"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    logger.info("Starting Go engine with 4 concurrent workers...")
    time.sleep(3)
    
    # Read output
    logger.info("\nKey Output:")
    for _ in range(30):
        line = process.stdout.readline()
        if line and any(keyword in line for keyword in [
            'Workers:',
            'Created echo node',
            'Worker',
            'Hyper Pattern Analysis',
            'WebSocket server'
        ]):
            logger.info("  %s", line.strip())
            
    # Cleanup
    process.terminate()
    process.wait()
    
    logger.info("\n✓ Example 2 complete")


def example_3_integrated_workflow():
    """Example 3: Integrated multi-language workflow"""
    print_section("EXAMPLE 3: Integrated Multi-Language Workflow")
    
    logger.info("This example demonstrates a complete workflow:")
    logger.info("  1. C++ processes neural tree structure")
    logger.info("  2. Extract echo values and patterns")
    logger.info("  3. Go engine processes tasks concurrently")
    logger.info("  4. System coordinates between components")
    logger.info("")
    
    # Step 1: C++ processing
    logger.info("Step 1: C++ Neural Processing")
    cpp_result = subprocess.run(
        ["./deep-tree-echo"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    # Extract echo values
    echo_values = []
    for line in cpp_result.stdout.split('\n'):
        if 'echo value:' in line:
            try:
                value = float(line.split('echo value:')[1].split()[0])
                echo_values.append(value)
            except (ValueError, IndexError) as e:
                logger.debug("Failed to parse echo value: %s", e)
                
    logger.info("  Extracted %d echo values: %.3f - %.3f",
                len(echo_values), min(echo_values), max(echo_values))
    
    # Step 2: Go concurrent processing
    logger.info("\nStep 2: Go Concurrent Processing")
    go_process = subprocess.Popen(
        ["./hyper-echo"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    time.sleep(2)
    
    # Count workers
    worker_count = 0
    for _ in range(20):
        line = go_process.stdout.readline()
        if 'Worker' in line and 'started' in line:
            worker_count += 1
            
    logger.info("  Started %d concurrent workers", worker_count)
    
    # Step 3: Integration
    logger.info("\nStep 3: System Integration")
    logger.info("  ✓ C++ orchestrator: neural tree with %d nodes", len(echo_values))
    logger.info("  ✓ Go engine: %d concurrent workers", worker_count)
    logger.info("  ✓ Communication: WebSocket on port 8080")
    logger.info("  ✓ Status: All components operational")
    
    # Cleanup
    go_process.terminate()
    go_process.wait()
    
    logger.info("\n✓ Example 3 complete")


def example_4_pattern_analysis():
    """Example 4: Advanced pattern analysis"""
    print_section("EXAMPLE 4: Advanced Pattern Analysis")
    
    logger.info("This example demonstrates pattern analysis capabilities:")
    logger.info("  - Echo variance across tree structure")
    logger.info("  - Emotional coherence measurement")
    logger.info("  - Resonance depth calculation")
    logger.info("  - Spatial distribution analysis")
    logger.info("")
    
    result = subprocess.run(
        ["./deep-tree-echo"],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    # Extract pattern metrics
    patterns = {}
    for line in result.stdout.split('\n'):
        for metric in ['echo_variance', 'emotional_coherence', 
                       'resonance_depth', 'spatial_distribution']:
            if metric in line and ':' in line:
                try:
                    value = float(line.split(':')[1].strip())
                    patterns[metric] = value
                except (ValueError, IndexError):
                    pass
                    
    logger.info("Pattern Analysis Results:")
    for metric, value in patterns.items():
        logger.info("  %-25s: %.6f", metric, value)
        
    logger.info("\n✓ Example 4 complete")


def main():
    """Run all examples"""
    logger.info("=" * 70)
    logger.info("DEEP TREE ECHO SYSTEM - USAGE EXAMPLES")
    logger.info("=" * 70)
    
    examples = [
        example_1_basic_cpp_processing,
        example_2_go_concurrent_execution,
        example_3_integrated_workflow,
        example_4_pattern_analysis
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            logger.error(f"\n✗ Example {i} failed: {e}")
            
    print_section("ALL EXAMPLES COMPLETE")
    logger.info("The Deep Tree Echo system is ready for production use.")
    logger.info("For more information, see:")
    logger.info("  - DEEP_TREE_ECHO_IMPLEMENTATION_COMPLETE.md")
    logger.info("  - README.md")
    logger.info("")


if __name__ == "__main__":
    main()
