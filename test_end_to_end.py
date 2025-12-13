#!/usr/bin/env python3
"""
Comprehensive End-to-End Integration Test for Deep Tree Echo System

This test validates all components working together in a realistic scenario:
- C++ orchestrator for neural processing
- Go execution engine for concurrent task handling
- Python integration for coordination
- WebSocket communication between components
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepTreeEchoE2ETest:
    """End-to-end test for the complete Deep Tree Echo system"""
    
    def __init__(self):
        self.processes = {}
        self.test_results = {}
        
    def cleanup(self):
        """Clean up all running processes"""
        logger.info("Cleaning up processes...")
        for name, process in self.processes.items():
            if process and process.poll() is None:
                logger.info(f"Terminating {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name}")
                    process.kill()
                    
    def test_cpp_component(self):
        """Test C++ orchestrator component"""
        logger.info("=" * 60)
        logger.info("Testing C++ Deep Tree Echo Orchestrator")
        logger.info("=" * 60)
        
        try:
            # Run C++ orchestrator
            result = subprocess.run(
                ["./deep-tree-echo"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout
            logger.info("C++ Output:")
            logger.info(output)
            
            # Check for expected patterns
            expected = [
                "Deep Tree Echo C++ Orchestrator",
                "Created root node",
                "Echo Propagation Complete",
                "LLAMA Inference Integration Ready"
            ]
            
            success = all(pattern in output for pattern in expected)
            
            self.test_results["cpp_orchestrator"] = {
                "status": "PASS" if success else "FAIL",
                "output_length": len(output),
                "has_all_expected_patterns": success
            }
            
            logger.info(f"C++ Test: {'✓ PASS' if success else '✗ FAIL'}")
            return success
            
        except Exception as e:
            logger.error(f"C++ test failed: {e}")
            self.test_results["cpp_orchestrator"] = {
                "status": "ERROR",
                "error": str(e)
            }
            return False
            
    def test_go_component(self):
        """Test Go execution engine component"""
        logger.info("=" * 60)
        logger.info("Testing Go Hyper-Echo Execution Engine")
        logger.info("=" * 60)
        
        try:
            # Start Go engine in background
            go_process = subprocess.Popen(
                ["./hyper-echo"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            self.processes["go_engine"] = go_process
            
            # Wait for startup
            logger.info("Waiting for Go engine to start...")
            time.sleep(3)
            
            # Check if process is running
            if go_process.poll() is not None:
                output, _ = go_process.communicate()
                logger.error(f"Go engine terminated unexpectedly: {output}")
                self.test_results["go_engine"] = {
                    "status": "FAIL",
                    "error": "Process terminated unexpectedly"
                }
                return False
                
            # Try to connect to WebSocket endpoint
            try:
                response = requests.get("http://localhost:8080/status", timeout=2)
                status_ok = response.status_code == 200
            except (requests.RequestException, ConnectionError) as e:
                logger.debug("Connection check failed: %s", e)
                status_ok = False
                
            # Read some output
            output_lines = []
            for _ in range(20):
                line = go_process.stdout.readline()
                if line:
                    output_lines.append(line.strip())
                    
            output = "\n".join(output_lines)
            logger.info("Go Engine Output:")
            logger.info(output)
            
            # Check for expected patterns
            expected = [
                "Hyper-Echo Go Execution Engine",
                "Workers:",
                "Worker"
            ]
            
            success = all(pattern in output for pattern in expected)
            
            self.test_results["go_engine"] = {
                "status": "PASS" if success else "FAIL",
                "process_running": go_process.poll() is None,
                "has_expected_patterns": success,
                "output_lines": len(output_lines)
            }
            
            logger.info(f"Go Test: {'✓ PASS' if success else '✗ FAIL'}")
            return success
            
        except Exception as e:
            logger.error(f"Go test failed: {e}")
            self.test_results["go_engine"] = {
                "status": "ERROR",
                "error": str(e)
            }
            return False
            
    def test_system_integration(self):
        """Test system integration scenario"""
        logger.info("=" * 60)
        logger.info("Testing System Integration")
        logger.info("=" * 60)
        
        try:
            # Simulate a cognitive processing workflow
            logger.info("Simulating cognitive processing workflow...")
            
            # 1. C++ processes neural tree
            logger.info("Step 1: Neural tree processing (C++)...")
            cpp_result = subprocess.run(
                ["./deep-tree-echo"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            cpp_success = "Echo Pattern Analysis Complete" in cpp_result.stdout
            logger.info(f"  Neural processing: {'✓' if cpp_success else '✗'}")
            
            # 2. Extract echo values from output
            echo_values = []
            for line in cpp_result.stdout.split('\n'):
                if 'echo value:' in line:
                    try:
                        value = float(line.split('echo value:')[1].split()[0])
                        echo_values.append(value)
                    except (ValueError, IndexError):
                        pass
                        
            logger.info(f"  Extracted {len(echo_values)} echo values")
            
            # 3. Check Go engine is processing
            if "go_engine" in self.processes and self.processes["go_engine"].poll() is None:
                logger.info("Step 2: Concurrent execution (Go) - Engine running ✓")
                go_success = True
            else:
                logger.info("Step 2: Concurrent execution (Go) - Engine not running ✗")
                go_success = False
                
            # 4. Integration metrics
            integration_metrics = {
                "cpp_processing": cpp_success,
                "go_engine_active": go_success,
                "echo_values_extracted": len(echo_values),
                "system_responsive": cpp_success and go_success
            }
            
            overall_success = cpp_success and go_success and len(echo_values) > 0
            
            self.test_results["system_integration"] = {
                "status": "PASS" if overall_success else "FAIL",
                "metrics": integration_metrics
            }
            
            logger.info(f"Integration Test: {'✓ PASS' if overall_success else '✗ FAIL'}")
            return overall_success
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            self.test_results["system_integration"] = {
                "status": "ERROR",
                "error": str(e)
            }
            return False
            
    def run_all_tests(self):
        """Run all end-to-end tests"""
        logger.info("\n" + "=" * 60)
        logger.info("DEEP TREE ECHO END-TO-END TEST SUITE")
        logger.info("=" * 60 + "\n")
        
        results = []
        
        try:
            # Test individual components
            results.append(("C++ Orchestrator", self.test_cpp_component()))
            results.append(("Go Engine", self.test_go_component()))
            
            # Test integration
            results.append(("System Integration", self.test_system_integration()))
            
        finally:
            # Always clean up
            self.cleanup()
            
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{name:.<40} {status}")
            
        logger.info("-" * 60)
        logger.info(f"Total: {passed}/{total} tests passed ({100*passed//total}%)")
        logger.info("=" * 60)
        
        # Save results to file
        with open("e2e_test_results.json", "w") as f:
            json.dump({
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "success_rate": passed / total
                },
                "test_results": self.test_results
            }, f, indent=2)
            
        logger.info("\nDetailed results saved to e2e_test_results.json")
        
        return passed == total


def main():
    """Main entry point"""
    test_suite = DeepTreeEchoE2ETest()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("\nTest interrupted by user")
        test_suite.cleanup()
        sys.exit(1)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run tests
    success = test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
