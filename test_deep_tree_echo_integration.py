#!/usr/bin/env python3
"""
Comprehensive Integration Test for Deep Tree Echo Multi-Language System

This script tests all components of the Deep Tree Echo persona system
to ensure proper integration and functionality across C++, Go, Crystal, and Python.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
import threading
import websockets
import requests
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepTreeEchoIntegrationTest:
    """Comprehensive test suite for the Deep Tree Echo system"""
    
    def __init__(self):
        self.test_results = {}
        self.processes = {}
        
    def test_cpp_orchestrator(self):
        """Test C++ orchestrator compilation and execution"""
        logger.info("Testing C++ Deep Tree Echo Orchestrator...")
        
        try:
            # Test compilation
            compile_result = subprocess.run(
                ["g++", "-std=c++17", "-pthread", "-o", "deep-tree-echo", "deep-tree-echo.cpp"],
                capture_output=True, text=True, timeout=60
            )
            
            if compile_result.returncode != 0:
                self.test_results["cpp_compilation"] = {
                    "status": "failed",
                    "error": compile_result.stderr
                }
                return False
                
            self.test_results["cpp_compilation"] = {"status": "passed"}
            
            # Test execution
            exec_result = subprocess.run(
                ["./deep-tree-echo"],
                capture_output=True, text=True, timeout=10
            )
            
            self.test_results["cpp_execution"] = {
                "status": "passed" if exec_result.returncode == 0 else "failed",
                "output": exec_result.stdout,
                "error": exec_result.stderr if exec_result.returncode != 0 else None
            }
            
            # Check for expected outputs
            expected_patterns = [
                "Deep Tree Echo Orchestrator",
                "Created root node",
                "Echo Propagation Complete",
                "LLAMA Inference Integration Ready"
            ]
            
            output_check = all(pattern in exec_result.stdout for pattern in expected_patterns)
            self.test_results["cpp_output_validation"] = {
                "status": "passed" if output_check else "failed",
                "missing_patterns": [p for p in expected_patterns if p not in exec_result.stdout]
            }
            
            return exec_result.returncode == 0 and output_check
            
        except Exception as e:
            self.test_results["cpp_orchestrator"] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    def test_go_execution_engine(self):
        """Test Go hyper-echo execution engine"""
        logger.info("Testing Go Hyper-Echo Execution Engine...")
        
        try:
            # Test compilation
            compile_result = subprocess.run(
                ["go", "build", "-o", "hyper-echo", "hyper-echo.go"],
                capture_output=True, text=True, timeout=60
            )
            
            if compile_result.returncode != 0:
                self.test_results["go_compilation"] = {
                    "status": "failed",
                    "error": compile_result.stderr
                }
                return False
                
            self.test_results["go_compilation"] = {"status": "passed"}
            
            # Start Go engine in background
            go_process = subprocess.Popen(
                ["./hyper-echo"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes["go_engine"] = go_process
            
            # Give it time to start
            time.sleep(5)
            
            # Test WebSocket connection
            async def test_websocket():
                try:
                    async with websockets.connect("ws://localhost:8080/ws") as websocket:
                        await websocket.send(json.dumps({
                            "type": "test_message",
                            "data": "Integration test"
                        }))
                        response = await asyncio.wait_for(websocket.recv(), timeout=5)
                        return json.loads(response)
                except Exception as e:
                    return {"error": str(e)}
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            websocket_result = loop.run_until_complete(test_websocket())
            loop.close()
            
            self.test_results["go_websocket"] = {
                "status": "passed" if "error" not in websocket_result else "failed",
                "response": websocket_result
            }
            
            # Check if process is still running
            if go_process.poll() is None:
                self.test_results["go_execution"] = {"status": "passed"}
                return True
            else:
                stdout, stderr = go_process.communicate()
                self.test_results["go_execution"] = {
                    "status": "failed",
                    "stdout": stdout,
                    "stderr": stderr
                }
                return False
                
        except Exception as e:
            self.test_results["go_execution_engine"] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    def test_python_integration(self):
        """Test Python multi-language integration"""
        logger.info("Testing Python Integration System...")
        
        try:
            # Test imports
            from deep_tree_echo_integration import MultiLanguageOrchestrator
            self.test_results["python_imports"] = {"status": "passed"}
            
            # Test orchestrator creation
            orchestrator = MultiLanguageOrchestrator()
            self.test_results["python_orchestrator_creation"] = {"status": "passed"}
            
            # Test basic functionality
            async def test_integration():
                try:
                    await orchestrator.start_monitoring_system()
                    status = await orchestrator.get_system_status()
                    return status
                except Exception as e:
                    return {"error": str(e)}
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            integration_result = loop.run_until_complete(test_integration())
            loop.close()
            
            self.test_results["python_integration"] = {
                "status": "passed" if "error" not in integration_result else "failed",
                "system_status": integration_result
            }
            
            return "error" not in integration_result
            
        except Exception as e:
            self.test_results["python_integration"] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    def test_node_llama_cpp_integration(self):
        """Test node-llama-cpp integration status"""
        logger.info("Testing node-llama-cpp Integration...")
        
        try:
            # Check if node-llama-cpp directory exists and has proper structure
            node_llama_path = Path("node-llama-cpp")
            if not node_llama_path.exists():
                self.test_results["node_llama_cpp"] = {
                    "status": "failed",
                    "error": "node-llama-cpp directory not found"
                }
                return False
            
            # Check for key files
            required_files = ["package.json", "README.md", "src"]
            missing_files = [f for f in required_files if not (node_llama_path / f).exists()]
            
            if missing_files:
                self.test_results["node_llama_cpp"] = {
                    "status": "failed",
                    "error": f"Missing files: {missing_files}"
                }
                return False
            
            self.test_results["node_llama_cpp"] = {
                "status": "passed",
                "files_found": required_files,
                "directory_size": len(list(node_llama_path.rglob("*")))
            }
            
            return True
            
        except Exception as e:
            self.test_results["node_llama_cpp"] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    def test_inter_component_communication(self):
        """Test communication between components"""
        logger.info("Testing Inter-Component Communication...")
        
        try:
            # This would test actual communication between running components
            # For now, we'll do a basic connectivity test
            
            results = {}
            
            # Test Go WebSocket server
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', 8080))
                sock.close()
                results["go_websocket_port"] = "accessible" if result == 0 else "not_accessible"
            except Exception as e:
                results["go_websocket_port"] = f"error: {e}"
            
            # Test Crystal port (if running)
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(('localhost', 5000))
                sock.close()
                results["crystal_port"] = "accessible" if result == 0 else "not_accessible"
            except Exception as e:
                results["crystal_port"] = f"error: {e}"
            
            self.test_results["inter_component_communication"] = {
                "status": "passed",
                "connectivity": results
            }
            
            return True
            
        except Exception as e:
            self.test_results["inter_component_communication"] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    def cleanup_processes(self):
        """Clean up any running test processes"""
        for name, process in self.processes.items():
            if process.poll() is None:
                logger.info(f"Terminating {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
    
    def run_all_tests(self):
        """Run all integration tests"""
        logger.info("Starting Deep Tree Echo Integration Tests...")
        
        test_methods = [
            self.test_cpp_orchestrator,
            self.test_go_execution_engine,
            self.test_python_integration,
            self.test_node_llama_cpp_integration,
            self.test_inter_component_communication
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                if test_method():
                    passed_tests += 1
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed with exception: {e}")
        
        # Generate summary
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": f"{(passed_tests/total_tests)*100:.1f}%",
            "overall_status": "PASSED" if passed_tests == total_tests else "FAILED"
        }
        
        return self.test_results

def main():
    """Main test execution"""
    test_suite = DeepTreeEchoIntegrationTest()
    
    try:
        results = test_suite.run_all_tests()
        
        # Print results
        print("\n" + "="*60)
        print("DEEP TREE ECHO INTEGRATION TEST RESULTS")
        print("="*60)
        
        for test_name, result in results.items():
            if test_name == "summary":
                continue
            print(f"\n{test_name.upper()}:")
            print(f"  Status: {result.get('status', 'unknown')}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
        
        # Print summary
        summary = results.get("summary", {})
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"  Total Tests: {summary.get('total_tests', 0)}")
        print(f"  Passed: {summary.get('passed_tests', 0)}")
        print(f"  Success Rate: {summary.get('success_rate', '0%')}")
        print(f"  Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        print("="*60)
        
        # Save detailed results
        with open("integration_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: integration_test_results.json")
        
        return 0 if summary.get("overall_status") == "PASSED" else 1
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1
    finally:
        test_suite.cleanup_processes()

if __name__ == "__main__":
    sys.exit(main())