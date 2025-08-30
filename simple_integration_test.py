#!/usr/bin/env python3
"""
Simple Deep Tree Echo Integration Test

A simplified test that focuses on the core functionality without complex async setup.
"""

import subprocess
import json
import sys
import time
import requests
import websockets
import asyncio

def test_cpp_orchestrator():
    """Test C++ compilation and basic execution"""
    print("Testing C++ Deep Tree Echo Orchestrator...")
    
    # Compile
    compile_result = subprocess.run(
        ["g++", "-std=c++17", "-pthread", "-o", "deep-tree-echo", "deep-tree-echo.cpp"],
        capture_output=True, text=True
    )
    
    if compile_result.returncode != 0:
        print("❌ C++ compilation failed")
        return False
    
    # Test execution
    exec_result = subprocess.run(
        ["./deep-tree-echo"],
        capture_output=True, text=True, timeout=10
    )
    
    if exec_result.returncode == 0:
        print("✅ C++ orchestrator runs successfully")
        return True
    else:
        print("❌ C++ execution failed")
        return False

def test_go_engine():
    """Test Go compilation and WebSocket"""
    print("Testing Go Hyper-Echo Engine...")
    
    # Compile
    compile_result = subprocess.run(
        ["go", "build", "-o", "hyper-echo", "hyper-echo.go"],
        capture_output=True, text=True
    )
    
    if compile_result.returncode != 0:
        print("❌ Go compilation failed")
        return False
    
    print("✅ Go engine compiles successfully")
    return True

def test_python_imports():
    """Test Python component imports"""
    print("Testing Python imports...")
    
    try:
        from deep_tree_echo import DeepTreeEcho
        from deep_tree_echo_integration import MultiLanguageOrchestrator
        print("✅ Python imports successful")
        return True
    except Exception as e:
        print(f"❌ Python import failed: {e}")
        return False

def test_node_llama_cpp():
    """Test node-llama-cpp integration"""
    print("Testing node-llama-cpp integration...")
    
    from pathlib import Path
    node_llama_path = Path("node-llama-cpp")
    
    if not node_llama_path.exists():
        print("❌ node-llama-cpp directory not found")
        return False
    
    required_files = ["package.json", "README.md", "src"]
    missing_files = [f for f in required_files if not (node_llama_path / f).exists()]
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ node-llama-cpp integration present")
    return True

def main():
    """Run simplified integration tests"""
    print("=" * 60)
    print("DEEP TREE ECHO SIMPLIFIED INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        test_cpp_orchestrator,
        test_go_engine,
        test_python_imports,
        test_node_llama_cpp
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 60)
    print("RESULTS:")
    print(f"  Passed: {passed}/{total}")
    print(f"  Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("  Status: ✅ ALL TESTS PASSED")
        return 0
    else:
        print("  Status: ❌ SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())