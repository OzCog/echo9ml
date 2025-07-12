#!/usr/bin/env python3
"""
Phase 6: Deep Testing Protocols for Echo9ML
Comprehensive test coverage and verification system

This module implements rigorous testing protocols as specified in Phase 6
of the Distributed Agentic Cognitive Grammar Network implementation.
"""

import unittest
import pytest
import coverage
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
import importlib.util
import inspect
import ast


class DeepTestingProtocols:
    """
    Implements comprehensive testing protocols for Phase 6
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.coverage_target = 100.0  # Phase 6 goal
        self.test_results = {}
        self.coverage_data = {}
        
    def discover_all_modules(self) -> List[str]:
        """Discover all Python modules in the project"""
        modules = []
        for py_file in self.project_root.glob("*.py"):
            if not py_file.name.startswith("test_") and py_file.name != "setup.py":
                modules.append(py_file.stem)
        return modules
    
    def analyze_module_complexity(self, module_name: str) -> Dict[str, Any]:
        """Analyze complexity of a module for test prioritization"""
        try:
            module_path = self.project_root / f"{module_name}.py"
            if not module_path.exists():
                return {"error": f"Module {module_name} not found"}
                
            with open(module_path, 'r', encoding='utf-8') as f:
                source = f.read()
                
            tree = ast.parse(source)
            
            functions = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": len(node.args.args),
                        "complexity": self._calculate_complexity(node)
                    })
                elif isinstance(node, ast.ClassDef):
                    class_methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_methods.append({
                                "name": item.name,
                                "line": item.lineno,
                                "args": len(item.args.args),
                                "complexity": self._calculate_complexity(item)
                            })
                    
                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": class_methods
                    })
            
            return {
                "module": module_name,
                "functions": functions,
                "classes": classes,
                "total_functions": len(functions),
                "total_classes": len(classes),
                "total_methods": sum(len(cls["methods"]) for cls in classes),
                "lines_of_code": len(source.splitlines())
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function/method"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all tests with comprehensive coverage analysis"""
        print("🧪 Running comprehensive test suite...")
        
        # Run pytest with coverage
        cmd = [
            sys.executable, "-m", "pytest", 
            "--cov=.",
            "--cov-report=json:coverage_detailed.json",
            "--cov-report=html:coverage_html_detailed",
            "--cov-report=term-missing",
            "--tb=short",
            "-v"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
        
        # Load coverage data
        coverage_file = self.project_root / "coverage_detailed.json"
        if coverage_file.exists():
            with open(coverage_file, 'r') as f:
                self.coverage_data = json.load(f)
        
        return {
            "test_output": result.stdout,
            "test_errors": result.stderr,
            "return_code": result.returncode,
            "coverage_data": self.coverage_data
        }
    
    def identify_coverage_gaps(self) -> Dict[str, List[str]]:
        """Identify modules and functions with low coverage"""
        gaps = {
            "untested_modules": [],
            "low_coverage_modules": [],
            "untested_functions": [],
            "critical_gaps": []
        }
        
        if not self.coverage_data:
            return gaps
            
        files = self.coverage_data.get("files", {})
        
        for file_path, file_data in files.items():
            if file_path.endswith(".py") and not file_path.startswith("test_"):
                coverage_percent = file_data.get("summary", {}).get("percent_covered", 0)
                
                if coverage_percent == 0:
                    gaps["untested_modules"].append(file_path)
                elif coverage_percent < 50:
                    gaps["low_coverage_modules"].append({
                        "file": file_path,
                        "coverage": coverage_percent,
                        "missing_lines": file_data.get("missing_lines", [])
                    })
                
                # Identify critical gaps in core modules
                if any(core in file_path for core in ["echo9ml.py", "cognitive_", "ggml_", "tensor_"]):
                    if coverage_percent < 80:
                        gaps["critical_gaps"].append({
                            "file": file_path,
                            "coverage": coverage_percent,
                            "reason": "Core module with insufficient coverage"
                        })
        
        return gaps
    
    def generate_test_priorities(self) -> List[Dict[str, Any]]:
        """Generate test priority list based on complexity and coverage"""
        modules = self.discover_all_modules()
        priorities = []
        
        for module in modules:
            analysis = self.analyze_module_complexity(module)
            if "error" not in analysis:
                
                # Calculate priority score
                complexity_score = (
                    analysis["total_functions"] * 1.0 +
                    analysis["total_methods"] * 1.2 +
                    analysis["total_classes"] * 1.5 +
                    analysis["lines_of_code"] * 0.01
                )
                
                # Get current coverage
                current_coverage = 0
                if self.coverage_data and "files" in self.coverage_data:
                    file_key = f"{module}.py"
                    if file_key in self.coverage_data["files"]:
                        current_coverage = self.coverage_data["files"][file_key].get(
                            "summary", {}
                        ).get("percent_covered", 0)
                
                # Priority = complexity / (coverage + 1) to prioritize complex, untested code
                priority_score = complexity_score / (current_coverage + 1)
                
                priorities.append({
                    "module": module,
                    "priority_score": priority_score,
                    "complexity_score": complexity_score,
                    "current_coverage": current_coverage,
                    "analysis": analysis
                })
        
        # Sort by priority score (highest first)
        return sorted(priorities, key=lambda x: x["priority_score"], reverse=True)


@pytest.mark.phase6
class TestDeepTestingProtocols(unittest.TestCase):
    """Test the deep testing protocols themselves"""
    
    def setUp(self):
        self.protocols = DeepTestingProtocols()
    
    def test_module_discovery(self):
        """Test that module discovery works correctly"""
        modules = self.protocols.discover_all_modules()
        self.assertIsInstance(modules, list)
        self.assertGreater(len(modules), 0)
        
        # Should include core modules
        core_modules = ["echo9ml", "cognitive_architecture", "ggml_tensor_kernel"]
        for module in core_modules:
            if (Path(".") / f"{module}.py").exists():
                self.assertIn(module, modules)
    
    def test_complexity_analysis(self):
        """Test module complexity analysis"""
        # Test with echo9ml module (should exist)
        analysis = self.protocols.analyze_module_complexity("echo9ml")
        
        if "error" not in analysis:
            self.assertIn("functions", analysis)
            self.assertIn("classes", analysis)
            self.assertIn("lines_of_code", analysis)
            self.assertIsInstance(analysis["total_functions"], int)
            self.assertIsInstance(analysis["total_classes"], int)
    
    def test_priority_generation(self):
        """Test test priority generation"""
        priorities = self.protocols.generate_test_priorities()
        self.assertIsInstance(priorities, list)
        
        if priorities:
            # Check first item has required fields
            first = priorities[0]
            self.assertIn("module", first)
            self.assertIn("priority_score", first)
            self.assertIn("complexity_score", first)
            self.assertIn("current_coverage", first)


if __name__ == "__main__":
    # Command-line interface for deep testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 6 Deep Testing Protocols")
    parser.add_argument("--analyze", action="store_true", help="Analyze code complexity")
    parser.add_argument("--test", action="store_true", help="Run comprehensive tests")
    parser.add_argument("--gaps", action="store_true", help="Identify coverage gaps")
    parser.add_argument("--priorities", action="store_true", help="Generate test priorities")
    parser.add_argument("--module", type=str, help="Analyze specific module")
    
    args = parser.parse_args()
    
    protocols = DeepTestingProtocols()
    
    if args.analyze or args.module:
        if args.module:
            analysis = protocols.analyze_module_complexity(args.module)
            print(f"Analysis for {args.module}:")
            print(json.dumps(analysis, indent=2))
        else:
            modules = protocols.discover_all_modules()
            print(f"Discovered {len(modules)} modules")
            for module in modules[:10]:  # Show first 10
                analysis = protocols.analyze_module_complexity(module)
                if "error" not in analysis:
                    print(f"{module}: {analysis['total_functions']} functions, "
                          f"{analysis['total_classes']} classes, "
                          f"{analysis['lines_of_code']} LOC")
    
    if args.test:
        results = protocols.run_comprehensive_tests()
        print("Test Results Summary:")
        print(f"Return code: {results['return_code']}")
        print("Coverage data loaded:", "coverage_data" in results and bool(results["coverage_data"]))
    
    if args.gaps:
        protocols.run_comprehensive_tests()  # Ensure we have coverage data
        gaps = protocols.identify_coverage_gaps()
        print("Coverage Gaps Analysis:")
        print(json.dumps(gaps, indent=2))
    
    if args.priorities:
        protocols.run_comprehensive_tests()  # Ensure we have coverage data
        priorities = protocols.generate_test_priorities()
        print("Test Priorities (Top 10):")
        for i, item in enumerate(priorities[:10], 1):
            print(f"{i}. {item['module']}: priority={item['priority_score']:.2f}, "
                  f"coverage={item['current_coverage']:.1f}%")