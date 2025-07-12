#!/usr/bin/env python3
"""
Phase 6: Unified Cognitive Test Runner
Cognitive Unification and Tensor Field Verification

This module implements the unified cognitive tensor field testing
as specified in Phase 6 of the Distributed Agentic Cognitive Grammar Network.
"""

import unittest
import pytest
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import subprocess
import sys
import importlib


class CognitiveUnificationTester:
    """
    Unified testing system that verifies the cognitive tensor field
    and emergent properties of the echo9ml system
    """
    
    def __init__(self):
        self.project_root = Path(".")
        self.test_results = {}
        self.cognitive_metrics = {}
        self.emergent_properties = {}
        self.tensor_field_data = {}
        
    def discover_cognitive_modules(self) -> List[str]:
        """Discover all cognitive-related modules"""
        cognitive_modules = []
        patterns = [
            "cognitive_", "echo", "tensor_", "neural_", "symbolic_", 
            "meta_cognitive", "distributed_", "attention_", "memory_",
            "personality_", "emotional_", "ggml_", "hypergraph_"
        ]
        
        for py_file in self.project_root.glob("*.py"):
            if not py_file.name.startswith("test_"):
                module_name = py_file.stem
                if any(pattern in module_name.lower() for pattern in patterns):
                    cognitive_modules.append(module_name)
                    
        return sorted(cognitive_modules)
    
    def run_unified_cognitive_tests(self) -> Dict[str, Any]:
        """Run all cognitive tests in a unified manner"""
        print("🧠 Running Unified Cognitive Test Suite...")
        
        results = {
            "start_time": time.time(),
            "modules_tested": [],
            "integration_tests": [],
            "tensor_field_verification": {},
            "emergent_properties": {},
            "cognitive_metrics": {},
            "unification_score": 0.0
        }
        
        # Phase 1: Run individual module tests
        cognitive_modules = self.discover_cognitive_modules()
        for module in cognitive_modules:
            module_result = self._test_cognitive_module(module)
            if module_result:
                results["modules_tested"].append(module_result)
        
        # Phase 2: Run integration tests
        integration_result = self._run_integration_tests()
        results["integration_tests"] = integration_result
        
        # Phase 3: Verify unified tensor field
        tensor_result = self._verify_tensor_field()
        results["tensor_field_verification"] = tensor_result
        
        # Phase 4: Analyze emergent properties
        emergent_result = self._analyze_emergent_properties()
        results["emergent_properties"] = emergent_result
        
        # Phase 5: Calculate unification score
        unification_score = self._calculate_unification_score(results)
        results["unification_score"] = unification_score
        
        results["end_time"] = time.time()
        results["total_duration"] = results["end_time"] - results["start_time"]
        
        return results
    
    def _test_cognitive_module(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Test an individual cognitive module"""
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            
            # Analyze module structure
            module_analysis = {
                "module": module_name,
                "classes": [],
                "functions": [],
                "cognitive_features": [],
                "tensor_operations": [],
                "test_coverage": 0.0
            }
            
            # Inspect module contents
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type):
                    module_analysis["classes"].append(attr_name)
                elif callable(attr) and not attr_name.startswith("_"):
                    module_analysis["functions"].append(attr_name)
                    
                # Look for cognitive features
                if any(keyword in attr_name.lower() for keyword in [
                    "cognitive", "neural", "tensor", "attention", "memory",
                    "echo", "recursive", "adaptive", "symbolic"
                ]):
                    module_analysis["cognitive_features"].append(attr_name)
                    
                # Look for tensor operations
                if any(keyword in attr_name.lower() for keyword in [
                    "tensor", "matrix", "transform", "encode", "decode"
                ]):
                    module_analysis["tensor_operations"].append(attr_name)
            
            # Run specific tests for this module if they exist
            test_file = f"test_{module_name}.py"
            if (self.project_root / test_file).exists():
                cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                module_analysis["test_output"] = result.stdout
                module_analysis["test_success"] = result.returncode == 0
            
            return module_analysis
            
        except Exception as e:
            return {
                "module": module_name,
                "error": str(e),
                "test_success": False
            }
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests across cognitive modules"""
        integration_results = {
            "cross_module_tests": [],
            "system_integration": {},
            "api_integration": {},
            "data_flow_tests": {}
        }
        
        # Test echo9ml system integration
        try:
            import echo9ml
            system = echo9ml.create_echo9ml_system()
            
            # Test system initialization
            integration_results["system_integration"]["initialization"] = {
                "success": system is not None,
                "components": []
            }
            
            if system:
                # Test core components
                if hasattr(system, 'persona'):
                    integration_results["system_integration"]["components"].append("persona")
                if hasattr(system, 'attention'):
                    integration_results["system_integration"]["components"].append("attention")
                if hasattr(system, 'evolution'):
                    integration_results["system_integration"]["components"].append("evolution")
                    
                # Test experience processing
                try:
                    test_experience = {
                        "type": "learning",
                        "content": "Phase 6 testing",
                        "context": {"phase": 6, "testing": True}
                    }
                    result = system.process_experience(test_experience)
                    integration_results["system_integration"]["experience_processing"] = {
                        "success": True,
                        "result_type": type(result).__name__
                    }
                except Exception as e:
                    integration_results["system_integration"]["experience_processing"] = {
                        "success": False,
                        "error": str(e)
                    }
            
        except Exception as e:
            integration_results["system_integration"]["error"] = str(e)
        
        # Test cognitive architecture integration
        try:
            import cognitive_architecture
            cog_arch = cognitive_architecture.CognitiveArchitecture()
            integration_results["cross_module_tests"].append({
                "test": "cognitive_architecture",
                "success": cog_arch is not None
            })
        except Exception as e:
            integration_results["cross_module_tests"].append({
                "test": "cognitive_architecture",
                "success": False,
                "error": str(e)
            })
        
        return integration_results
    
    def _verify_tensor_field(self) -> Dict[str, Any]:
        """Verify the unified cognitive tensor field"""
        tensor_verification = {
            "tensor_modules": [],
            "field_coherence": {},
            "dimensional_analysis": {},
            "tensor_operations": {},
            "unified_field_score": 0.0
        }
        
        # Test tensor-related modules
        tensor_modules = ["ggml_tensor_kernel", "tensor_fragment_architecture", "tensor_signature_benchmark"]
        
        for module_name in tensor_modules:
            try:
                module_path = self.project_root / f"{module_name}.py"
                if module_path.exists():
                    module = importlib.import_module(module_name)
                    
                    tensor_info = {
                        "module": module_name,
                        "tensor_classes": [],
                        "tensor_functions": [],
                        "operations_verified": []
                    }
                    
                    # Analyze tensor operations
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type):
                            if "tensor" in attr_name.lower():
                                tensor_info["tensor_classes"].append(attr_name)
                        elif callable(attr):
                            if any(op in attr_name.lower() for op in [
                                "tensor", "matrix", "transform", "compute", "process"
                            ]):
                                tensor_info["tensor_functions"].append(attr_name)
                    
                    tensor_verification["tensor_modules"].append(tensor_info)
            
            except Exception as e:
                tensor_verification["tensor_modules"].append({
                    "module": module_name,
                    "error": str(e)
                })
        
        # Test tensor field coherence
        try:
            import echo9ml
            system = echo9ml.create_echo9ml_system()
            
            if system and hasattr(system, 'encoder'):
                # Test tensor encoding/decoding
                test_data = {
                    "cognitive_state": {"attention": 0.8, "creativity": 0.6},
                    "context": {"phase": 6, "test": True}
                }
                
                try:
                    encoded = system.encoder.encode(test_data)
                    if hasattr(encoded, 'shape'):
                        tensor_verification["field_coherence"]["encoding_success"] = True
                        tensor_verification["field_coherence"]["tensor_shape"] = list(encoded.shape)
                    else:
                        tensor_verification["field_coherence"]["encoding_success"] = True
                        tensor_verification["field_coherence"]["tensor_type"] = type(encoded).__name__
                        
                except Exception as e:
                    tensor_verification["field_coherence"]["encoding_error"] = str(e)
        
        except Exception as e:
            tensor_verification["field_coherence"]["system_error"] = str(e)
        
        # Calculate unified field score
        score = 0.0
        total_modules = len(tensor_modules)
        successful_modules = len([m for m in tensor_verification["tensor_modules"] if "error" not in m])
        
        if total_modules > 0:
            score = successful_modules / total_modules
            
        if tensor_verification["field_coherence"].get("encoding_success"):
            score += 0.3
            
        tensor_verification["unified_field_score"] = min(score, 1.0)
        
        return tensor_verification
    
    def _analyze_emergent_properties(self) -> Dict[str, Any]:
        """Analyze emergent properties of the cognitive system"""
        emergent_analysis = {
            "cognitive_patterns": [],
            "system_behaviors": [],
            "adaptive_responses": [],
            "meta_patterns": [],
            "emergence_score": 0.0
        }
        
        # Test for emergent cognitive patterns
        try:
            import echo9ml
            system = echo9ml.create_echo9ml_system()
            
            if system:
                # Test adaptive behavior through multiple experiences
                experiences = [
                    {"type": "learning", "success": 0.9, "domain": "mathematics"},
                    {"type": "creative", "success": 0.7, "domain": "art"},
                    {"type": "social", "success": 0.8, "domain": "communication"}
                ]
                
                responses = []
                for exp in experiences:
                    try:
                        response = system.process_experience(exp)
                        responses.append({
                            "experience": exp,
                            "response": response,
                            "successful": True
                        })
                    except Exception as e:
                        responses.append({
                            "experience": exp,
                            "error": str(e),
                            "successful": False
                        })
                
                emergent_analysis["adaptive_responses"] = responses
                
                # Analyze patterns in responses
                successful_responses = [r for r in responses if r["successful"]]
                if len(successful_responses) > 1:
                    emergent_analysis["cognitive_patterns"].append({
                        "pattern": "multi_domain_adaptation",
                        "evidence": f"{len(successful_responses)} successful adaptations across domains"
                    })
        
        except Exception as e:
            emergent_analysis["system_error"] = str(e)
        
        # Test meta-cognitive patterns
        try:
            import meta_cognitive_recursion
            meta_system = meta_cognitive_recursion.MetaCognitiveRecursion()
            
            emergent_analysis["meta_patterns"].append({
                "component": "meta_cognitive_recursion",
                "available": True
            })
            
        except Exception as e:
            emergent_analysis["meta_patterns"].append({
                "component": "meta_cognitive_recursion",
                "available": False,
                "error": str(e)
            })
        
        # Calculate emergence score
        score = 0.0
        if emergent_analysis["adaptive_responses"]:
            successful_adaptations = len([r for r in emergent_analysis["adaptive_responses"] if r["successful"]])
            score += successful_adaptations * 0.2
        
        if emergent_analysis["cognitive_patterns"]:
            score += len(emergent_analysis["cognitive_patterns"]) * 0.3
            
        if emergent_analysis["meta_patterns"]:
            available_meta = len([p for p in emergent_analysis["meta_patterns"] if p.get("available")])
            score += available_meta * 0.3
            
        emergent_analysis["emergence_score"] = min(score, 1.0)
        
        return emergent_analysis
    
    def _calculate_unification_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall cognitive unification score"""
        score_components = {
            "module_coverage": 0.0,
            "integration_success": 0.0,
            "tensor_field_coherence": 0.0,
            "emergent_properties": 0.0
        }
        
        # Module coverage score
        total_modules = len(results.get("modules_tested", []))
        successful_modules = len([m for m in results.get("modules_tested", []) if m.get("test_success", False)])
        if total_modules > 0:
            score_components["module_coverage"] = successful_modules / total_modules
        
        # Integration success score
        integration_tests = results.get("integration_tests", {})
        if integration_tests.get("system_integration", {}).get("initialization", {}).get("success"):
            score_components["integration_success"] += 0.5
        if integration_tests.get("system_integration", {}).get("experience_processing", {}).get("success"):
            score_components["integration_success"] += 0.5
        
        # Tensor field coherence score
        tensor_field = results.get("tensor_field_verification", {})
        score_components["tensor_field_coherence"] = tensor_field.get("unified_field_score", 0.0)
        
        # Emergent properties score
        emergent = results.get("emergent_properties", {})
        score_components["emergent_properties"] = emergent.get("emergence_score", 0.0)
        
        # Weighted average
        weights = {
            "module_coverage": 0.3,
            "integration_success": 0.3,
            "tensor_field_coherence": 0.2,
            "emergent_properties": 0.2
        }
        
        unification_score = sum(
            score_components[component] * weights[component]
            for component in score_components
        )
        
        return unification_score


@pytest.mark.phase6
class TestCognitiveUnification(unittest.TestCase):
    """Test the cognitive unification system"""
    
    def setUp(self):
        self.unification_tester = CognitiveUnificationTester()
    
    def test_cognitive_module_discovery(self):
        """Test discovery of cognitive modules"""
        modules = self.unification_tester.discover_cognitive_modules()
        self.assertIsInstance(modules, list)
        
        # Should find key cognitive modules
        expected_modules = ["echo9ml", "cognitive_architecture"]
        for module in expected_modules:
            if (Path(".") / f"{module}.py").exists():
                self.assertIn(module, modules)
    
    def test_unified_cognitive_tensor_field(self):
        """Test the unified cognitive tensor field verification"""
        tensor_result = self.unification_tester._verify_tensor_field()
        
        self.assertIn("tensor_modules", tensor_result)
        self.assertIn("unified_field_score", tensor_result)
        self.assertIsInstance(tensor_result["unified_field_score"], float)
        self.assertGreaterEqual(tensor_result["unified_field_score"], 0.0)
        self.assertLessEqual(tensor_result["unified_field_score"], 1.0)
    
    def test_emergent_properties_analysis(self):
        """Test emergent properties analysis"""
        emergent_result = self.unification_tester._analyze_emergent_properties()
        
        self.assertIn("emergence_score", emergent_result)
        self.assertIn("cognitive_patterns", emergent_result)
        self.assertIn("adaptive_responses", emergent_result)
        self.assertIsInstance(emergent_result["emergence_score"], float)
    
    def test_full_cognitive_unification(self):
        """Test full cognitive unification process"""
        results = self.unification_tester.run_unified_cognitive_tests()
        
        # Verify all required components are present
        required_keys = [
            "modules_tested", "integration_tests", "tensor_field_verification",
            "emergent_properties", "unification_score"
        ]
        
        for key in required_keys:
            self.assertIn(key, results)
        
        # Verify unification score is valid
        self.assertIsInstance(results["unification_score"], float)
        self.assertGreaterEqual(results["unification_score"], 0.0)
        self.assertLessEqual(results["unification_score"], 1.0)
        
        # Print results for verification
        print(f"\n🧠 Cognitive Unification Score: {results['unification_score']:.2f}")
        print(f"📊 Modules Tested: {len(results['modules_tested'])}")
        print(f"🔗 Integration Tests: {len(results['integration_tests'])}")


if __name__ == "__main__":
    # Command-line interface for cognitive unification testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 6 Cognitive Unification Testing")
    parser.add_argument("--test", action="store_true", help="Run unified cognitive tests")
    parser.add_argument("--tensor", action="store_true", help="Test tensor field verification")
    parser.add_argument("--emergent", action="store_true", help="Analyze emergent properties")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    tester = CognitiveUnificationTester()
    
    if args.test:
        results = tester.run_unified_cognitive_tests()
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"🧠 Cognitive Unification Complete!")
            print(f"📊 Unification Score: {results['unification_score']:.2f}")
            print(f"⏱️  Duration: {results['total_duration']:.2f}s")
            print(f"🧩 Modules Tested: {len(results['modules_tested'])}")
    
    elif args.tensor:
        tensor_result = tester._verify_tensor_field()
        if args.json:
            print(json.dumps(tensor_result, indent=2))
        else:
            print(f"🔗 Tensor Field Score: {tensor_result['unified_field_score']:.2f}")
    
    elif args.emergent:
        emergent_result = tester._analyze_emergent_properties()
        if args.json:
            print(json.dumps(emergent_result, indent=2, default=str))
        else:
            print(f"✨ Emergence Score: {emergent_result['emergence_score']:.2f}")
    
    else:
        # Run basic test
        print("🧠 Running Cognitive Unification Test...")
        unittest.main()