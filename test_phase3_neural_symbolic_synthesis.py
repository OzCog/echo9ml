"""
Comprehensive Test Suite for Phase 3: Neural-Symbolic Synthesis via Custom ggml Kernels

This test suite validates all Phase 3 requirements:
- Custom ggml kernels for symbolic operations
- Neural inference hooks for AtomSpace integration
- Tensor operation validation with real data
- End-to-end neural-symbolic pipeline testing
- Performance metrics and documentation
"""

import unittest
import time
import json
from typing import Dict, List, Any

# Import Phase 3 components
try:
    from neural_symbolic_synthesis import (
        NeuralSymbolicSynthesisEngine, SynthesisOperationType, 
        AtomSpaceIntegrationMode, SymbolicPattern, create_neural_symbolic_synthesis_engine
    )
    from tensor_signature_benchmark import (
        TensorSignatureBenchmark, BenchmarkType, DataSourceType,
        create_tensor_signature_benchmark
    )
    from ggml_tensor_kernel import GGMLTensorKernel, TensorOperationType
    PHASE3_COMPONENTS_AVAILABLE = True
except ImportError as e:
    PHASE3_COMPONENTS_AVAILABLE = False
    print(f"Phase 3 components not fully available: {e}")

class TestPhase3NeuralSymbolicSynthesis(unittest.TestCase):
    """Test neural-symbolic synthesis operations"""
    
    def setUp(self):
        """Set up test environment"""
        if PHASE3_COMPONENTS_AVAILABLE:
            self.engine = create_neural_symbolic_synthesis_engine("test_agent")
        else:
            self.skipTest("Phase 3 components not available")
    
    def test_symbolic_pattern_creation(self):
        """Test creation of symbolic patterns"""
        pattern_data = {
            "structure": {"type": "test_rule", "premise": "A", "conclusion": "B"},
            "confidence": 0.85,
            "temporal_context": "test_context"
        }
        
        pattern = self.engine.create_symbolic_pattern(pattern_data, "test_pattern")
        
        self.assertIsInstance(pattern, SymbolicPattern)
        self.assertEqual(pattern.pattern_type, "test_pattern")
        self.assertEqual(pattern.confidence, 0.85)
        self.assertIn(pattern.pattern_id, self.engine.symbolic_patterns)
    
    def test_symbolic_pattern_encoding(self):
        """Test symbolic pattern encoding operation"""
        pattern_data = {
            "structure": {"type": "encoding_test", "data": [1, 2, 3]},
            "confidence": 0.9
        }
        
        result = self.engine.execute_synthesis_operation(
            SynthesisOperationType.SYMBOLIC_PATTERN_ENCODE,
            {"patterns": [pattern_data], "encoding_strategy": "dense"}
        )
        
        self.assertTrue(result.get("success", False))
        self.assertIn("encoded_patterns", result)
        self.assertGreater(len(result["encoded_patterns"]), 0)
        self.assertIn("performance_metrics", result)
    
    def test_neural_symbolic_bridge(self):
        """Test neural-symbolic bridge operation"""
        result = self.engine.execute_synthesis_operation(
            SynthesisOperationType.NEURAL_SYMBOLIC_BRIDGE,
            {
                "neural_tensor_id": "test_tensor_123",
                "symbolic_context": {"features": ["feature1", "feature2"]},
                "bridge_strength": 0.8
            }
        )
        
        self.assertTrue(result.get("success", False))
        self.assertIn("bridge_result", result)
        self.assertIn("accuracy", result)
        self.assertIn("coherence", result)
        
        # Check pathway creation
        bridge_result = result["bridge_result"]
        pathway_id = bridge_result["pathway_id"]
        self.assertIn(pathway_id, self.engine.synthesis_pathways)
    
    def test_atomspace_integration(self):
        """Test AtomSpace integration with different modes"""
        test_patterns = ["pattern_1", "pattern_2"]
        
        for integration_mode in AtomSpaceIntegrationMode:
            with self.subTest(mode=integration_mode):
                result = self.engine.execute_synthesis_operation(
                    SynthesisOperationType.ATOMSPACE_INTEGRATION,
                    {
                        "patterns": test_patterns,
                        "integration_mode": integration_mode.value
                    }
                )
                
                self.assertTrue(result.get("success", False))
                self.assertIn("integration_result", result)
                self.assertIn("atomspace_score", result)
                self.assertEqual(result["integration_result"]["integration_mode"], integration_mode.value)
    
    def test_inference_synthesis(self):
        """Test neural-symbolic inference synthesis"""
        premises = [
            {"type": "premise", "content": "If A then B"},
            {"type": "premise", "content": "A is true"}
        ]
        inference_rules = [
            {"rule": "modus_ponens", "confidence": 0.9}
        ]
        
        result = self.engine.execute_synthesis_operation(
            SynthesisOperationType.INFERENCE_SYNTHESIS,
            {
                "premises": premises,
                "inference_rules": inference_rules,
                "neural_context": {"activation": [0.8, 0.6, 0.9]}
            }
        )
        
        self.assertTrue(result.get("success", False))
        self.assertIn("synthesis_results", result)
        self.assertEqual(len(result["synthesis_results"]), len(premises))
    
    def test_pathway_validation(self):
        """Test neural-symbolic pathway validation"""
        # First create some pathways
        self.engine.execute_synthesis_operation(
            SynthesisOperationType.NEURAL_SYMBOLIC_BRIDGE,
            {"neural_tensor_id": "test_tensor_1", "bridge_strength": 0.7}
        )
        
        self.engine.execute_synthesis_operation(
            SynthesisOperationType.NEURAL_SYMBOLIC_BRIDGE,
            {"neural_tensor_id": "test_tensor_2", "bridge_strength": 0.8}
        )
        
        # Validate pathways
        result = self.engine.execute_synthesis_operation(
            SynthesisOperationType.PATHWAY_VALIDATION,
            {
                "validation_criteria": {
                    "temporal_coherence": 0.8,
                    "semantic_consistency": 0.7
                }
            }
        )
        
        self.assertTrue(result.get("success", False))
        self.assertIn("validation_results", result)
        self.assertIn("valid_pathway_count", result)
        self.assertIn("overall_validation_score", result)
    
    def test_knowledge_distillation(self):
        """Test knowledge distillation from pathways"""
        # Create pathways for distillation
        pathway_ids = []
        for i in range(3):
            bridge_result = self.engine.execute_synthesis_operation(
                SynthesisOperationType.NEURAL_SYMBOLIC_BRIDGE,
                {"neural_tensor_id": f"distill_tensor_{i}", "bridge_strength": 0.75}
            )
            pathway_id = bridge_result["bridge_result"]["pathway_id"]
            pathway_ids.append(pathway_id)
        
        # Distill knowledge
        result = self.engine.execute_synthesis_operation(
            SynthesisOperationType.KNOWLEDGE_DISTILLATION,
            {
                "source_pathways": pathway_ids,
                "distillation_method": "pattern_extraction"
            }
        )
        
        self.assertTrue(result.get("success", False))
        self.assertIn("distilled_knowledge", result)
        self.assertEqual(len(result["distilled_knowledge"]), len(pathway_ids))
    
    def test_semantic_coherence_enforcement(self):
        """Test semantic coherence enforcement"""
        # Create patterns for coherence testing
        pattern_ids = []
        for i in range(3):
            pattern_data = {
                "structure": {"coherence_test": i, "data": [i, i+1, i+2]},
                "confidence": 0.7 + i * 0.1
            }
            pattern = self.engine.create_symbolic_pattern(pattern_data)
            pattern_ids.append(pattern.pattern_id)
        
        result = self.engine.execute_synthesis_operation(
            SynthesisOperationType.SEMANTIC_COHERENCE,
            {
                "target_patterns": pattern_ids,
                "coherence_threshold": 0.8
            }
        )
        
        self.assertTrue(result.get("success", False))
        self.assertIn("coherence_adjustments", result)
        self.assertIn("coherence_threshold", result)
    
    def test_symbolic_grounding(self):
        """Test symbolic pattern grounding in neural space"""
        # Create patterns for grounding
        pattern_ids = []
        for i in range(2):
            pattern_data = {
                "structure": {"grounding_test": i, "symbolic_data": f"symbol_{i}"},
                "confidence": 0.8
            }
            pattern = self.engine.create_symbolic_pattern(pattern_data)
            pattern_ids.append(pattern.pattern_id)
        
        result = self.engine.execute_synthesis_operation(
            SynthesisOperationType.SYMBOLIC_GROUNDING,
            {
                "patterns": pattern_ids,
                "grounding_strategy": "semantic_embedding"
            }
        )
        
        self.assertTrue(result.get("success", False))
        self.assertIn("grounding_results", result)
        self.assertEqual(len(result["grounding_results"]), len(pattern_ids))
    
    def test_performance_metrics_collection(self):
        """Test that performance metrics are properly collected"""
        initial_metrics_count = len(self.engine.synthesis_metrics)
        
        # Execute operation
        self.engine.execute_synthesis_operation(
            SynthesisOperationType.SYMBOLIC_PATTERN_ENCODE,
            {"patterns": [{"structure": {"test": "metrics"}}]}
        )
        
        # Check metrics were recorded
        self.assertEqual(len(self.engine.synthesis_metrics), initial_metrics_count + 1)
        
        latest_metric = self.engine.synthesis_metrics[-1]
        self.assertGreater(latest_metric.operation_latency, 0)
        self.assertGreater(latest_metric.memory_usage, 0)
        self.assertGreaterEqual(latest_metric.synthesis_accuracy, 0)
        self.assertLessEqual(latest_metric.synthesis_accuracy, 1)
    
    def test_synthesis_documentation(self):
        """Test synthesis engine documentation generation"""
        docs = self.engine.get_synthesis_documentation()
        
        self.assertIn("agent_id", docs)
        self.assertIn("synthesis_operations", docs)
        self.assertIn("atomspace_integration_modes", docs)
        self.assertIn("performance_summary", docs)
        self.assertIn("synthesis_capabilities", docs)
        
        # Check all synthesis operations are documented
        expected_operations = [op.value for op in SynthesisOperationType]
        self.assertEqual(set(docs["synthesis_operations"]), set(expected_operations))
        
        # Check all AtomSpace modes are documented
        expected_modes = [mode.value for mode in AtomSpaceIntegrationMode]
        self.assertEqual(set(docs["atomspace_integration_modes"]), set(expected_modes))


class TestPhase3TensorBenchmarking(unittest.TestCase):
    """Test tensor signature benchmarking system"""
    
    def setUp(self):
        """Set up test environment"""
        if PHASE3_COMPONENTS_AVAILABLE:
            self.benchmark = create_tensor_signature_benchmark("benchmark_agent")
        else:
            self.skipTest("Phase 3 components not available")
    
    def test_tensor_signature_creation(self):
        """Test tensor signature creation and documentation"""
        signature = self.benchmark.create_tensor_signature(
            "test_operation",
            [(64, 32), (32, 16)],
            (64, 16),
            ["input_neural", "input_symbolic", "output_synthesis"]
        )
        
        self.assertIn(signature.signature_id, self.benchmark.tensor_signatures)
        self.assertEqual(signature.operation_type, "test_operation")
        self.assertEqual(len(signature.input_shapes), 2)
        self.assertEqual(signature.output_shape, (64, 16))
        self.assertGreater(signature.parameter_count, 0)
        self.assertGreater(signature.memory_footprint, 0)
    
    def test_operation_latency_benchmark(self):
        """Test operation latency benchmarking"""
        results = self.benchmark.run_benchmark_suite([BenchmarkType.OPERATION_LATENCY], {
            "iterations": 3,
            "operation_types": ["pattern_encode"]
        })
        
        self.assertIn(BenchmarkType.OPERATION_LATENCY.value, results)
        latency_results = results[BenchmarkType.OPERATION_LATENCY.value]
        self.assertGreater(len(latency_results), 0)
        
        for result in latency_results:
            self.assertGreater(result.execution_time, 0)
            self.assertGreaterEqual(result.accuracy_score, 0)
            self.assertLessEqual(result.accuracy_score, 1)
            self.assertIn("latency_distribution", result.metadata)
    
    def test_memory_usage_benchmark(self):
        """Test memory usage benchmarking"""
        results = self.benchmark.run_benchmark_suite([BenchmarkType.MEMORY_USAGE], {
            "tensor_sizes": [(100, 100), (200, 200)]
        })
        
        self.assertIn(BenchmarkType.MEMORY_USAGE.value, results)
        memory_results = results[BenchmarkType.MEMORY_USAGE.value]
        self.assertEqual(len(memory_results), 2)  # Two tensor sizes
        
        for result in memory_results:
            self.assertGreater(result.memory_usage, 0)
            self.assertIn("memory_efficiency", result.metadata)
    
    def test_synthesis_accuracy_benchmark(self):
        """Test synthesis accuracy benchmarking"""
        results = self.benchmark.run_benchmark_suite([BenchmarkType.SYNTHESIS_ACCURACY], {
            "test_cases": 3
        })
        
        self.assertIn(BenchmarkType.SYNTHESIS_ACCURACY.value, results)
        accuracy_results = results[BenchmarkType.SYNTHESIS_ACCURACY.value]
        self.assertEqual(len(accuracy_results), 3)  # Three test cases
        
        for result in accuracy_results:
            self.assertGreaterEqual(result.accuracy_score, 0)
            self.assertLessEqual(result.accuracy_score, 1)
            self.assertGreaterEqual(result.error_rate, 0)
            self.assertLessEqual(result.error_rate, 1)
    
    def test_throughput_benchmark(self):
        """Test throughput benchmarking"""
        results = self.benchmark.run_benchmark_suite([BenchmarkType.THROUGHPUT], {
            "duration": 1.0,  # 1 second test
            "operation_types": ["pattern_encode"]
        })
        
        self.assertIn(BenchmarkType.THROUGHPUT.value, results)
        throughput_results = results[BenchmarkType.THROUGHPUT.value]
        self.assertGreater(len(throughput_results), 0)
        
        for result in throughput_results:
            self.assertGreater(result.throughput, 0)
            self.assertIn("operations_completed", result.metadata)
            self.assertIn("operations_per_second", result.metadata)
    
    def test_tensor_coherence_benchmark(self):
        """Test tensor coherence benchmarking"""
        results = self.benchmark.run_benchmark_suite([BenchmarkType.TENSOR_COHERENCE], {
            "coherence_tests": 2
        })
        
        self.assertIn(BenchmarkType.TENSOR_COHERENCE.value, results)
        coherence_results = results[BenchmarkType.TENSOR_COHERENCE.value]
        self.assertEqual(len(coherence_results), 2)
        
        for result in coherence_results:
            self.assertGreaterEqual(result.accuracy_score, 0)  # Coherence score
            self.assertLessEqual(result.accuracy_score, 1)
            self.assertIn("coherence_scores", result.metadata)
    
    def test_neural_symbolic_bridge_benchmark(self):
        """Test neural-symbolic bridge benchmarking"""
        results = self.benchmark.run_benchmark_suite([BenchmarkType.NEURAL_SYMBOLIC_BRIDGE], {
            "bridge_tests": 3
        })
        
        self.assertIn(BenchmarkType.NEURAL_SYMBOLIC_BRIDGE.value, results)
        bridge_results = results[BenchmarkType.NEURAL_SYMBOLIC_BRIDGE.value]
        self.assertEqual(len(bridge_results), 3)
        
        for result in bridge_results:
            self.assertGreaterEqual(result.accuracy_score, 0)
            self.assertLessEqual(result.accuracy_score, 1)
            self.assertIn("bridge_coherence", result.metadata)
            self.assertIn("bridge_accuracy", result.metadata)
    
    def test_real_data_validation(self):
        """Test real data validation across different sources"""
        data_sources = [DataSourceType.SYNTHETIC_PATTERNS, DataSourceType.COGNITIVE_LOGS]
        validations = self.benchmark.validate_with_real_data(data_sources, {
            "pattern_count": 5,
            "log_count": 3
        })
        
        self.assertEqual(len(validations), 2)  # Two data sources
        
        for validation in validations:
            self.assertGreater(validation.data_size, 0)
            self.assertGreaterEqual(validation.validation_accuracy, 0)
            self.assertLessEqual(validation.validation_accuracy, 1)
            self.assertGreaterEqual(validation.pattern_consistency, 0)
            self.assertLessEqual(validation.pattern_consistency, 1)
            self.assertGreaterEqual(validation.semantic_coherence, 0)
            self.assertLessEqual(validation.semantic_coherence, 1)
    
    def test_cross_agent_consistency_benchmark(self):
        """Test cross-agent consistency benchmarking"""
        results = self.benchmark.run_benchmark_suite([BenchmarkType.CROSS_AGENT_CONSISTENCY], {
            "agent_count": 3
        })
        
        self.assertIn(BenchmarkType.CROSS_AGENT_CONSISTENCY.value, results)
        consistency_results = results[BenchmarkType.CROSS_AGENT_CONSISTENCY.value]
        self.assertEqual(len(consistency_results), 1)  # Single consistency test
        
        result = consistency_results[0]
        self.assertIn("agent_results", result.metadata)
        self.assertIn("consistency_score", result.metadata)
        self.assertEqual(len(result.metadata["agent_results"]), 3)  # Three agents
    
    def test_benchmark_summary_generation(self):
        """Test benchmark summary generation"""
        # Run some benchmarks first
        self.benchmark.run_benchmark_suite([
            BenchmarkType.OPERATION_LATENCY,
            BenchmarkType.MEMORY_USAGE
        ], {"iterations": 2, "tensor_sizes": [(50, 50)]})
        
        summary = self.benchmark.get_benchmark_summary()
        
        self.assertIn("agent_id", summary)
        self.assertIn("total_benchmarks", summary)
        self.assertIn("benchmark_types", summary)
        self.assertIn("performance_summary", summary)
        self.assertGreater(summary["total_benchmarks"], 0)
        
        # Check performance summary structure
        perf_summary = summary["performance_summary"]
        for benchmark_type in summary["benchmark_types"]:
            self.assertIn(benchmark_type, perf_summary)
            type_summary = perf_summary[benchmark_type]
            self.assertIn("test_count", type_summary)
            self.assertIn("average_execution_time", type_summary)
            self.assertIn("average_accuracy", type_summary)
    
    def test_performance_report_export(self):
        """Test performance report export functionality"""
        # Run benchmark and validation
        self.benchmark.run_benchmark_suite([BenchmarkType.SYNTHESIS_ACCURACY], {"test_cases": 2})
        self.benchmark.validate_with_real_data([DataSourceType.SYNTHETIC_PATTERNS])
        
        report = self.benchmark.get_performance_report()
        
        self.assertIn("benchmark_summary", report)
        self.assertIn("detailed_results", report)
        self.assertIn("real_data_validations", report)
        self.assertIn("tensor_signatures", report)
        self.assertIn("report_timestamp", report)
        
        # Check detailed results structure
        detailed_results = report["detailed_results"]
        self.assertGreater(len(detailed_results), 0)
        for result in detailed_results:
            self.assertIn("benchmark_id", result)
            self.assertIn("benchmark_type", result)
            self.assertIn("execution_time", result)
            self.assertIn("accuracy_score", result)


class TestPhase3EndToEndPipeline(unittest.TestCase):
    """Test end-to-end neural-symbolic pipeline"""
    
    def setUp(self):
        """Set up test environment"""
        if PHASE3_COMPONENTS_AVAILABLE:
            self.synthesis_engine = create_neural_symbolic_synthesis_engine("pipeline_agent")
            self.benchmark = create_tensor_signature_benchmark("pipeline_agent")
        else:
            self.skipTest("Phase 3 components not available")
    
    def test_complete_neural_symbolic_pipeline(self):
        """Test complete neural-symbolic synthesis pipeline"""
        # Step 1: Create symbolic patterns
        pattern_data = {
            "structure": {
                "type": "logical_rule",
                "premise": "neural_activation > 0.8",
                "conclusion": "high_confidence_inference",
                "domain": "cognitive_reasoning"
            },
            "confidence": 0.85,
            "temporal_context": "reasoning_session"
        }
        
        pattern = self.synthesis_engine.create_symbolic_pattern(pattern_data, "logical_inference")
        self.assertIsNotNone(pattern)
        
        # Step 2: Encode symbolic patterns to neural tensors
        encoding_result = self.synthesis_engine.execute_synthesis_operation(
            SynthesisOperationType.SYMBOLIC_PATTERN_ENCODE,
            {"patterns": [pattern_data], "encoding_strategy": "dense"}
        )
        
        self.assertTrue(encoding_result.get("success", False))
        encoded_patterns = encoding_result["encoded_patterns"]
        self.assertGreater(len(encoded_patterns), 0)
        
        # Step 3: Create neural-symbolic bridge
        bridge_result = self.synthesis_engine.execute_synthesis_operation(
            SynthesisOperationType.NEURAL_SYMBOLIC_BRIDGE,
            {
                "neural_tensor_id": "reasoning_tensor_001",
                "symbolic_context": {
                    "features": ["logical_structure", "confidence_level", "temporal_marker"],
                    "domain": "cognitive_reasoning"
                },
                "bridge_strength": 0.8
            }
        )
        
        self.assertTrue(bridge_result.get("success", False))
        pathway_id = bridge_result["bridge_result"]["pathway_id"]
        
        # Step 4: Integrate with AtomSpace
        atomspace_result = self.synthesis_engine.execute_synthesis_operation(
            SynthesisOperationType.ATOMSPACE_INTEGRATION,
            {
                "patterns": [pattern.pattern_id],
                "integration_mode": "pattern_matching"
            }
        )
        
        self.assertTrue(atomspace_result.get("success", False))
        
        # Step 5: Perform inference synthesis
        inference_result = self.synthesis_engine.execute_synthesis_operation(
            SynthesisOperationType.INFERENCE_SYNTHESIS,
            {
                "premises": [
                    {"type": "neural_state", "activation": 0.9},
                    {"type": "symbolic_rule", "pattern_id": pattern.pattern_id}
                ],
                "inference_rules": [{"rule": "neural_symbolic_modus_ponens"}],
                "neural_context": {"pathway_id": pathway_id}
            }
        )
        
        self.assertTrue(inference_result.get("success", False))
        
        # Step 6: Validate synthesis pathways
        validation_result = self.synthesis_engine.execute_synthesis_operation(
            SynthesisOperationType.PATHWAY_VALIDATION,
            {
                "pathway_ids": [pathway_id],
                "validation_criteria": {
                    "temporal_coherence": 0.7,
                    "semantic_consistency": 0.8,
                    "neural_symbolic_alignment": 0.75
                }
            }
        )
        
        self.assertTrue(validation_result.get("success", False))
        self.assertGreater(validation_result["valid_pathway_count"], 0)
        
        # Step 7: Benchmark the complete pipeline
        pipeline_benchmark_results = self.benchmark.run_benchmark_suite([
            BenchmarkType.NEURAL_SYMBOLIC_BRIDGE,
            BenchmarkType.TENSOR_COHERENCE
        ], {"bridge_tests": 2, "coherence_tests": 2})
        
        self.assertEqual(len(pipeline_benchmark_results), 2)
        
        # Verify all benchmark types completed successfully
        for benchmark_type, results in pipeline_benchmark_results.items():
            self.assertGreater(len(results), 0, f"No results for {benchmark_type}")
            for result in results:
                self.assertGreaterEqual(result.accuracy_score, 0)
                self.assertLessEqual(result.accuracy_score, 1)
        
        # Step 8: Validate with real data
        real_data_validations = self.benchmark.validate_with_real_data([
            DataSourceType.COGNITIVE_LOGS,
            DataSourceType.BEHAVIORAL_TRACES
        ])
        
        self.assertEqual(len(real_data_validations), 2)
        for validation in real_data_validations:
            self.assertGreater(validation.data_size, 0)
            self.assertGreaterEqual(validation.validation_accuracy, 0)
    
    def test_pipeline_performance_documentation(self):
        """Test pipeline performance documentation and metrics"""
        # Execute multiple operations to generate performance data
        operations = [
            (SynthesisOperationType.SYMBOLIC_PATTERN_ENCODE, {"patterns": [{"structure": {"test": 1}}]}),
            (SynthesisOperationType.NEURAL_SYMBOLIC_BRIDGE, {"neural_tensor_id": "test_1"}),
            (SynthesisOperationType.ATOMSPACE_INTEGRATION, {"patterns": ["pattern_1"]})
        ]
        
        for i, (op_type, params) in enumerate(operations):
            result = self.synthesis_engine.execute_synthesis_operation(op_type, params)
            self.assertTrue(result.get("success", False), f"Operation {op_type.value} failed")
        
        # Get synthesis documentation
        synthesis_docs = self.synthesis_engine.get_synthesis_documentation()
        
        # Verify documentation completeness
        required_sections = [
            "agent_id", "synthesis_operations", "atomspace_integration_modes",
            "active_patterns", "active_pathways", "performance_summary", "synthesis_capabilities"
        ]
        
        for section in required_sections:
            self.assertIn(section, synthesis_docs, f"Missing documentation section: {section}")
        
        # Verify performance summary
        perf_summary = synthesis_docs["performance_summary"]
        self.assertIn("total_operations", perf_summary)
        self.assertGreater(perf_summary["total_operations"], 0)
        
        # Get benchmark performance report
        self.benchmark.run_benchmark_suite([BenchmarkType.OPERATION_LATENCY], {"iterations": 2})
        benchmark_report = self.benchmark.get_performance_report()
        
        # Verify benchmark report structure
        required_report_sections = [
            "benchmark_summary", "detailed_results", "real_data_validations", 
            "tensor_signatures", "report_timestamp"
        ]
        
        for section in required_report_sections:
            self.assertIn(section, benchmark_report, f"Missing report section: {section}")
    
    def test_pipeline_error_handling_and_recovery(self):
        """Test pipeline error handling and recovery mechanisms"""
        # Test with invalid parameters
        invalid_result = self.synthesis_engine.execute_synthesis_operation(
            SynthesisOperationType.NEURAL_SYMBOLIC_BRIDGE,
            {}  # Missing required parameters
        )
        
        self.assertFalse(invalid_result.get("success", True))
        self.assertIn("error", invalid_result)
        
        # Test with invalid operation type
        try:
            # This should be handled gracefully
            invalid_op_result = self.synthesis_engine.execute_synthesis_operation(
                "invalid_operation_type",  # This will cause a ValueError
                {"test": "data"}
            )
            self.assertFalse(invalid_op_result.get("success", True))
        except (ValueError, TypeError):
            # Expected behavior for invalid operation type
            pass
        
        # Test benchmark with invalid parameters
        invalid_benchmark = self.benchmark.run_benchmark_suite([BenchmarkType.OPERATION_LATENCY], {
            "iterations": -1  # Invalid iteration count
        })
        
        # Should handle gracefully and return results (possibly empty)
        self.assertIsInstance(invalid_benchmark, dict)


def run_phase3_tests():
    """Run all Phase 3 tests"""
    print("Running Phase 3: Neural-Symbolic Synthesis Tests")
    print("=" * 60)
    
    if not PHASE3_COMPONENTS_AVAILABLE:
        print("⚠️  Phase 3 components not available - tests will be skipped")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPhase3NeuralSymbolicSynthesis,
        TestPhase3TensorBenchmarking,
        TestPhase3EndToEndPipeline
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Phase 3 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_phase3_tests()
    exit(0 if success else 1)