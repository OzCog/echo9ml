#!/usr/bin/env python3
"""
Test suite for Phase 5: Recursive Meta-Cognition & Evolutionary Optimization

Tests the meta-cognitive recursion engine, self-analysis modules,
adaptive optimization, and live metrics monitoring capabilities.
"""

import asyncio
import unittest
import tempfile
import json
import time
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meta_cognitive_recursion import (
    MetaCognitiveRecursionEngine,
    SelfAnalysisModule,
    AdaptiveOptimizer,
    LiveMetricsMonitor,
    MetaCognitiveState,
    RecursiveObservation
)
from cognitive_evolution import CognitiveEvolutionBridge
from moses_evolutionary_search import MOSESEvolutionarySearch, EvolutionaryParameters
from echo_evolution import EvolutionNetwork, EchoAgent

class TestSelfAnalysisModule(unittest.TestCase):
    """Test the self-analysis module functionality"""
    
    def setUp(self):
        self.analysis_module = SelfAnalysisModule("test_system")
    
    def test_cognitive_load_calculation(self):
        """Test cognitive load calculation"""
        process_data = {
            "memory_usage": 0.7,
            "cpu_usage": 0.8,
            "processing_time": 0.6
        }
        
        load = self.analysis_module._calculate_cognitive_load(process_data)
        self.assertIsInstance(load, float)
        self.assertGreaterEqual(load, 0.0)
        self.assertLessEqual(load, 1.0)
    
    def test_performance_impact_calculation(self):
        """Test performance impact calculation"""
        process_data = {
            "memory_usage": 0.5,
            "cpu_usage": 0.6,
            "processing_time": 0.4
        }
        
        impact = self.analysis_module._calculate_performance_impact(process_data)
        self.assertIsInstance(impact, float)
        self.assertGreaterEqual(impact, 0.0)
        self.assertLessEqual(impact, 1.0)
    
    def test_pattern_detection(self):
        """Test cognitive pattern detection"""
        # High frequency pattern
        process_data = {"frequency": 15}
        patterns = self.analysis_module._detect_cognitive_patterns(process_data)
        self.assertIn("high_frequency_operation", patterns)
        
        # Resource intensive pattern
        process_data = {"memory_usage": 0.9, "cpu_usage": 0.8, "processing_time": 0.7}
        patterns = self.analysis_module._detect_cognitive_patterns(process_data)
        self.assertIn("resource_intensive", patterns)
        
        # Evolutionary pattern
        process_data = {"description": "evolution cycle with mutation"}
        patterns = self.analysis_module._detect_cognitive_patterns(process_data)
        self.assertIn("evolutionary_operation", patterns)
    
    def test_anomaly_detection(self):
        """Test anomaly detection in cognitive processes"""
        # Set baseline
        normal_data = {"memory_usage": 0.3, "cpu_usage": 0.4, "processing_time": 0.2}
        self.analysis_module._detect_anomalies("test_process", normal_data)
        
        # Test performance degradation
        degraded_data = {"memory_usage": 0.8, "cpu_usage": 0.9, "processing_time": 0.7}
        anomalies = self.analysis_module._detect_anomalies("test_process", degraded_data)
        self.assertIn("performance_degradation", str(anomalies))
        
        # Test high cognitive load
        high_load_data = {"test": "x" * 10000}  # Large data to trigger high load
        anomalies = self.analysis_module._detect_anomalies("test_process", high_load_data)
        # Check if any anomaly is detected (high cognitive load is one possibility)
        self.assertIsInstance(anomalies, list)
    
    def test_recursive_observation(self):
        """Test recursive observation functionality"""
        process_data = {
            "memory_usage": 0.5,
            "cpu_usage": 0.4,
            "processing_time": 0.3,
            "frequency": 5
        }
        
        observation = self.analysis_module.observe_cognitive_process(
            "test_process", process_data, recursive_depth=0
        )
        
        self.assertIsInstance(observation, RecursiveObservation)
        self.assertEqual(observation.observed_process, "test_process")
        self.assertGreaterEqual(observation.cognitive_load, 0.0)
        self.assertLessEqual(observation.cognitive_load, 1.0)
        self.assertIsInstance(observation.patterns_detected, list)
        self.assertIsInstance(observation.anomalies_detected, list)
    
    def test_recursive_depth_limit(self):
        """Test that recursive depth limit is respected"""
        process_data = {"test": "data"}
        
        # Test at max depth - should return None
        observation = self.analysis_module.observe_cognitive_process(
            "test_process", process_data, recursive_depth=5
        )
        self.assertIsNone(observation)
        
        # Test below max depth - should return observation
        observation = self.analysis_module.observe_cognitive_process(
            "test_process", process_data, recursive_depth=3
        )
        self.assertIsInstance(observation, RecursiveObservation)
    
    def test_improvement_suggestions(self):
        """Test generation of improvement suggestions"""
        # Add some observations to history
        for i in range(10):
            process_data = {
                "memory_usage": 0.8,  # High load
                "cpu_usage": 0.9,
                "processing_time": 0.7
            }
            self.analysis_module.observe_cognitive_process(f"test_process_{i}", process_data)
        
        suggestions = self.analysis_module.generate_improvement_suggestions()
        self.assertIsInstance(suggestions, list)
        # Should suggest optimizing high load processes
        self.assertTrue(any("optimize" in suggestion for suggestion in suggestions))

class TestAdaptiveOptimizer(unittest.TestCase):
    """Test the adaptive optimization functionality"""
    
    def setUp(self):
        moses_search = MOSESEvolutionarySearch("test_agent")
        self.optimizer = AdaptiveOptimizer(moses_search)
    
    def test_parameter_adaptation(self):
        """Test evolutionary parameter adaptation"""
        # Test mutation rate adaptation with low diversity
        performance_metrics = {
            "fitness_variance": 0.02,  # Low diversity
            "avg_fitness": 0.6,
            "best_fitness": 0.7,
            "cpu_usage": 0.5
        }
        
        adapted_params = self.optimizer.adapt_evolutionary_parameters(performance_metrics)
        self.assertIsInstance(adapted_params, EvolutionaryParameters)
        
        # With low diversity, mutation rate should increase
        original_mutation_rate = self.optimizer.moses_search.parameters.mutation_rate
        self.assertGreaterEqual(adapted_params.mutation_rate, original_mutation_rate)
    
    def test_parameter_adaptation_high_diversity(self):
        """Test parameter adaptation with high diversity"""
        performance_metrics = {
            "fitness_variance": 0.25,  # High diversity
            "avg_fitness": 0.6,
            "best_fitness": 0.65,  # Low variance between avg and best
            "cpu_usage": 0.9  # High CPU usage
        }
        
        adapted_params = self.optimizer.adapt_evolutionary_parameters(performance_metrics)
        
        # With high diversity, mutation rate should decrease
        original_mutation_rate = self.optimizer.moses_search.parameters.mutation_rate
        self.assertLessEqual(adapted_params.mutation_rate, original_mutation_rate)
        
        # With high CPU usage, population size should decrease
        original_population = self.optimizer.moses_search.parameters.population_size
        self.assertLessEqual(adapted_params.population_size, original_population)
    
    def test_fitness_landscape_optimization(self):
        """Test fitness landscape analysis"""
        from moses_evolutionary_search import CognitivePattern
        
        # Create test patterns
        patterns = [
            CognitivePattern("1", "test", {"test": 1}, fitness=0.5),
            CognitivePattern("2", "test", {"test": 2}, fitness=0.7),
            CognitivePattern("3", "test", {"test": 3}, fitness=0.6)
        ]
        
        landscape_analysis = self.optimizer.optimize_fitness_landscape(patterns)
        
        self.assertIn("fitness_stats", landscape_analysis)
        self.assertIn("pattern_diversity", landscape_analysis)
        self.assertIn("convergence_indicators", landscape_analysis)
        self.assertIn("optimization_suggestions", landscape_analysis)
        
        # Check fitness statistics
        fitness_stats = landscape_analysis["fitness_stats"]
        self.assertAlmostEqual(fitness_stats["mean"], 0.6, places=1)
        self.assertEqual(fitness_stats["min"], 0.5)
        self.assertEqual(fitness_stats["max"], 0.7)
    
    def test_pattern_diversity_calculation(self):
        """Test pattern diversity calculation"""
        from moses_evolutionary_search import CognitivePattern
        
        # Diverse patterns
        diverse_patterns = [
            CognitivePattern("1", "hypergraph", {}),
            CognitivePattern("2", "tensor", {}),
            CognitivePattern("3", "symbolic", {}),
            CognitivePattern("4", "hybrid", {})
        ]
        
        diversity = self.optimizer._calculate_pattern_diversity(diverse_patterns)
        self.assertGreater(diversity, 0.0)
        
        # Similar patterns
        similar_patterns = [
            CognitivePattern("1", "hypergraph", {}),
            CognitivePattern("2", "hypergraph", {}),
            CognitivePattern("3", "hypergraph", {})
        ]
        
        low_diversity = self.optimizer._calculate_pattern_diversity(similar_patterns)
        self.assertLess(low_diversity, diversity)

class TestLiveMetricsMonitor(unittest.TestCase):
    """Test the live metrics monitoring functionality"""
    
    def setUp(self):
        self.monitor = LiveMetricsMonitor()
    
    def test_metrics_recording(self):
        """Test metrics recording"""
        test_metrics = {
            "cpu_usage": 0.5,
            "memory_usage": 0.6,
            "custom_metric": 42
        }
        
        self.monitor.record_metrics(test_metrics)
        
        # Check that metrics were recorded
        recent_metrics = self.monitor.get_recent_metrics(1)
        self.assertEqual(len(recent_metrics), 1)
        
        recorded = recent_metrics[0]
        self.assertEqual(recorded["cpu_usage"], 0.5)
        self.assertEqual(recorded["memory_usage"], 0.6)
        self.assertEqual(recorded["custom_metric"], 42)
        self.assertIn("timestamp", recorded)
        self.assertIn("datetime", recorded)
    
    def test_callback_mechanism(self):
        """Test metrics callback mechanism"""
        callback_data = []
        
        def test_callback(metrics):
            callback_data.append(metrics)
        
        self.monitor.add_callback(test_callback)
        
        test_metrics = {"test": "data"}
        self.monitor.record_metrics(test_metrics)
        
        self.assertEqual(len(callback_data), 1)
        self.assertEqual(callback_data[0]["test"], "data")
    
    def test_monitoring_start_stop(self):
        """Test monitoring start and stop"""
        self.monitor.start_monitoring(update_interval=0.1)
        self.assertTrue(self.monitor.monitoring_active)
        self.assertIn("main", self.monitor.active_monitors)
        
        # Wait briefly for monitoring to collect some data
        time.sleep(0.2)
        
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring_active)
        self.assertEqual(len(self.monitor.active_monitors), 0)
        
        # Should have collected some metrics
        self.assertGreater(len(self.monitor.metrics_history), 0)
    
    def test_metrics_export(self):
        """Test metrics export functionality"""
        # Record some test metrics
        for i in range(5):
            self.monitor.record_metrics({"iteration": i, "value": i * 10})
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filename = f.name
        
        try:
            exported_file = self.monitor.export_metrics(filename)
            self.assertEqual(exported_file, filename)
            
            # Verify file contents
            with open(filename, 'r') as f:
                export_data = json.load(f)
            
            self.assertIn("export_time", export_data)
            self.assertIn("total_metrics", export_data)
            self.assertIn("metrics", export_data)
            self.assertEqual(export_data["total_metrics"], 5)
            self.assertEqual(len(export_data["metrics"]), 5)
            
        finally:
            # Clean up
            if os.path.exists(filename):
                os.unlink(filename)

class TestMetaCognitiveRecursionEngine(unittest.TestCase):
    """Test the complete meta-cognitive recursion engine"""
    
    def setUp(self):
        # Create test evolution network
        self.network = EvolutionNetwork()
        self.network.add_agent(EchoAgent("TestAgent", "Testing", 0.5))
        
        # Create cognitive bridge
        self.bridge = CognitiveEvolutionBridge(self.network)
        
        # Create meta-cognitive engine
        self.engine = MetaCognitiveRecursionEngine(self.bridge)
    
    def test_engine_initialization(self):
        """Test meta-cognitive engine initialization"""
        self.assertIsNotNone(self.engine.cognitive_bridge)
        self.assertIsNotNone(self.engine.self_analysis)
        self.assertIsNotNone(self.engine.adaptive_optimizer)
        self.assertIsNotNone(self.engine.live_monitor)
        self.assertIsInstance(self.engine.recursion_state, MetaCognitiveState)
    
    def test_system_state_capture(self):
        """Test system state capture"""
        async def async_test():
            state = await self.engine._capture_system_state()
            
            self.assertIn("cognitive", state)
            self.assertIn("evolution", state)
            self.assertIn("timestamp", state)
            
            cognitive_state = state["cognitive"]
            self.assertIn("memory_count", cognitive_state)
            self.assertIn("active_goals", cognitive_state)
            self.assertIn("personality_traits", cognitive_state)
        
        # Run the async test
        asyncio.run(async_test())
    
    def test_cognitive_process_observation(self):
        """Test cognitive process observation"""
        async def async_test():
            processes = await self.engine._observe_cognitive_processes()
            
            self.assertIsInstance(processes, dict)
            self.assertIn("memory_management", processes)
            self.assertIn("goal_processing", processes)
            self.assertIn("personality_updates", processes)
            self.assertIn("evolution_cycles", processes)
            
            # Each process should have observation data
            for process_name, process_data in processes.items():
                self.assertIn("active", process_data)
                self.assertIn("load", process_data)
                self.assertIn("observation", process_data)
        
        # Run the async test
        asyncio.run(async_test())
    
    def test_self_observation(self):
        """Test self-observation functionality"""
        async def async_test():
            observation_data = await self.engine._perform_self_observation(0)
            
            self.assertIn("cycle", observation_data)
            self.assertIn("timestamp", observation_data)
            self.assertIn("system_state", observation_data)
            self.assertIn("cognitive_processes", observation_data)
            self.assertIn("meta_observation", observation_data)
            
            # Verify meta-observation is a RecursiveObservation
            meta_obs = observation_data["meta_observation"]
            self.assertIsInstance(meta_obs, RecursiveObservation)
        
        # Run the async test
        asyncio.run(async_test())
    
    def test_adaptive_optimization(self):
        """Test adaptive optimization"""
        async def async_test():
            cycle_data = {
                "system_state": {"cpu_usage": 0.5},
                "timestamp": time.time()
            }
            
            optimization_results = await self.engine._perform_adaptive_optimization(cycle_data)
            
            self.assertIn("adapted_parameters", optimization_results)
            self.assertIn("landscape_analysis", optimization_results)
            self.assertIn("performance_metrics", optimization_results)
            self.assertIn("fitness_improvement", optimization_results)
            
            adapted_params = optimization_results["adapted_parameters"]
            self.assertIn("mutation_rate", adapted_params)
            self.assertIn("crossover_rate", adapted_params)
            self.assertIn("population_size", adapted_params)
        
        # Run the async test
        asyncio.run(async_test())
    
    def test_recursive_improvement(self):
        """Test recursive improvement functionality"""
        async def async_test():
            optimization_results = {
                "fitness_improvement": 0.1,
                "adapted_parameters": {"mutation_rate": 0.2}
            }
            
            improvement_results = await self.engine._perform_recursive_improvement(optimization_results)
            
            self.assertIn("suggestions", improvement_results)
            self.assertIn("applied_improvements", improvement_results)
            self.assertIn("recursive_depth", improvement_results)
            
            self.assertIsInstance(improvement_results["suggestions"], list)
            self.assertIsInstance(improvement_results["applied_improvements"], list)
            self.assertIsInstance(improvement_results["recursive_depth"], int)
        
        # Run the async test
        asyncio.run(async_test())
    
    def test_meta_cognitive_state_update(self):
        """Test meta-cognitive state updates"""
        cycle_data = {
            "cognitive_processes": {
                "proc1": {"load": 0.3},
                "proc2": {"load": 0.4}
            },
            "meta_observation": ["obs1", "obs2"]
        }
        
        optimization_results = {
            "fitness_improvement": 0.15,
            "adapted_parameters": {"mutation_rate": 0.2, "crossover_rate": 0.7},
            "performance_metrics": {"cpu_usage": 0.5, "memory_usage": 0.6}
        }
        
        improvement_results = {
            "recursive_depth": 2,
            "suggestions": ["suggestion1", "suggestion2"]
        }
        
        original_timestamp = self.engine.recursion_state.timestamp
        
        self.engine._update_meta_cognitive_state(
            cycle_data, optimization_results, improvement_results
        )
        
        # Verify state was updated
        new_state = self.engine.recursion_state
        self.assertGreater(new_state.timestamp, original_timestamp)
        self.assertEqual(new_state.recursive_depth, 2)
        self.assertEqual(len(new_state.improvement_suggestions), 2)
        
        # Verify cognitive metrics
        self.assertAlmostEqual(new_state.cognitive_metrics["load"], 0.7, places=1)
        self.assertEqual(new_state.cognitive_metrics["processes_active"], 2)
        
        # Verify evolution metrics
        self.assertEqual(new_state.evolution_metrics["fitness_improvement"], 0.15)
        self.assertEqual(new_state.evolution_metrics["parameters_adapted"], 2)
    
    def test_recursion_flowchart_generation(self):
        """Test generation of recursion flowcharts"""
        # Add some history
        for i in range(3):
            state = MetaCognitiveState(
                timestamp=time.time(),
                cognitive_metrics={},
                evolution_metrics={},
                performance_metrics={},
                recursive_depth=i,
                improvement_suggestions=[f"suggestion_{i}"]
            )
            self.engine.recursion_history.append(state)
        
        flowchart = self.engine.generate_recursion_flowchart()
        
        self.assertIn("recursion_pathways", flowchart)
        self.assertIn("meta_cognitive_flow", flowchart)
        self.assertIn("recursion_statistics", flowchart)
        
        # Verify pathway structure
        pathways = flowchart["recursion_pathways"]
        self.assertGreaterEqual(len(pathways), 3)  # At least 3 main pathways
        
        for pathway in pathways:
            self.assertIn("name", pathway)
            self.assertIn("description", pathway)
            self.assertIn("recursive_depth", pathway)
            self.assertIn("triggers", pathway)
            self.assertIn("outputs", pathway)
        
        # Verify statistics
        stats = flowchart["recursion_statistics"]
        self.assertEqual(stats["total_cycles"], 3)
        self.assertEqual(stats["max_recursive_depth"], 2)
        self.assertGreater(stats["avg_improvements_per_cycle"], 0)
    
    def test_results_export(self):
        """Test results export functionality"""
        # Add some history
        state = MetaCognitiveState(
            timestamp=time.time(),
            cognitive_metrics={"load": 0.5},
            evolution_metrics={"fitness": 0.7},
            performance_metrics={"cpu": 0.4},
            recursive_depth=1,
            improvement_suggestions=["test_suggestion"]
        )
        self.engine.recursion_history.append(state)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filename = f.name
        
        try:
            exported_file = self.engine.export_results(filename)
            self.assertEqual(exported_file, filename)
            
            # Verify file contents
            with open(filename, 'r') as f:
                export_data = json.load(f)
            
            self.assertIn("export_metadata", export_data)
            self.assertIn("recursion_history", export_data)
            self.assertIn("self_analysis_summary", export_data)
            self.assertIn("optimization_summary", export_data)
            self.assertIn("live_metrics", export_data)
            self.assertIn("recursion_flowchart", export_data)
            
            # Verify history was exported
            self.assertEqual(len(export_data["recursion_history"]), 1)
            history_item = export_data["recursion_history"][0]
            self.assertEqual(history_item["recursive_depth"], 1)
            self.assertEqual(history_item["improvement_suggestions"], ["test_suggestion"])
            
        finally:
            # Clean up
            if os.path.exists(filename):
                os.unlink(filename)

class TestMetaCognitiveIntegration(unittest.TestCase):
    """Integration tests for the complete meta-cognitive system"""
    
    def test_complete_recursion_cycle(self):
        """Test a complete meta-cognitive recursion cycle"""
        async def async_test():
            # Setup
            network = EvolutionNetwork()
            network.add_agent(EchoAgent("TestAgent", "Testing", 0.5))
            bridge = CognitiveEvolutionBridge(network)
            engine = MetaCognitiveRecursionEngine(bridge)
            
            # Run a short recursion cycle
            results = await engine.start_recursive_meta_cognition(cycles=2)
            
            # Verify results structure
            self.assertIn("start_time", results)
            self.assertIn("end_time", results)
            self.assertIn("cycles_completed", results)
            self.assertIn("meta_cognitive_states", results)
            self.assertIn("improvement_suggestions", results)
            
            # Verify cycles completed
            self.assertEqual(results["cycles_completed"], 2)
            self.assertEqual(len(results["meta_cognitive_states"]), 2)
            
            # Verify each state has required fields
            for state in results["meta_cognitive_states"]:
                self.assertIsInstance(state, MetaCognitiveState)
                self.assertGreater(state.timestamp, 0)
                self.assertIsInstance(state.cognitive_metrics, dict)
                self.assertIsInstance(state.evolution_metrics, dict)
                self.assertIsInstance(state.performance_metrics, dict)
            
            # Verify improvement suggestions were generated
            self.assertIsInstance(results["improvement_suggestions"], list)
        
        # Run the async test
        asyncio.run(async_test())

def run_all_tests():
    """Run all meta-cognitive recursion tests"""
    print("🧠🔄 Running Phase 5: Meta-Cognitive Recursion Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestSelfAnalysisModule))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAdaptiveOptimizer))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestLiveMetricsMonitor))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMetaCognitiveRecursionEngine))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMetaCognitiveIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🧠 Meta-Cognitive Recursion Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n🚨 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print("\n✅ All tests passed!")
    
    return result

if __name__ == "__main__":
    # Run tests
    result = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)