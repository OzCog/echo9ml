"""
Round-Trip Translation Tests for Ko6ml ↔ AtomSpace Adapter

This module provides comprehensive tests for bidirectional translation between
ko6ml cognitive primitives and AtomSpace hypergraph patterns. The tests validate
the Phase 1 requirements for the Distributed Agentic Cognitive Grammar Network.

Key Test Categories:
- Basic primitive translation tests
- Round-trip validation tests
- Tensor fragment architecture tests
- Complex pattern translation tests
- Performance and consistency validation
"""

import unittest
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import logging

# Import the adapter and related components
try:
    from ko6ml_atomspace_adapter import (
        Ko6mlAtomSpaceAdapter, Ko6mlPrimitive, Ko6mlPrimitiveType, 
        AtomSpaceFragment, create_ko6ml_adapter, create_test_primitives
    )
    ADAPTER_AVAILABLE = True
except ImportError:
    ADAPTER_AVAILABLE = False
    logging.warning("Ko6ml adapter not available for testing")

logger = logging.getLogger(__name__)

class TestKo6mlAtomSpaceTranslation(unittest.TestCase):
    """Comprehensive tests for ko6ml ↔ AtomSpace translation"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not ADAPTER_AVAILABLE:
            self.skipTest("Ko6ml adapter not available")
        
        self.adapter = create_ko6ml_adapter()
        self.test_primitives = create_test_primitives()
        self.translation_results = []
    
    def tearDown(self):
        """Clean up after tests"""
        # Log test results for analysis
        if self.translation_results:
            logger.info(f"Test completed with {len(self.translation_results)} translations")
    
    def test_adapter_initialization(self):
        """Test adapter initialization and basic functionality"""
        self.assertIsInstance(self.adapter, Ko6mlAtomSpaceAdapter)
        self.assertTrue(len(self.adapter.primitive_mappings) > 0)
        self.assertTrue(len(self.adapter.atomspace_mappings) > 0)
        self.assertTrue(len(self.adapter.tensor_shapes) > 0)
        
        # Test tensor shape documentation
        tensor_docs = self.adapter.get_tensor_shape_documentation()
        self.assertIn("tensor_shapes", tensor_docs)
        self.assertIn("prime_factorization_rationale", tensor_docs)
        self.assertIn("semantic_mappings", tensor_docs)
    
    def test_ko6ml_to_atomspace_basic(self):
        """Test basic ko6ml to AtomSpace translation"""
        for primitive in self.test_primitives:
            with self.subTest(primitive_id=primitive.primitive_id):
                fragment = self.adapter.ko6ml_to_atomspace(primitive)
                
                # Validate fragment structure
                self.assertIsInstance(fragment, AtomSpaceFragment)
                self.assertTrue(fragment.fragment_id)
                self.assertEqual(fragment.source_primitive, primitive.primitive_id)
                
                # Validate atoms and links
                self.assertIsInstance(fragment.atoms, list)
                self.assertIsInstance(fragment.links, list)
                
                # At least one atom or link should be present
                self.assertTrue(len(fragment.atoms) > 0 or len(fragment.links) > 0)
                
                # Validate atom structure
                for atom in fragment.atoms:
                    self.assertIn("id", atom)
                    self.assertIn("type", atom)
                    self.assertIn("name", atom)
                    if "truth_value" in atom:
                        tv = atom["truth_value"]
                        self.assertTrue(0.0 <= tv[0] <= 1.0)  # strength
                        self.assertTrue(0.0 <= tv[1] <= 1.0)  # confidence
                
                # Validate link structure
                for link in fragment.links:
                    self.assertIn("id", link)
                    self.assertIn("type", link)
                    self.assertIn("outgoing", link)
                    self.assertIsInstance(link["outgoing"], list)
    
    def test_atomspace_to_ko6ml_basic(self):
        """Test basic AtomSpace to ko6ml translation"""
        # First convert primitives to atomspace
        fragments = []
        for primitive in self.test_primitives:
            fragment = self.adapter.ko6ml_to_atomspace(primitive)
            fragments.append(fragment)
        
        # Then convert back to ko6ml
        for fragment in fragments:
            with self.subTest(fragment_id=fragment.fragment_id):
                recovered_primitives = self.adapter.atomspace_to_ko6ml(fragment)
                
                # Should recover at least one primitive
                self.assertTrue(len(recovered_primitives) > 0)
                
                # Validate recovered primitive structure
                for primitive in recovered_primitives:
                    self.assertIsInstance(primitive, Ko6mlPrimitive)
                    self.assertTrue(primitive.primitive_id)
                    self.assertIn(primitive.primitive_type, Ko6mlPrimitiveType)
                    self.assertIsInstance(primitive.content, dict)
    
    def test_round_trip_translation_all_primitives(self):
        """Test round-trip translation for all primitive types"""
        success_count = 0
        total_count = len(self.test_primitives)
        
        for primitive in self.test_primitives:
            with self.subTest(primitive_id=primitive.primitive_id):
                result = self.adapter.validate_round_trip(primitive)
                self.translation_results.append(result)
                
                # Check basic result structure
                self.assertIn("success", result)
                self.assertIn("original", result)
                
                if result["success"]:
                    success_count += 1
                    self.assertIn("similarity", result)
                    self.assertIn("recovered", result)
                    self.assertIn("atomspace_fragment", result)
                    
                    # Validate similarity score
                    similarity = result["similarity"]
                    self.assertTrue(0.0 <= similarity <= 1.0)
                    self.assertGreaterEqual(similarity, 0.6)  # 60% threshold for structural differences
                    
                    # Validate recovered primitive
                    recovered = result["recovered"]
                    self.assertIsInstance(recovered, Ko6mlPrimitive)
                    self.assertEqual(recovered.primitive_type, primitive.primitive_type)
                
                else:
                    self.assertIn("error", result)
                    logger.warning(f"Round-trip failed for {primitive.primitive_id}: {result['error']}")
        
        # Overall success rate should be reasonable for structural translation
        success_rate = success_count / total_count
        self.assertGreaterEqual(success_rate, 0.6, f"Success rate too low: {success_rate:.2f}")  # 60% threshold
        
        logger.info(f"Round-trip success rate: {success_rate:.2f} ({success_count}/{total_count})")
    
    def test_agent_state_round_trip(self):
        """Test specific round-trip for agent state primitives"""
        agent_state = Ko6mlPrimitive(
            primitive_id="test_agent_detailed",
            primitive_type=Ko6mlPrimitiveType.AGENT_STATE,
            content={
                "agent_id": "test_cognitive_agent",
                "state": {
                    "active": True,
                    "attention_level": 0.75,
                    "processing_load": 0.45,
                    "memory_utilization": 0.6,
                    "reasoning_mode": "analytical"
                }
            },
            truth_value=(0.85, 0.9),
            tensor_signature=(13, 11, 7, 5, 3)
        )
        
        result = self.adapter.validate_round_trip(agent_state)
        self.assertTrue(result["success"], f"Agent state round-trip failed: {result.get('error', 'Unknown')}")
        self.assertGreaterEqual(result["similarity"], 0.6)
    
    def test_memory_fragment_round_trip(self):
        """Test specific round-trip for memory fragment primitives"""
        memory_fragment = Ko6mlPrimitive(
            primitive_id="test_memory_detailed",
            primitive_type=Ko6mlPrimitiveType.MEMORY_FRAGMENT,
            content={
                "content": "Learned about hypergraph encoding for cognitive primitives",
                "type": "semantic",
                "salience": 0.8,
                "tags": ["learning", "hypergraph", "encoding"],
                "temporal_context": "recent"
            },
            truth_value=(0.8, 0.85),
            tensor_signature=(101, 8, 5, 7, 3)
        )
        
        result = self.adapter.validate_round_trip(memory_fragment)
        self.assertTrue(result["success"], f"Memory fragment round-trip failed: {result.get('error', 'Unknown')}")
        self.assertGreaterEqual(result["similarity"], 0.6)
    
    def test_reasoning_pattern_round_trip(self):
        """Test specific round-trip for reasoning pattern primitives"""
        reasoning_pattern = Ko6mlPrimitive(
            primitive_id="test_reasoning_detailed",
            primitive_type=Ko6mlPrimitiveType.REASONING_PATTERN,
            content={
                "pattern_type": "hypothetical_syllogism",
                "premises": [
                    "If hypergraphs encode knowledge, then reasoning can be distributed",
                    "If reasoning can be distributed, then agents can collaborate"
                ],
                "conclusion": "If hypergraphs encode knowledge, then agents can collaborate",
                "confidence": 0.9,
                "inference_steps": 2
            },
            truth_value=(0.9, 0.8),
            tensor_signature=(23, 5, 7, 3, 2)
        )
        
        result = self.adapter.validate_round_trip(reasoning_pattern)
        self.assertTrue(result["success"], f"Reasoning pattern round-trip failed: {result.get('error', 'Unknown')}")
        self.assertGreaterEqual(result["similarity"], 0.6)
    
    def test_attention_allocation_round_trip(self):
        """Test specific round-trip for attention allocation primitives"""
        attention_allocation = Ko6mlPrimitive(
            primitive_id="test_attention_detailed",
            primitive_type=Ko6mlPrimitiveType.ATTENTION_ALLOCATION,
            content={
                "allocation": {
                    "hypergraph_processing": 35.0,
                    "primitive_translation": 25.0,
                    "round_trip_validation": 20.0,
                    "meta_cognition": 15.0,
                    "background_processing": 5.0
                },
                "total_attention": 100.0,
                "allocation_strategy": "priority_based",
                "temporal_decay": 0.95
            },
            truth_value=(0.9, 0.85),
            tensor_signature=(17, 17, 11, 7, 2)
        )
        
        result = self.adapter.validate_round_trip(attention_allocation)
        self.assertTrue(result["success"], f"Attention allocation round-trip failed: {result.get('error', 'Unknown')}")
        self.assertGreaterEqual(result["similarity"], 0.6)
    
    def test_persona_trait_round_trip(self):
        """Test specific round-trip for persona trait primitives"""
        persona_trait = Ko6mlPrimitive(
            primitive_id="test_trait_detailed",
            primitive_type=Ko6mlPrimitiveType.PERSONA_TRAIT,
            content={
                "trait_type": "adaptability",
                "value": 0.85,
                "persona_id": "deep_tree_echo",
                "evolution_history": [0.7, 0.75, 0.8, 0.85],
                "context_sensitivity": 0.9
            },
            truth_value=(0.85, 0.9),
            tensor_signature=(3, 7, 13, 5, 2)
        )
        
        result = self.adapter.validate_round_trip(persona_trait)
        self.assertTrue(result["success"], f"Persona trait round-trip failed: {result.get('error', 'Unknown')}")
        self.assertGreaterEqual(result["similarity"], 0.6)
    
    def test_tensor_fragment_architecture(self):
        """Test tensor fragment architecture and encoding"""
        tensor_fragment = Ko6mlPrimitive(
            primitive_id="test_tensor_fragment",
            primitive_type=Ko6mlPrimitiveType.TENSOR_FRAGMENT,
            content={
                "shape": (3, 7, 13, 5, 2),
                "semantic_dimensions": ["persona_id", "trait_id", "time", "context", "valence"],
                "data_type": "float32",
                "compression": "none",
                "fragment_size": 2730  # 3*7*13*5*2
            },
            truth_value=(0.9, 0.9),
            tensor_signature=(3, 7, 13, 5, 2)
        )
        
        # Test forward translation
        fragment = self.adapter.ko6ml_to_atomspace(tensor_fragment)
        self.assertIsInstance(fragment, AtomSpaceFragment)
        self.assertTrue(len(fragment.atoms) > 0)
        
        # Check for tensor-specific atoms
        tensor_atoms = [atom for atom in fragment.atoms if "tensor" in atom.get("name", "").lower()]
        self.assertTrue(len(tensor_atoms) > 0)
        
        # Test round-trip
        result = self.adapter.validate_round_trip(tensor_fragment)
        self.assertTrue(result["success"], f"Tensor fragment round-trip failed: {result.get('error', 'Unknown')}")
        self.assertGreaterEqual(result["similarity"], 0.6)
    
    def test_complex_hypergraph_pattern(self):
        """Test complex hypergraph pattern with multiple interconnected primitives"""
        # Create interconnected primitives
        agent = Ko6mlPrimitive(
            primitive_id="complex_agent",
            primitive_type=Ko6mlPrimitiveType.AGENT_STATE,
            content={"agent_id": "complex_agent", "state": {"active": True}},
            truth_value=(0.9, 0.8)
        )
        
        memory = Ko6mlPrimitive(
            primitive_id="complex_memory",
            primitive_type=Ko6mlPrimitiveType.MEMORY_FRAGMENT,
            content={"content": "Agent memory", "type": "working", "agent_ref": "complex_agent"},
            truth_value=(0.8, 0.9)
        )
        
        reasoning = Ko6mlPrimitive(
            primitive_id="complex_reasoning",
            primitive_type=Ko6mlPrimitiveType.REASONING_PATTERN,
            content={
                "pattern_type": "agent_memory_reasoning",
                "agent_ref": "complex_agent",
                "memory_ref": "complex_memory"
            },
            truth_value=(0.85, 0.8)
        )
        
        complex_primitives = [agent, memory, reasoning]
        
        # Test each primitive individually
        for primitive in complex_primitives:
            result = self.adapter.validate_round_trip(primitive)
            self.assertTrue(result["success"], 
                          f"Complex pattern primitive {primitive.primitive_id} failed: {result.get('error', 'Unknown')}")
            self.assertGreaterEqual(result["similarity"], 0.6)
    
    def test_tensor_shape_prime_factorization(self):
        """Test tensor shape prime factorization documentation"""
        tensor_docs = self.adapter.get_tensor_shape_documentation()
        
        # Validate tensor shapes
        expected_shapes = ["persona", "memory", "attention", "reasoning", "agent_state"]
        for shape_name in expected_shapes:
            self.assertIn(shape_name, tensor_docs["tensor_shapes"])
            shape = tensor_docs["tensor_shapes"][shape_name]
            
            # Validate shape is tuple of integers
            self.assertIsInstance(shape, tuple)
            self.assertTrue(all(isinstance(dim, int) for dim in shape))
            
            # Validate prime factorization explanation exists
            self.assertIn(shape_name, tensor_docs["prime_factorization_rationale"])
            
            # Validate semantic mapping exists
            self.assertIn(shape_name, tensor_docs["semantic_mappings"])
            semantic_dims = tensor_docs["semantic_mappings"][shape_name]
            self.assertEqual(len(semantic_dims), len(shape))
    
    def test_scheme_expression_generation(self):
        """Test Scheme expression generation for primitives"""
        for primitive in self.test_primitives:
            scheme_expr = primitive.to_scheme_expr()
            
            # Basic structure validation
            self.assertTrue(scheme_expr.startswith("("))
            self.assertTrue(scheme_expr.endswith(")"))
            self.assertIn(primitive.primitive_type.value, scheme_expr)
            self.assertIn(primitive.primitive_id, scheme_expr)
            
            # Truth value should be included if present
            if primitive.truth_value:
                self.assertIn("(tv", scheme_expr)
                self.assertIn(str(primitive.truth_value[0]), scheme_expr)
                self.assertIn(str(primitive.truth_value[1]), scheme_expr)
            
            # Tensor signature should be included if present
            if primitive.tensor_signature:
                self.assertIn("(tensor-shape", scheme_expr)
    
    def test_translation_consistency(self):
        """Test consistency of translation across multiple runs"""
        test_primitive = self.test_primitives[0]  # Use first primitive
        
        results = []
        for i in range(5):  # Run 5 times
            result = self.adapter.validate_round_trip(test_primitive)
            results.append(result)
        
        # All runs should succeed (if the first one does)
        first_success = results[0]["success"]
        for result in results:
            self.assertEqual(result["success"], first_success)
            
            if first_success:
                # Similarity scores should be consistent (within tolerance)
                similarity_diff = abs(result["similarity"] - results[0]["similarity"])
                self.assertLess(similarity_diff, 0.1, "Similarity scores should be consistent")
    
    def test_translation_performance(self):
        """Test translation performance for acceptable response times"""
        # Measure forward translation time
        start_time = time.time()
        for primitive in self.test_primitives:
            self.adapter.ko6ml_to_atomspace(primitive)
        forward_time = time.time() - start_time
        
        # Forward translation should be fast
        avg_forward_time = forward_time / len(self.test_primitives)
        self.assertLess(avg_forward_time, 0.1, f"Forward translation too slow: {avg_forward_time:.3f}s per primitive")
        
        # Measure round-trip time
        start_time = time.time()
        for primitive in self.test_primitives[:3]:  # Test subset for performance
            self.adapter.validate_round_trip(primitive)
        roundtrip_time = time.time() - start_time
        
        avg_roundtrip_time = roundtrip_time / 3
        self.assertLess(avg_roundtrip_time, 0.5, f"Round-trip translation too slow: {avg_roundtrip_time:.3f}s per primitive")
    
    def test_translation_history_tracking(self):
        """Test translation history tracking functionality"""
        initial_history_length = len(self.adapter.translation_history)
        
        # Perform some translations
        for primitive in self.test_primitives[:3]:
            self.adapter.validate_round_trip(primitive)
        
        # History should have grown
        final_history_length = len(self.adapter.translation_history)
        self.assertGreater(final_history_length, initial_history_length)
        
        # Validate history entries
        recent_entries = self.adapter.translation_history[-6:]  # Should be 6 entries (2 per round-trip)
        for entry in recent_entries:
            self.assertIn("direction", entry)
            self.assertIn("timestamp", entry)
            self.assertIn(entry["direction"], ["ko6ml_to_atomspace", "atomspace_to_ko6ml"])

class TestHypergraphFragmentVisualization(unittest.TestCase):
    """Tests for hypergraph fragment visualization and documentation"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not ADAPTER_AVAILABLE:
            self.skipTest("Ko6ml adapter not available")
        
        self.adapter = create_ko6ml_adapter()
    
    def test_fragment_serialization(self):
        """Test AtomSpace fragment serialization"""
        test_primitive = create_test_primitives()[0]
        fragment = self.adapter.ko6ml_to_atomspace(test_primitive)
        
        # Test serialization
        fragment_dict = fragment.to_dict()
        self.assertIsInstance(fragment_dict, dict)
        self.assertIn("fragment_id", fragment_dict)
        self.assertIn("atoms", fragment_dict)
        self.assertIn("links", fragment_dict)
        self.assertIn("source_primitive", fragment_dict)
        
        # Validate JSON serialization
        json_str = json.dumps(fragment_dict)
        self.assertIsInstance(json_str, str)
        
        # Validate deserialization
        deserialized = json.loads(json_str)
        self.assertEqual(deserialized["fragment_id"], fragment.fragment_id)
    
    def test_documentation_generation(self):
        """Test documentation generation for tensor signatures"""
        docs = self.adapter.get_tensor_shape_documentation()
        
        # Should contain required sections
        required_sections = ["tensor_shapes", "prime_factorization_rationale", "semantic_mappings"]
        for section in required_sections:
            self.assertIn(section, docs)
        
        # Should contain all expected tensor types
        expected_tensors = ["persona", "memory", "attention", "reasoning", "agent_state"]
        for tensor_type in expected_tensors:
            self.assertIn(tensor_type, docs["tensor_shapes"])
            self.assertIn(tensor_type, docs["prime_factorization_rationale"])
            self.assertIn(tensor_type, docs["semantic_mappings"])

def run_comprehensive_tests():
    """Run comprehensive test suite for ko6ml ↔ AtomSpace translation"""
    print("Ko6ml ↔ AtomSpace Translation Test Suite")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestKo6mlAtomSpaceTranslation))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestHypergraphFragmentVisualization))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_comprehensive_tests()