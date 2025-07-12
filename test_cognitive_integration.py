#!/usr/bin/env python3
"""
Comprehensive test for the Cognitive Integration Orchestrator
Tests all major functionality including phase management, integration, and workflow automation
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cognitive_integration_orchestrator import CognitiveIntegrationOrchestrator, CognitivePhase

class TestCognitiveIntegrationOrchestrator(unittest.TestCase):
    """Test suite for cognitive integration orchestrator"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create mock cognitive component files
        self.create_mock_components()
        
        # Initialize orchestrator
        self.orchestrator = CognitiveIntegrationOrchestrator()
        
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_mock_components(self):
        """Create mock cognitive component files"""
        components = [
            "distributed_cognitive_grammar.py",
            "cognitive_architecture.py", 
            "ggml_tensor_kernel.py",
            "symbolic_reasoning.py",
            "echoself_introspection.py",
            "deep_tree_echo.py"
        ]
        
        for component in components:
            component_file = Path(component)
            component_file.write_text(f"# Mock {component}\nprint('Mock component loaded')\n")
    
    def test_cognitive_phases_initialization(self):
        """Test that all 6 cognitive phases are properly initialized"""
        phases = self.orchestrator.cognitive_phases
        
        self.assertEqual(len(phases), 6)
        
        # Test phase 1
        phase1 = phases[1]
        self.assertEqual(phase1.id, 1)
        self.assertEqual(phase1.name, "Cognitive Primitives & Foundational Hypergraph Encoding")
        self.assertIn("atomic vocabulary", phase1.description)
        self.assertEqual(len(phase1.objectives), 4)
        self.assertEqual(len(phase1.dependencies), 0)
        
        # Test phase 6 (should depend on all previous phases)
        phase6 = phases[6]
        self.assertEqual(phase6.id, 6)
        self.assertEqual(phase6.name, "Rigorous Testing, Documentation, and Cognitive Unification")
        self.assertEqual(phase6.dependencies, [1, 2, 3, 4, 5])
    
    def test_component_detection(self):
        """Test detection of cognitive components"""
        component_status = self.orchestrator._check_cognitive_components()
        
        # All mock components should be detected
        expected_components = [
            "distributed_cognitive_grammar",
            "cognitive_architecture", 
            "ggml_tensor_kernel",
            "symbolic_reasoning",
            "echoself_introspection"
        ]
        
        for component in expected_components:
            self.assertTrue(component_status[component], f"{component} should be detected")
    
    @patch('requests.get')
    def test_existing_issues_check(self, mock_get):
        """Test checking for existing GitHub issues"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"title": "Phase 1: Cognitive Primitives", "number": 1},
            {"title": "Phase 2: ECAN Attention", "number": 2}
        ]
        mock_get.return_value = mock_response
        
        # Set GitHub token for test
        self.orchestrator.github_token = "test_token"
        
        existing = self.orchestrator._get_existing_issues("Phase 1: Cognitive Primitives")
        self.assertEqual(len(existing), 1)
        self.assertEqual(existing[0]["number"], 1)
    
    @patch('requests.post')
    def test_github_issue_creation(self, mock_post):
        """Test GitHub issue creation"""
        # Mock successful issue creation
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"number": 123}
        mock_post.return_value = mock_response
        
        # Set GitHub token for test
        self.orchestrator.github_token = "test_token"
        
        phase = self.orchestrator.cognitive_phases[1]
        result = self.orchestrator._create_github_issue(phase)
        
        self.assertTrue(result)
        mock_post.assert_called_once()
        
        # Check that the API was called with correct data
        call_args = mock_post.call_args
        self.assertIn("Phase 1:", call_args[1]["json"]["title"])
        self.assertIn("cognitive-phase", call_args[1]["json"]["labels"])
    
    def test_cognitive_integration_execution(self):
        """Test cognitive integration execution"""
        # This should work without external dependencies
        result = self.orchestrator.execute_cognitive_integration("1")
        self.assertTrue(result)
        
        # Check that workspace files were created
        self.assertTrue(Path("cognitive_workspace").exists())
        self.assertTrue(Path("cognitive_logs").exists())
        
        # Check for integration config file
        config_file = Path("cognitive_workspace/distributed_grammar_config.json")
        self.assertTrue(config_file.exists())
        
        with open(config_file) as f:
            config = json.load(f)
            self.assertEqual(config["agent_network"], "echo9ml_reservoir")
            self.assertEqual(config["attention_allocation"], "ECAN_style")
    
    def test_reservoir_nodes_update(self):
        """Test reservoir computing nodes update"""
        result = self.orchestrator.update_reservoir_nodes()
        self.assertTrue(result)
        
        # Check reservoir state file
        reservoir_file = Path("cognitive_workspace/reservoir_state.json")
        self.assertTrue(reservoir_file.exists())
        
        with open(reservoir_file) as f:
            state = json.load(f)
            self.assertIn("timestamp", state)
            self.assertEqual(state["integration_status"], "active")
            self.assertEqual(state["deep_tree_echo"], "available")
    
    def test_status_report_generation(self):
        """Test status report generation"""
        report = self.orchestrator.generate_status_report()
        
        self.assertIn("Distributed Agentic Cognitive Grammar - Status Report", report)
        self.assertIn("Phase 1: Cognitive Primitives", report)
        self.assertIn("Phase 6: Rigorous Testing", report)
        self.assertIn("Component Availability", report)
        self.assertIn("✅ Available", report)  # Should show available components
    
    def test_phase_filtering(self):
        """Test phase filtering functionality"""
        # Test single phase
        with patch.object(self.orchestrator, '_create_github_issue') as mock_create:
            mock_create.return_value = True
            self.orchestrator.github_token = "test_token"
            
            result = self.orchestrator.create_phase_issues("1")
            self.assertTrue(result)
            mock_create.assert_called_once()
            
            # Should have called with phase 1
            called_phase = mock_create.call_args[0][0]
            self.assertEqual(called_phase.id, 1)
    
    def test_workspace_creation(self):
        """Test workspace directory creation"""
        self.assertTrue(self.orchestrator.workspace_dir.exists())
        self.assertTrue(self.orchestrator.logs_dir.exists())
        self.assertEqual(self.orchestrator.workspace_dir.name, "cognitive_workspace")
        self.assertEqual(self.orchestrator.logs_dir.name, "cognitive_logs")
    
    def test_integration_logging(self):
        """Test integration logging functionality"""
        # Execute integration to generate logs
        self.orchestrator.execute_cognitive_integration("1")
        
        # Check that logs were created
        log_files = list(Path("cognitive_logs").glob("integration_*.json"))
        self.assertGreater(len(log_files), 0)
        
        # Check log content
        with open(log_files[0]) as f:
            log_data = json.load(f)
            self.assertIn("timestamp", log_data)
            self.assertEqual(log_data["phase_filter"], "1")
            self.assertEqual(log_data["action"], "cognitive_integration")

class TestCognitivePhaseStructure(unittest.TestCase):
    """Test the CognitivePhase data structure"""
    
    def test_cognitive_phase_creation(self):
        """Test CognitivePhase data class"""
        phase = CognitivePhase(
            id=1,
            name="Test Phase",
            description="Test description",
            objectives=["Objective 1", "Objective 2"],
            sub_steps=["Step 1", "Step 2"],
            verification_criteria=["Criteria 1"],
            dependencies=[],
            status="pending"
        )
        
        self.assertEqual(phase.id, 1)
        self.assertEqual(phase.name, "Test Phase")
        self.assertEqual(len(phase.objectives), 2)
        self.assertEqual(phase.status, "pending")

class TestIntegrationWorkflow(unittest.TestCase):
    """Test the complete integration workflow"""
    
    def setUp(self):
        """Set up workflow test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create all required files for full workflow test
        self.create_full_environment()
        
    def tearDown(self):
        """Clean up workflow test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_full_environment(self):
        """Create full test environment with all files"""
        files = {
            "distributed_cognitive_grammar.py": "# Distributed cognitive grammar",
            "cognitive_architecture.py": "# Cognitive architecture", 
            "ggml_tensor_kernel.py": "# GGML tensor kernel",
            "symbolic_reasoning.py": "# Symbolic reasoning",
            "echoself_introspection.py": "# Echoself introspection",
            "deep_tree_echo.py": "# Deep tree echo"
        }
        
        for filename, content in files.items():
            Path(filename).write_text(content)
    
    def test_full_workflow_execution(self):
        """Test complete workflow from start to finish"""
        orchestrator = CognitiveIntegrationOrchestrator()
        
        # 1. Test component detection
        components = orchestrator._check_cognitive_components()
        self.assertTrue(all(components.values()))
        
        # 2. Test integration execution
        result = orchestrator.execute_cognitive_integration("all")
        self.assertTrue(result)
        
        # 3. Test reservoir update
        result = orchestrator.update_reservoir_nodes()
        self.assertTrue(result)
        
        # 4. Test status report
        report = orchestrator.generate_status_report()
        self.assertIn("✅ Available", report)
        
        # 5. Verify workspace structure
        self.assertTrue(Path("cognitive_workspace").exists())
        self.assertTrue(Path("cognitive_logs").exists())
        self.assertTrue(Path("cognitive_workspace/distributed_grammar_config.json").exists())
        self.assertTrue(Path("cognitive_workspace/reservoir_state.json").exists())

def run_comprehensive_test():
    """Run the comprehensive test suite"""
    print("🧠 Running Comprehensive Cognitive Integration Tests...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestCognitiveIntegrationOrchestrator))
    test_suite.addTest(unittest.makeSuite(TestCognitivePhaseStructure))
    test_suite.addTest(unittest.makeSuite(TestIntegrationWorkflow))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print("🧠 Cognitive Integration Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n✅ All tests passed!" if success else "❌ Some tests failed!")
    
    return success

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)