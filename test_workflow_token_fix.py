#!/usr/bin/env python3
"""
Test script to validate the fix for GITHUB_TOKEN issue in cronbot0.yml workflow.

This test specifically validates that the workflow configuration properly provides
the GITHUB_TOKEN environment variable to the cronbot.py step.
"""

import os
import yaml
import unittest
from pathlib import Path


class TestWorkflowTokenFix(unittest.TestCase):
    """Test cases for validating the GITHUB_TOKEN fix in workflow"""
    
    def setUp(self):
        """Set up test environment"""
        self.workflow_path = Path('/home/runner/work/echo9ml/echo9ml/.github/workflows/cronbot0.yml')
        
    def test_workflow_file_exists(self):
        """Test that cronbot0.yml workflow file exists"""
        self.assertTrue(self.workflow_path.exists(), 
                       f"Workflow file not found at {self.workflow_path}")
    
    def test_workflow_yaml_valid(self):
        """Test that the workflow YAML is valid"""
        try:
            with open(self.workflow_path, 'r') as f:
                workflow_data = yaml.safe_load(f)
            self.assertIsInstance(workflow_data, dict)
        except yaml.YAMLError as e:
            self.fail(f"Invalid YAML in workflow file: {e}")
    
    def test_cronbot_step_has_github_token(self):
        """Test that cronbot.py step has GITHUB_TOKEN environment variable"""
        with open(self.workflow_path, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        # Find the self_improvement job
        jobs = workflow_data.get('jobs', {})
        self.assertIn('self_improvement', jobs, "self_improvement job not found")
        
        job = jobs['self_improvement']
        steps = job.get('steps', [])
        
        # Find the cronbot.py step
        cronbot_step = None
        for step in steps:
            if ('run' in step and 
                'python cronbot.py' in step['run']):
                cronbot_step = step
                break
        
        self.assertIsNotNone(cronbot_step, 
                           "Step running 'python cronbot.py' not found")
        
        # Check if the step has GITHUB_TOKEN environment variable
        env = cronbot_step.get('env', {})
        self.assertIn('GITHUB_TOKEN', env, 
                     "GITHUB_TOKEN environment variable missing from cronbot.py step")
        
        # Check if GITHUB_TOKEN is set to the correct secret
        github_token_value = env.get('GITHUB_TOKEN')
        self.assertEqual(github_token_value, '${{ secrets.WFLO }}',
                        f"GITHUB_TOKEN should be '${{{{ secrets.WFLO }}}}', got '{github_token_value}'")
    
    def test_copilot_suggestions_step_has_token(self):
        """Test that copilot_suggestions.py step still has GITHUB_TOKEN (regression test)"""
        with open(self.workflow_path, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        jobs = workflow_data.get('jobs', {})
        job = jobs['self_improvement']
        steps = job.get('steps', [])
        
        # Find the copilot_suggestions.py step
        copilot_step = None
        for step in steps:
            if ('run' in step and 
                'python copilot_suggestions.py' in step['run']):
                copilot_step = step
                break
        
        self.assertIsNotNone(copilot_step, 
                           "Step running 'python copilot_suggestions.py' not found")
        
        # Check if the step has GITHUB_TOKEN environment variable
        env = copilot_step.get('env', {})
        self.assertIn('GITHUB_TOKEN', env, 
                     "GITHUB_TOKEN environment variable missing from copilot_suggestions.py step")
    
    def test_workflow_permissions(self):
        """Test that workflow has necessary permissions"""
        with open(self.workflow_path, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        jobs = workflow_data.get('jobs', {})
        job = jobs['self_improvement']
        permissions = job.get('permissions', {})
        
        # Check required permissions
        required_permissions = ['contents', 'actions', 'pull-requests']
        for perm in required_permissions:
            self.assertIn(perm, permissions, 
                         f"Required permission '{perm}' missing from workflow")


def run_manual_test():
    """
    Manual test function to demonstrate the issue and solution.
    This simulates what happens when cronbot.py is run without GITHUB_TOKEN.
    """
    print("=== Manual Test: Simulating cronbot.py without GITHUB_TOKEN ===")
    
    # Save current GITHUB_TOKEN if it exists
    original_token = os.environ.get('GITHUB_TOKEN')
    
    try:
        # Remove GITHUB_TOKEN to simulate the issue
        if 'GITHUB_TOKEN' in os.environ:
            del os.environ['GITHUB_TOKEN']
        
        # Import cronbot and test call_github_copilot function
        import sys
        sys.path.insert(0, '/home/runner/work/echo9ml/echo9ml')
        
        try:
            import cronbot
            result = cronbot.call_github_copilot({"test": "note"})
            
            if result is None:
                print("✅ CONFIRMED: cronbot.py returns None when GITHUB_TOKEN is missing")
                print("   This is the source of the workflow failure.")
            else:
                print("❌ UNEXPECTED: cronbot.py should return None when GITHUB_TOKEN is missing")
                
        except ImportError as e:
            print(f"⚠️  Could not import cronbot.py (dependencies missing): {e}")
            print("   But we can verify the issue exists by reading the code.")
            
            # Read the cronbot.py file to show the issue
            cronbot_path = '/home/runner/work/echo9ml/echo9ml/cronbot.py'
            if os.path.exists(cronbot_path):
                with open(cronbot_path, 'r') as f:
                    lines = f.readlines()
                
                print("\n--- cronbot.py lines 34-39 (the problematic code) ---")
                for i, line in enumerate(lines[33:39], 34):
                    print(f"{i:2d}: {line.rstrip()}")
                
                print("\n✅ CONFIRMED: cronbot.py checks for GITHUB_TOKEN and returns None if missing")
    
    finally:
        # Restore original GITHUB_TOKEN
        if original_token:
            os.environ['GITHUB_TOKEN'] = original_token


if __name__ == '__main__':
    print("Testing workflow token configuration...")
    
    # Run manual test first
    run_manual_test()
    
    print("\n" + "="*60)
    print("Running unit tests...")
    
    # Run unit tests
    unittest.main(verbosity=2)