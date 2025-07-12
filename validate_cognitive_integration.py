#!/usr/bin/env python3
"""
Final validation script for the Distributed Agentic Cognitive Grammar GitHub Action
Validates workflow structure, component integration, and system readiness
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple

def validate_workflow_file() -> Tuple[bool, str]:
    """Validate the GitHub workflow YAML file"""
    try:
        workflow_path = Path(".github/workflows/cognitive-integration.yml")
        if not workflow_path.exists():
            return False, "Workflow file not found"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Handle YAML parser quirk where 'on:' becomes True key
        # Check for 'on' field or True field (which represents 'on:' in this YAML parser)
        has_on_field = 'on' in workflow or True in workflow
        if not has_on_field:
            return False, "Missing required field: on"
        
        # Check required fields (handle the True key case)
        required_fields = ['name', 'permissions', 'jobs']
        for field in required_fields:
            if field not in workflow:
                return False, f"Missing required field: {field}"
        
        # Check jobs
        jobs = workflow.get('jobs', {})
        expected_jobs = ['cognitive-integration', 'validate-integration']
        for job in expected_jobs:
            if job not in jobs:
                return False, f"Missing required job: {job}"
        
        # Check permissions
        permissions = workflow.get('permissions', {})
        required_permissions = ['contents', 'issues', 'actions']
        for perm in required_permissions:
            if perm not in permissions:
                return False, f"Missing required permission: {perm}"
        
        # Check that the on/True field contains expected triggers
        on_field = workflow.get('on', workflow.get(True, {}))
        if isinstance(on_field, dict):
            expected_triggers = ['workflow_dispatch', 'schedule', 'push']
            for trigger in expected_triggers:
                if trigger not in on_field:
                    return False, f"Missing workflow trigger: {trigger}"
        
        return True, "Workflow validation passed"
        
    except yaml.YAMLError as e:
        return False, f"YAML parsing error: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"

def validate_orchestrator_script() -> Tuple[bool, str]:
    """Validate the cognitive integration orchestrator script"""
    try:
        # Check file exists and is executable
        script_path = Path("cognitive_integration_orchestrator.py")
        if not script_path.exists():
            return False, "Orchestrator script not found"
        
        # Try importing to check syntax
        import cognitive_integration_orchestrator
        
        # Check main classes exist
        required_classes = ['CognitiveIntegrationOrchestrator', 'CognitivePhase']
        for cls_name in required_classes:
            if not hasattr(cognitive_integration_orchestrator, cls_name):
                return False, f"Missing required class: {cls_name}"
        
        # Test instantiation
        orchestrator = cognitive_integration_orchestrator.CognitiveIntegrationOrchestrator()
        
        # Check phase initialization
        if len(orchestrator.cognitive_phases) != 6:
            return False, f"Expected 6 phases, got {len(orchestrator.cognitive_phases)}"
        
        return True, "Orchestrator script validation passed"
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Script validation error: {e}"

def validate_cognitive_components() -> Tuple[bool, str]:
    """Validate that required cognitive components exist"""
    required_components = [
        "distributed_cognitive_grammar.py",
        "cognitive_architecture.py", 
        "ggml_tensor_kernel.py",
        "symbolic_reasoning.py",
        "echoself_introspection.py"
    ]
    
    missing_components = []
    for component in required_components:
        if not Path(component).exists():
            missing_components.append(component)
    
    if missing_components:
        return False, f"Missing components: {', '.join(missing_components)}"
    
    return True, "All cognitive components found"

def validate_issue_template() -> Tuple[bool, str]:
    """Validate the GitHub issue template"""
    try:
        template_path = Path(".github/ISSUE_TEMPLATE/cognitive-phase.md")
        if not template_path.exists():
            return False, "Issue template not found"
        
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Check for required sections
        required_sections = [
            "## Phase Description",
            "## Objectives", 
            "## Sub-Steps",
            "## Verification Criteria",
            "## Dependencies"
        ]
        
        for section in required_sections:
            if section not in content:
                return False, f"Missing required section: {section}"
        
        return True, "Issue template validation passed"
        
    except Exception as e:
        return False, f"Template validation error: {e}"

def validate_documentation() -> Tuple[bool, str]:
    """Validate documentation files"""
    required_docs = [
        "COGNITIVE_INTEGRATION_README.md"
    ]
    
    missing_docs = []
    for doc in required_docs:
        if not Path(doc).exists():
            missing_docs.append(doc)
    
    if missing_docs:
        return False, f"Missing documentation: {', '.join(missing_docs)}"
    
    return True, "Documentation validation passed"

def run_integration_test() -> Tuple[bool, str]:
    """Run a quick integration test"""
    try:
        import cognitive_integration_orchestrator
        
        orchestrator = cognitive_integration_orchestrator.CognitiveIntegrationOrchestrator()
        
        # Test component detection
        components = orchestrator._check_cognitive_components()
        available_count = sum(1 for available in components.values() if available)
        
        if available_count == 0:
            return False, "No cognitive components detected"
        
        # Test status report generation
        report = orchestrator.generate_status_report()
        if not report or len(report) < 100:
            return False, "Status report generation failed"
        
        return True, f"Integration test passed ({available_count}/5 components available)"
        
    except Exception as e:
        return False, f"Integration test error: {e}"

def validate_gitignore() -> Tuple[bool, str]:
    """Validate .gitignore has workspace exclusions"""
    try:
        gitignore_path = Path(".gitignore")
        if not gitignore_path.exists():
            return False, ".gitignore not found"
        
        with open(gitignore_path, 'r') as f:
            content = f.read()
        
        required_exclusions = ["cognitive_workspace/", "cognitive_logs/"]
        for exclusion in required_exclusions:
            if exclusion not in content:
                return False, f"Missing .gitignore exclusion: {exclusion}"
        
        return True, ".gitignore validation passed"
        
    except Exception as e:
        return False, f".gitignore validation error: {e}"

def main():
    """Run comprehensive validation"""
    print("🧠 Distributed Agentic Cognitive Grammar - Final Validation")
    print("=" * 65)
    
    validations = [
        ("GitHub Workflow", validate_workflow_file),
        ("Orchestrator Script", validate_orchestrator_script),
        ("Cognitive Components", validate_cognitive_components),
        ("Issue Template", validate_issue_template),
        ("Documentation", validate_documentation),
        ("GitIgnore Configuration", validate_gitignore),
        ("Integration Test", run_integration_test)
    ]
    
    all_passed = True
    results = []
    
    for name, validator in validations:
        try:
            passed, message = validator()
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {name:<25} | {message}")
            results.append((name, passed, message))
            
            if not passed:
                all_passed = False
                
        except Exception as e:
            print(f"❌ FAIL {name:<25} | Validation error: {e}")
            results.append((name, False, f"Validation error: {e}"))
            all_passed = False
    
    print("\n" + "=" * 65)
    print("🧠 VALIDATION SUMMARY")
    print("=" * 65)
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    print(f"Total validations: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    
    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ The Distributed Agentic Cognitive Grammar GitHub Action is ready for deployment!")
        print("\n📋 Next Steps:")
        print("1. The workflow will run automatically on:")
        print("   - Weekly schedule (Mondays at 6 AM UTC)")
        print("   - Push to main branch (cognitive component changes)")
        print("   - Manual workflow dispatch")
        print("2. Issues will be created automatically for each cognitive phase")
        print("3. Integration will run with existing echo9ml components")
        print("4. Reservoir computing nodes will be updated")
        print("\n🚀 Ready to begin recursive self-optimization spiral!")
    else:
        print(f"\n⚠️  {total_count - passed_count} VALIDATIONS FAILED")
        print("Please fix the issues above before deployment.")
        
        print("\nFailed validations:")
        for name, passed, message in results:
            if not passed:
                print(f"  - {name}: {message}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)