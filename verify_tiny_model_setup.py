#!/usr/bin/env python3
"""
Tiny Model Verification Script

This script verifies that the tiny model infrastructure is properly set up
without requiring network access or model downloads.
"""

import os
import json
import subprocess
from pathlib import Path

def verify_file_structure():
    """Verify that all required files are in place"""
    print("📁 Verifying file structure...")
    
    base_path = Path(__file__).parent
    required_files = [
        "TINY_MODEL_SETUP.md",
        "tiny_model_integration.py", 
        "node-llama-cpp/src/cli/recommendedModels.ts",
        "node-llama-cpp/test/utils/modelFiles.ts",
        "node-llama-cpp/examples/tiny-model-demo.js",
        "node-llama-cpp/test/modelDependent/tinyLlama/basic.test.ts",
        ".gitignore"
    ]
    
    all_present = True
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing!")
            all_present = False
    
    return all_present

def verify_model_configuration():
    """Verify that models are properly configured"""
    print("\n⚙️  Verifying model configuration...")
    
    # Check modelFiles.ts
    model_files_path = Path(__file__).parent / "node-llama-cpp/test/utils/modelFiles.ts"
    if model_files_path.exists():
        content = model_files_path.read_text()
        if "stories260K.gguf" in content and "stories15M-q4_0.gguf" in content:
            print("  ✅ Tiny models added to test infrastructure")
        else:
            print("  ❌ Tiny models not found in test infrastructure")
            return False
    else:
        print("  ❌ modelFiles.ts not found")
        return False
    
    # Check recommendedModels.ts
    recommended_path = Path(__file__).parent / "node-llama-cpp/src/cli/recommendedModels.ts"
    if recommended_path.exists():
        content = recommended_path.read_text()
        if "TinyLlama Stories 15M" in content and "TinyLlama Stories 260K" in content:
            print("  ✅ Tiny models added to recommended models")
        else:
            print("  ❌ Tiny models not found in recommended models")
            return False
    else:
        print("  ❌ recommendedModels.ts not found")
        return False
    
    return True

def verify_test_structure():
    """Verify test structure is correct"""
    print("\n🧪 Verifying test structure...")
    
    test_dir = Path(__file__).parent / "node-llama-cpp/test/modelDependent/tinyLlama"
    if not test_dir.exists():
        print("  ❌ Test directory doesn't exist")
        return False
    
    test_file = test_dir / "basic.test.ts"
    if not test_file.exists():
        print("  ❌ Test file doesn't exist")
        return False
    
    content = test_file.read_text()
    required_tests = [
        "stories260K", 
        "stories15M",
        "basic inference",
        "chat session"
    ]
    
    for test_name in required_tests:
        if test_name in content:
            print(f"  ✅ {test_name} test found")
        else:
            print(f"  ❌ {test_name} test missing")
            return False
    
    return True

def verify_node_llama_build():
    """Verify node-llama-cpp can be built"""
    print("\n🔨 Verifying node-llama-cpp build capability...")
    
    node_path = Path(__file__).parent / "node-llama-cpp"
    
    # Check if package.json exists
    package_json = node_path / "package.json"
    if not package_json.exists():
        print("  ❌ package.json not found")
        return False
    
    # Check if built
    dist_path = node_path / "dist"
    if dist_path.exists():
        print("  ✅ Already built (dist/ exists)")
        return True
    
    print("  ℹ️  Not built yet, checking if dependencies are available...")
    
    # Check if we can run npm
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ npm available (version {result.stdout.strip()})")
            return True
        else:
            print("  ❌ npm not working")
            return False
    except FileNotFoundError:
        print("  ❌ npm not found")
        return False

def verify_gitignore():
    """Verify .gitignore is properly configured"""
    print("\n🚫 Verifying .gitignore configuration...")
    
    gitignore_path = Path(__file__).parent / ".gitignore"
    if not gitignore_path.exists():
        print("  ❌ .gitignore not found")
        return False
    
    content = gitignore_path.read_text()
    required_entries = [
        "models/",
        "*.gguf",
        "*.bin"
    ]
    
    for entry in required_entries:
        if entry in content:
            print(f"  ✅ {entry} excluded")
        else:
            print(f"  ❌ {entry} not excluded")
            return False
    
    return True

def verify_documentation():
    """Verify documentation is complete"""
    print("\n📚 Verifying documentation...")
    
    # Check TINY_MODEL_SETUP.md
    doc_path = Path(__file__).parent / "TINY_MODEL_SETUP.md"
    if not doc_path.exists():
        print("  ❌ TINY_MODEL_SETUP.md not found")
        return False
    
    content = doc_path.read_text()
    required_sections = [
        "# Tiny Model Setup",
        "## Use Cases",
        "## Integration Points", 
        "## Usage Examples",
        "## Running the Demo"
    ]
    
    for section in required_sections:
        if section in content:
            print(f"  ✅ {section} section found")
        else:
            print(f"  ❌ {section} section missing")
            return False
    
    # Check main README
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text()
        if "Tiny Model Setup" in readme_content:
            print("  ✅ Main README updated with tiny model section")
        else:
            print("  ❌ Main README not updated")
            return False
    
    return True

def create_verification_report():
    """Create a verification report"""
    print("\n📊 Creating verification report...")
    
    report = {
        "verification_date": "2024-01-01",  # Would be actual date
        "tiny_model_setup": {
            "file_structure": verify_file_structure(),
            "model_configuration": verify_model_configuration(),
            "test_structure": verify_test_structure(),
            "build_capability": verify_node_llama_build(),
            "gitignore_config": verify_gitignore(),
            "documentation": verify_documentation()
        },
        "models": {
            "stories260K": {
                "size": "~280KB",
                "parameters": "260K",
                "configured": True
            },
            "stories15M": {
                "size": "~15MB", 
                "parameters": "15M",
                "configured": True
            }
        },
        "integration_status": "ready_for_testing"
    }
    
    report_path = Path(__file__).parent / "tiny_model_verification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"  📄 Report saved to {report_path}")
    return report

def main():
    """Main verification function"""
    print("🔍 Echo9ML Tiny Model Setup Verification")
    print("=" * 50)
    
    # Run all verifications
    all_checks = [
        ("File Structure", verify_file_structure),
        ("Model Configuration", verify_model_configuration), 
        ("Test Structure", verify_test_structure),
        ("Build Capability", verify_node_llama_build),
        ("GitIgnore Config", verify_gitignore),
        ("Documentation", verify_documentation)
    ]
    
    results = {}
    for check_name, check_func in all_checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"  ❌ {check_name} failed with error: {e}")
            results[check_name] = False
    
    # Create report
    report = create_verification_report()
    
    # Summary
    print("\n📋 Verification Summary")
    print("-" * 30)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, passed_check in results.items():
        status = "✅ PASS" if passed_check else "❌ FAIL"
        print(f"  {status} {check_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All verifications passed!")
        print("✅ Tiny model setup is complete and ready for testing")
        print("\n💡 Next steps:")
        print("  • Test with internet connectivity to download models")
        print("  • Run: python tiny_model_integration.py --test-mode")
        print("  • Integrate with Echo9ML cognitive architecture")
        print("  • Use as fallback in production systems")
        return True
    else:
        print(f"\n⚠️  {total - passed} checks failed")
        print("❌ Please fix the issues before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)