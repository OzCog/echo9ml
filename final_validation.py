#!/usr/bin/env python3
"""
Final Validation Test for Deep Tree Echo Multi-Language System
"""

import subprocess
import time
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_components():
    """Validate all system components"""
    results = {}
    
    # Test C++ Orchestrator
    logger.info("🔧 Validating C++ Deep Tree Echo Orchestrator...")
    try:
        result = subprocess.run(
            ["./deep-tree-echo"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd="/home/runner/work/echo9ml/echo9ml"
        )
        
        success = (result.returncode == 0 and 
                  "Deep Tree Echo Orchestrator" in result.stdout and
                  "Echo Pattern Analysis Complete" in result.stdout)
        
        results["cpp_orchestrator"] = {
            "status": "✅ PASSED" if success else "❌ FAILED",
            "details": "C++ orchestrator creates echo nodes and runs pattern analysis"
        }
        
    except Exception as e:
        results["cpp_orchestrator"] = {
            "status": "❌ ERROR",
            "details": str(e)
        }
    
    # Test Go Engine
    logger.info("🚀 Validating Go Hyper-Echo Engine...")
    try:
        # Start Go engine briefly
        process = subprocess.Popen(
            ["./hyper-echo"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/home/runner/work/echo9ml/echo9ml"
        )
        
        time.sleep(3)
        
        # Check if it's still running and producing output
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
            
            stdout, stderr = process.communicate()
            success = ("WebSocket server running on :8080" in stdout.decode() and
                      "Workers:" in stdout.decode())
        else:
            success = False
            
        results["go_engine"] = {
            "status": "✅ PASSED" if success else "❌ FAILED",
            "details": "Go engine starts WebSocket server with workers"
        }
        
    except Exception as e:
        results["go_engine"] = {
            "status": "❌ ERROR", 
            "details": str(e)
        }
    
    # Test Python Integration
    logger.info("🐍 Validating Python Integration...")
    try:
        import sys
        sys.path.append("/home/runner/work/echo9ml/echo9ml")
        from deep_tree_echo_integration import MultiLanguageOrchestrator
        
        orchestrator = MultiLanguageOrchestrator()
        
        results["python_integration"] = {
            "status": "✅ PASSED",
            "details": "Python integration orchestrator imports and initializes successfully"
        }
        
    except Exception as e:
        results["python_integration"] = {
            "status": "❌ ERROR",
            "details": str(e)
        }
    
    # Test Python Crystal Substitute
    logger.info("🌟 Validating Python Crystal Substitute...")
    try:
        # Start Crystal substitute briefly
        process = subprocess.Popen(
            ["python3", "python_crystal_echo.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/home/runner/work/echo9ml/echo9ml"
        )
        
        time.sleep(5)
        
        # Test API
        try:
            response = requests.get("http://localhost:5000/api/status", timeout=5)
            api_success = response.status_code == 200
            data = response.json()
            feature_success = "echo_value_propagation" in data.get("features", [])
            success = api_success and feature_success
        except:
            success = False
            
        process.terminate()
        process.wait(timeout=5)
        
        results["crystal_substitute"] = {
            "status": "✅ PASSED" if success else "❌ FAILED",
            "details": "Python Crystal substitute provides Flask API with echo features"
        }
        
    except Exception as e:
        results["crystal_substitute"] = {
            "status": "❌ ERROR",
            "details": str(e)
        }
    
    return results

def main():
    logger.info("🎯 Final Validation of Deep Tree Echo Multi-Language System")
    logger.info("=" * 70)
    
    results = validate_components()
    
    # Print results
    logger.info("📋 VALIDATION RESULTS")
    logger.info("=" * 70)
    
    passed = 0
    total = len(results)
    
    for component, result in results.items():
        status = result["status"]
        details = result["details"]
        
        logger.info(f"{component.upper().replace('_', ' ')}: {status}")
        logger.info(f"  Details: {details}")
        
        if "✅" in status:
            passed += 1
    
    logger.info("=" * 70)
    logger.info(f"🎯 FINAL SCORE: {passed}/{total} components validated successfully")
    
    if passed == total:
        logger.info("🎉 SUCCESS! Deep Tree Echo Multi-Language System is COMPLETE and OPERATIONAL!")
        logger.info("")
        logger.info("✨ System Components:")
        logger.info("   • C++ Deep Tree Echo Orchestrator - Neural processing & inference")
        logger.info("   • Go Hyper-Echo WebSocket Engine - High-performance execution")
        logger.info("   • Python Crystal Echo Interface - Web-based chatbot")
        logger.info("   • Python Integration Orchestrator - Multi-language coordination")
        logger.info("")
        logger.info("🌐 Access Points:")
        logger.info("   • Web Interface: http://localhost:5000")
        logger.info("   • WebSocket API: ws://localhost:8080/ws")
        logger.info("   • REST API: http://localhost:5000/api/status")
        logger.info("")
        logger.info("🚀 To run the complete system: python3 launch_deep_tree_echo_complete.py")
        return True
    else:
        logger.warning("⚠️  Some components need attention")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)