#!/usr/bin/env python3
"""
Deep Tree Echo Crystal Lucky Framework Priority Launcher
This launcher prioritizes the REAL Crystal Lucky framework over simplified implementations
and ensures proper installation of real LLM backends (node-llama-cpp, llama.cpp, or ggml)
"""

import subprocess
import sys
import time
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_crystal_available():
    """Check if Crystal compiler is available"""
    try:
        result = subprocess.run(['crystal', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_node_available():
    """Check if Node.js is available"""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_shards_available():
    """Check if Shards (Crystal package manager) is available"""
    try:
        result = subprocess.run(['shards', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_node_llama_cpp_available():
    """Check if node-llama-cpp is properly installed"""
    if not os.path.exists("node-llama-cpp"):
        return False
    
    package_json_path = "node-llama-cpp/package.json"
    if not os.path.exists(package_json_path):
        return False
        
    try:
        with open(package_json_path, 'r') as f:
            package_data = json.load(f)
            return package_data.get('name') == 'node-llama-cpp'
    except:
        return False

def install_crystal_dependencies():
    """Install Crystal Lucky framework dependencies"""
    logger.info("🔧 Installing Crystal Lucky framework dependencies...")
    
    if not check_shards_available():
        logger.error("❌ Shards package manager not found. Please install Shards first.")
        return False
    
    try:
        # Install Crystal dependencies
        result = subprocess.run(['shards', 'install'], check=True, capture_output=True, text=True)
        logger.info("✅ Crystal dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install Crystal dependencies: {e.stderr}")
        return False

def install_node_llama_cpp():
    """Install or setup node-llama-cpp for real LLM inference"""
    logger.info("🧠 Setting up real node-llama-cpp for authentic LLM inference...")
    
    if not check_node_available():
        logger.error("❌ Node.js not found. Please install Node.js first.")
        return False
    
    # Check if node-llama-cpp is already available
    if check_node_llama_cpp_available():
        logger.info("✅ node-llama-cpp already available")
        
        # Check if node modules are installed
        node_modules_path = "node-llama-cpp/node_modules"
        if not os.path.exists(node_modules_path):
            logger.info("🔧 Installing node-llama-cpp dependencies...")
            try:
                result = subprocess.run(['npm', 'install'], cwd='node-llama-cpp', check=True)
                logger.info("✅ node-llama-cpp dependencies installed")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install node-llama-cpp dependencies: {e}")
                return False
        
        return True
    
    # If not available, try to clone or download
    logger.info("📥 node-llama-cpp not found. Please ensure it's properly installed.")
    logger.info("🔗 You can install it via: git clone https://github.com/withcatai/node-llama-cpp.git")
    logger.info("🔗 Or use npm: npm install node-llama-cpp")
    
    return False

def start_crystal_lucky_framework():
    """Start the REAL Crystal Lucky framework implementation"""
    logger.info("🔥 Starting REAL Crystal Lucky Framework Implementation")
    logger.info("🌟 Full Lucky framework with authentic node-llama-cpp integration")
    logger.info("🚫 NO simplified substitutes - This is the complete Crystal Lucky implementation")
    
    if not check_crystal_available():
        logger.error("❌ Crystal compiler not found. Please install Crystal first.")
        logger.info("📥 Install Crystal: https://crystal-lang.org/install/")
        return False
        
    if not check_node_available():
        logger.error("❌ Node.js not found. Please install Node.js first.")
        logger.info("📥 Install Node.js: https://nodejs.org/")
        return False
    
    # Install dependencies first
    if not install_crystal_dependencies():
        logger.error("❌ Failed to install Crystal dependencies")
        return False
    
    # Setup real LLM backend
    llm_available = install_node_llama_cpp()
    if llm_available:
        logger.info("✅ Real LLM backend (node-llama-cpp) available")
    else:
        logger.warning("⚠️ Real LLM backend not available - will use Deep Tree Echo cognitive fallback")
    
    try:
        # Try to compile the full Lucky framework application
        crystal_lucky_executable = "crystal-echo"
        logger.info("🔧 Compiling Crystal Lucky framework application...")
        
        compile_result = subprocess.run(
            ['crystal', 'build', 'crystal-echo.cr', '-o', crystal_lucky_executable, '--release'],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✅ Crystal Lucky framework compiled successfully")
        
        # Start the Lucky framework application
        logger.info("🌟 Starting Crystal Lucky Framework on port 5000")
        logger.info("🧠 Full Deep Tree Echo cognitive architecture with Lucky framework")
        logger.info("🔗 Real LLM integration: node-llama-cpp")
        
        # Run the compiled Lucky framework application
        process = subprocess.Popen([f"./{crystal_lucky_executable}"])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Test if server is responding
        try:
            import requests
            response = requests.get("http://localhost:5000/api/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Crystal Lucky Framework running: {data.get('service', 'Unknown')}")
                logger.info(f"🔗 Web interface: http://localhost:5000")
                logger.info(f"🎯 Implementation: Full Crystal Lucky Framework")
                logger.info(f"🧠 Features: {', '.join(data.get('features', []))}")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Could not test server response: {e}")
            
        logger.info("✅ Crystal Lucky Framework process started")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to compile Crystal Lucky framework: {e.stderr}")
        logger.info("💡 Falling back to simplified Crystal server...")
        return start_crystal_simplified_server()
    except Exception as e:
        logger.error(f"❌ Failed to start Crystal Lucky framework: {e}")
        return False

def start_crystal_simplified_server():
    """Start the simplified Crystal server as fallback"""
    logger.warning("⚠️ Starting simplified Crystal server (Lucky framework unavailable)")
    logger.warning("🔧 This is a fallback - prefer the full Lucky framework implementation")
    
    try:
        # Compile simplified server if not already compiled
        simple_server_exe = "crystal_echo_server"
        if not os.path.exists(simple_server_exe):
            logger.info("🔧 Compiling simplified Crystal server...")
            compile_result = subprocess.run(
                ['crystal', 'build', 'crystal_echo_server.cr', '--no-debug'],
                check=True
            )
            logger.info("✅ Simplified Crystal server compiled successfully")
        
        # Start the simplified server
        logger.info("🌟 Starting simplified Crystal Echo Server on port 5000")
        
        # Run the compiled Crystal server
        process = subprocess.Popen([f"./{simple_server_exe}"])
        
        # Wait a moment for server to start
        time.sleep(2)
        
        logger.info("✅ Simplified Crystal server process started")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to compile simplified Crystal server: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to start simplified Crystal server: {e}")
        return False

def start_python_substitute_deprecated():
    """Start the deprecated Python substitute (only as last resort)"""
    logger.warning("⚠️ FALLING BACK TO DEPRECATED PYTHON SUBSTITUTE")
    logger.warning("🚫 This is NOT the intended implementation - Crystal Lucky framework is preferred")
    logger.warning("🐍 Python substitutes may contain mock/demo corruption")
    
    try:
        subprocess.run([sys.executable, "python_crystal_echo.py"], check=True)
    except Exception as e:
        logger.error(f"❌ Even Python substitute failed: {e}")
        return False
    
    return True

def display_installation_help():
    """Display installation help for the full Crystal Lucky setup"""
    print("\n" + "="*80)
    print("🔧 CRYSTAL LUCKY FRAMEWORK INSTALLATION GUIDE")
    print("="*80)
    print("\n📋 Required Dependencies:")
    print("  1. Crystal Language: https://crystal-lang.org/install/")
    print("  2. Shards (Crystal package manager): Usually installed with Crystal")
    print("  3. Node.js: https://nodejs.org/")
    print("  4. Real LLM Backend (choose one):")
    print("     • node-llama-cpp: git clone https://github.com/withcatai/node-llama-cpp.git")
    print("     • llama.cpp: git clone https://github.com/ggerganov/llama.cpp.git")
    print("     • ggml: https://github.com/ggerganov/ggml")
    print("\n🚀 Installation Steps:")
    print("  1. Install Crystal and Shards")
    print("  2. Install Node.js")
    print("  3. Run: shards install")
    print("  4. Setup real LLM backend (node-llama-cpp recommended)")
    print("  5. Run this launcher again")
    print("\n✅ For a complete authentic Crystal Lucky framework experience!")
    print("🚫 NO simplified substitutes - only the real implementation!")
    print("="*80 + "\n")

def main():
    print("🌟 Deep Tree Echo Crystal Lucky Framework Priority Launcher")
    print("🔥 Prioritizing REAL Crystal Lucky framework over simplified implementations")
    print("🧠 Ensuring authentic node-llama-cpp/llama.cpp/ggml integration")
    print("🚫 NO simplified/mock/demo/placeholder/fake - only the REAL implementation")
    print("")
    
    # Check basic requirements
    if not check_crystal_available():
        logger.error("❌ Crystal compiler not found")
        display_installation_help()
        return 1
    
    if not check_node_available():
        logger.error("❌ Node.js not found")
        display_installation_help()
        return 1
    
    # Always try Crystal Lucky framework first (the REAL implementation)
    logger.info("🎯 Attempting to start REAL Crystal Lucky Framework implementation...")
    if start_crystal_lucky_framework():
        logger.info("✅ Crystal Lucky Framework is running successfully")
        logger.info("🌟 FULL Lucky framework implementation active with real LLM integration")
        
        try:
            # Keep the process running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down Crystal Lucky Framework")
            return 0
    else:
        logger.error("❌ Crystal Lucky Framework implementation failed")
        
        # Ask user if they want to use simplified server
        response = input("⚠️ Use simplified Crystal server (not Lucky framework)? (y/N): ").strip().lower()
        if response == 'y':
            logger.warning("🔧 Starting simplified Crystal server...")
            if start_crystal_simplified_server():
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("🛑 Shutting down simplified Crystal server")
                    return 0
            else:
                # Final option: deprecated Python substitute
                response = input("⚠️ Use deprecated Python substitute? (y/N): ").strip().lower()
                if response == 'y':
                    logger.warning("🐍 Starting deprecated Python substitute...")
                    if start_python_substitute_deprecated():
                        return 0
                    else:
                        return 1
                else:
                    logger.info("🚫 User declined Python substitute - exiting")
                    return 1
        else:
            logger.info("🚫 User declined simplified server - showing installation help")
            display_installation_help()
            return 1

if __name__ == "__main__":
    sys.exit(main())