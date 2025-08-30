#!/usr/bin/env python3
"""
Deep Tree Echo Crystal Priority Launcher
This launcher prioritizes the REAL Crystal implementation and deprecates Python substitutes
"""

import subprocess
import sys
import time
import os
import logging

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

def start_crystal_server():
    """Start the REAL Crystal Echo server"""
    logger.info("🔥 Starting REAL Crystal Echo Implementation")
    logger.info("🚫 NO Python substitutes - Using authentic Crystal")
    
    if not check_crystal_available():
        logger.error("❌ Crystal compiler not found. Please install Crystal first.")
        return False
        
    if not check_node_available():
        logger.error("❌ Node.js not found. Please install Node.js first.")
        return False
    
    try:
        # Compile Crystal server if not already compiled
        crystal_exe = "crystal_echo_server"
        if not os.path.exists(crystal_exe):
            logger.info("🔧 Compiling Crystal Echo server...")
            compile_result = subprocess.run(
                ['crystal', 'build', 'crystal_echo_server.cr', '--no-debug'],
                check=True
            )
            logger.info("✅ Crystal server compiled successfully")
        
        # Start the Crystal server
        logger.info("🌟 Starting Crystal Echo Server on port 5000")
        logger.info("🧠 Real Deep Tree Echo cognitive architecture active")
        
        # Run the compiled Crystal server
        process = subprocess.Popen([f"./{crystal_exe}"])
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Test if server is responding
        try:
            import requests
            response = requests.get("http://localhost:5000/api/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Crystal server running: {data.get('service', 'Unknown')}")
                logger.info(f"🔗 Web interface: http://localhost:5000")
                logger.info(f"🎯 Implementation: {data.get('implementation', 'unknown')}")
                logger.info(f"🧠 Inference engine: {data.get('inference_engine', 'unknown')}")
                return True
        except:
            pass
            
        logger.info("✅ Crystal server process started")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to compile Crystal server: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to start Crystal server: {e}")
        return False

def start_python_substitute_deprecated():
    """Start the deprecated Python substitute (only as last resort)"""
    logger.warning("⚠️ FALLING BACK TO DEPRECATED PYTHON SUBSTITUTE")
    logger.warning("🚫 This is NOT the intended implementation - Crystal is preferred")
    logger.warning("🐍 Python substitutes may contain mock/demo corruption")
    
    try:
        subprocess.run([sys.executable, "python_crystal_echo.py"], check=True)
    except Exception as e:
        logger.error(f"❌ Even Python substitute failed: {e}")
        return False
    
    return True

def main():
    print("🌟 Deep Tree Echo Multi-Language System - Crystal Priority Launcher")
    print("🔥 Prioritizing REAL Crystal implementation over Python substitutes")
    print("🚫 Python substitutes are DEPRECATED due to mock/demo corruption concerns")
    print("")
    
    # Always try Crystal first
    logger.info("🎯 Attempting to start REAL Crystal Echo implementation...")
    if start_crystal_server():
        logger.info("✅ Crystal Echo Server is running successfully")
        logger.info("🌟 REAL Crystal implementation active - NO Python substitutes")
        
        try:
            # Keep the process running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down Crystal Echo Server")
            return 0
    else:
        logger.error("❌ Crystal implementation failed")
        
        # Ask user if they want to use deprecated Python substitute
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

if __name__ == "__main__":
    sys.exit(main())