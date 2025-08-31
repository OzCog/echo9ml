#!/usr/bin/env python3
"""
Complete Deep Tree Echo Multi-Language System Launcher
"""

import asyncio
import subprocess
import time
import logging
import signal
import sys
import os
import json
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepTreeEchoLauncher:
    def __init__(self):
        self.processes = {}
        self.base_dir = "/home/runner/work/echo9ml/echo9ml"
        
    def start_go_engine(self):
        """Start Go WebSocket server"""
        logger.info("🚀 Starting Go Hyper-Echo Engine...")
        try:
            process = subprocess.Popen(
                ["./hyper-echo"],
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes["go"] = process
            time.sleep(3)  # Allow time to start
            
            if process.poll() is None:
                logger.info("✅ Go Engine started successfully")
                return True
            else:
                logger.error("❌ Go Engine failed to start")
                return False
        except Exception as e:
            logger.error(f"❌ Error starting Go Engine: {e}")
            return False
            
    def start_python_crystal_substitute(self):
        """Start Python Crystal substitute"""
        logger.info("🐍 Starting Python Crystal Echo Interface...")
        try:
            process = subprocess.Popen(
                [sys.executable, "python_crystal_echo.py"],
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes["crystal"] = process
            time.sleep(3)  # Allow time to start
            
            # Test if it's responding
            try:
                response = requests.get("http://localhost:5000/api/status", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Python Crystal substitute started successfully")
                    return True
            except:
                pass
                
            logger.error("❌ Python Crystal substitute failed to start")
            return False
        except Exception as e:
            logger.error(f"❌ Error starting Python Crystal substitute: {e}")
            return False
            
    def test_cpp_orchestrator(self):
        """Test C++ orchestrator"""
        logger.info("🔧 Testing C++ Deep Tree Echo Orchestrator...")
        try:
            result = subprocess.run(
                ["./deep-tree-echo"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.base_dir
            )
            
            if result.returncode == 0 and "Deep Tree Echo Orchestrator" in result.stdout:
                logger.info("✅ C++ Orchestrator working")
                return True
            else:
                logger.error("❌ C++ Orchestrator failed")
                return False
        except Exception as e:
            logger.error(f"❌ C++ Orchestrator error: {e}")
            return False
            
    async def test_integration(self):
        """Test full system integration"""
        logger.info("🌐 Testing Multi-Language Integration...")
        
        # Test WebSocket communication
        try:
            import websockets
            uri = "ws://localhost:8080/ws"
            async with websockets.connect(uri) as websocket:
                test_msg = {"type": "hyper_inference", "target": "test"}
                await websocket.send(json.dumps(test_msg))
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                logger.info("✅ WebSocket communication working")
        except Exception as e:
            logger.error(f"❌ WebSocket communication failed: {e}")
            return False
            
        # Test HTTP API
        try:
            response = requests.get("http://localhost:5000/api/status", timeout=5)
            if response.status_code == 200:
                logger.info("✅ HTTP API communication working")
            else:
                logger.error("❌ HTTP API communication failed")
                return False
        except Exception as e:
            logger.error(f"❌ HTTP API error: {e}")
            return False
            
        return True
        
    def show_status(self):
        """Show system status"""
        logger.info("=" * 60)
        logger.info("🎯 DEEP TREE ECHO MULTI-LANGUAGE SYSTEM STATUS")
        logger.info("=" * 60)
        
        # Component status
        components = [
            ("C++ Orchestrator", self.test_cpp_orchestrator()),
            ("Go WebSocket Engine", self.processes.get("go") and self.processes["go"].poll() is None),
            ("Python Crystal Interface", self.processes.get("crystal") and self.processes["crystal"].poll() is None),
        ]
        
        for name, status in components:
            status_icon = "✅" if status else "❌"
            logger.info(f"{status_icon} {name}: {'RUNNING' if status else 'STOPPED'}")
            
        # Service endpoints
        logger.info("")
        logger.info("🔗 Service Endpoints:")
        logger.info("   Go WebSocket: ws://localhost:8080/ws")
        logger.info("   Crystal Interface: http://localhost:5000")
        logger.info("   API Status: http://localhost:5000/api/status")
        
        return all(status for _, status in components)
        
    def cleanup(self):
        """Clean up processes"""
        logger.info("🧹 Cleaning up processes...")
        for name, process in self.processes.items():
            if process.poll() is None:
                logger.info(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
    async def run_demo(self):
        """Run the complete system demo"""
        logger.info("🎉 Starting Deep Tree Echo Multi-Language System Demo")
        logger.info("=" * 60)
        
        try:
            # Start components
            go_started = self.start_go_engine()
            crystal_started = self.start_python_crystal_substitute()
            
            if go_started and crystal_started:
                # Test integration
                integration_ok = await self.test_integration()
                
                # Show status
                self.show_status()
                
                if integration_ok:
                    logger.info("")
                    logger.info("🎉 SUCCESS! Deep Tree Echo Multi-Language System is OPERATIONAL!")
                    logger.info("🌐 Open http://localhost:5000 to access the chat interface")
                    logger.info("🔗 The system includes:")
                    logger.info("   • C++ Deep Tree Echo Orchestrator")
                    logger.info("   • Go Hyper-Echo WebSocket Engine")
                    logger.info("   • Python Crystal Echo Interface")
                    logger.info("   • Multi-language coordination")
                    logger.info("")
                    logger.info("⏯️  Press Ctrl+C to stop the system")
                    
                    # Keep running
                    try:
                        while True:
                            await asyncio.sleep(1)
                    except KeyboardInterrupt:
                        logger.info("🛑 Shutdown requested")
                        
                else:
                    logger.error("❌ Integration tests failed")
                    return False
            else:
                logger.error("❌ Failed to start core components")
                return False
                
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return False
        finally:
            self.cleanup()
            
        return True

async def main():
    launcher = DeepTreeEchoLauncher()
    
    # Handle signals
    def signal_handler(sig, frame):
        logger.info("🛑 Received shutdown signal")
        launcher.cleanup()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    success = await launcher.run_demo()
    return success

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested")
        sys.exit(0)