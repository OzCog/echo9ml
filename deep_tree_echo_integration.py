#!/usr/bin/env python3
"""
Deep Tree Echo Multi-Language Integration System

This script integrates the C++, Go, and Crystal components with the existing
Python Deep Tree Echo system, providing a unified orchestration interface.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
import websockets
import requests
from dataclasses import dataclass, asdict

# Import existing Deep Tree Echo system
from deep_tree_echo import DeepTreeEcho
from cognitive_architecture import CognitiveArchitecture
from ai_integration import AIIntegration

@dataclass
class ComponentStatus:
    """Status of a system component"""
    name: str
    status: str  # running, stopped, error
    pid: Optional[int] = None
    port: Optional[int] = None
    last_heartbeat: Optional[float] = None
    error_message: Optional[str] = None

class MultiLanguageOrchestrator:
    """
    Orchestrates the multi-language Deep Tree Echo system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Component processes
        self.processes: Dict[str, subprocess.Popen] = {}
        self.component_status: Dict[str, ComponentStatus] = {}
        
        # Initialize existing Python components
        self.deep_tree_echo = DeepTreeEcho()
        self.cognitive_arch = CognitiveArchitecture()
        self.ai_integration = AIIntegration()
        
        # Communication channels
        self.websocket_connections: Dict[str, Any] = {}
        self.message_queue = asyncio.Queue()
        
        # Configuration
        self.config = {
            "cpp_executable": "./deep-tree-echo",
            "go_executable": "./hyper-echo",
            "crystal_port": 5000,
            "go_websocket_port": 8080,
            "heartbeat_interval": 30,
            "max_restart_attempts": 3
        }
        
        self.logger.info("=== Multi-Language Deep Tree Echo Orchestrator Initialized ===")
    
    def setup_logging(self):
        """Configure logging for the orchestrator"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('deep_tree_echo_orchestrator.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    async def start_all_components(self):
        """Start all language components of the system"""
        self.logger.info("Starting all Deep Tree Echo components...")
        
        # Compile and start C++ orchestrator
        await self.start_cpp_orchestrator()
        
        # Start Go execution engine
        await self.start_go_engine()
        
        # Start Crystal chatbot interface
        await self.start_crystal_interface()
        
        # Start monitoring and heartbeat system
        await self.start_monitoring_system()
        
        # Initialize communication channels
        await self.initialize_communication()
        
        self.logger.info("=== All Deep Tree Echo Components Started ===")
    
    async def start_cpp_orchestrator(self):
        """Compile and start the C++ orchestrating agent"""
        try:
            self.logger.info("Compiling C++ Deep Tree Echo orchestrator...")
            
            # Compile C++ code
            compile_cmd = [
                "g++", "-std=c++17", "-O2", "-pthread",
                "deep-tree-echo.cpp", "-o", "deep-tree-echo"
            ]
            
            compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
            
            if compile_result.returncode != 0:
                raise Exception(f"C++ compilation failed: {compile_result.stderr}")
            
            self.logger.info("C++ orchestrator compiled successfully")
            
            # Start the C++ process
            cpp_process = subprocess.Popen(
                ["./deep-tree-echo"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes["cpp_orchestrator"] = cpp_process
            self.component_status["cpp_orchestrator"] = ComponentStatus(
                name="cpp_orchestrator",
                status="running",
                pid=cpp_process.pid,
                last_heartbeat=time.time()
            )
            
            self.logger.info(f"C++ orchestrator started with PID: {cpp_process.pid}")
            
        except Exception as e:
            self.logger.error(f"Failed to start C++ orchestrator: {e}")
            self.component_status["cpp_orchestrator"] = ComponentStatus(
                name="cpp_orchestrator",
                status="error",
                error_message=str(e)
            )
    
    async def start_go_engine(self):
        """Start the Go hyper-echo execution engine"""
        try:
            self.logger.info("Starting Go hyper-echo execution engine...")
            
            # Check if Go is installed
            go_version = subprocess.run(["go", "version"], capture_output=True, text=True)
            if go_version.returncode != 0:
                # Install required Go packages
                self.logger.info("Installing Go dependencies...")
                subprocess.run(["go", "mod", "init", "hyper-echo"], check=True)
                subprocess.run(["go", "mod", "tidy"], check=True)
            
            # Start the Go process
            go_process = subprocess.Popen(
                ["go", "run", "hyper-echo.go"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes["go_engine"] = go_process
            self.component_status["go_engine"] = ComponentStatus(
                name="go_engine",
                status="running",
                pid=go_process.pid,
                port=self.config["go_websocket_port"],
                last_heartbeat=time.time()
            )
            
            self.logger.info(f"Go engine started with PID: {go_process.pid}")
            
            # Wait a moment for the server to start
            await asyncio.sleep(2)
            
        except Exception as e:
            self.logger.error(f"Failed to start Go engine: {e}")
            self.component_status["go_engine"] = ComponentStatus(
                name="go_engine",
                status="error",
                error_message=str(e)
            )
    
    async def start_crystal_interface(self):
        """Start the Crystal Lucky chatbot interface"""
        try:
            self.logger.info("Starting Crystal Lucky chatbot interface...")
            
            # Check if Crystal is installed
            crystal_version = subprocess.run(["crystal", "--version"], capture_output=True, text=True)
            if crystal_version.returncode != 0:
                self.logger.warning("Crystal not found, skipping Crystal interface")
                self.component_status["crystal_interface"] = ComponentStatus(
                    name="crystal_interface",
                    status="error",
                    error_message="Crystal not installed"
                )
                return
            
            # Install Crystal dependencies
            self.logger.info("Installing Crystal dependencies...")
            subprocess.run(["shards", "install"], check=True)
            
            # Start the Crystal process
            crystal_process = subprocess.Popen(
                ["crystal", "run", "crystal-echo.cr"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes["crystal_interface"] = crystal_process
            self.component_status["crystal_interface"] = ComponentStatus(
                name="crystal_interface",
                status="running",
                pid=crystal_process.pid,
                port=self.config["crystal_port"],
                last_heartbeat=time.time()
            )
            
            self.logger.info(f"Crystal interface started with PID: {crystal_process.pid}")
            
        except Exception as e:
            self.logger.error(f"Failed to start Crystal interface: {e}")
            self.component_status["crystal_interface"] = ComponentStatus(
                name="crystal_interface",
                status="error",
                error_message=str(e)
            )
    
    async def initialize_communication(self):
        """Initialize communication channels between components"""
        self.logger.info("Initializing inter-component communication...")
        
        # Connect to Go WebSocket server
        try:
            go_ws_uri = f"ws://localhost:{self.config['go_websocket_port']}/ws"
            go_websocket = await websockets.connect(go_ws_uri)
            self.websocket_connections["go_engine"] = go_websocket
            self.logger.info("Connected to Go engine WebSocket")
        except Exception as e:
            self.logger.warning(f"Could not connect to Go WebSocket: {e}")
        
        # Test Crystal API connection
        try:
            crystal_url = f"http://localhost:{self.config['crystal_port']}/api/status"
            response = requests.get(crystal_url, timeout=5)
            if response.status_code == 200:
                self.logger.info("Crystal API connection verified")
            else:
                self.logger.warning(f"Crystal API returned status: {response.status_code}")
        except Exception as e:
            self.logger.warning(f"Could not connect to Crystal API: {e}")
    
    async def start_monitoring_system(self):
        """Start the monitoring and heartbeat system"""
        self.logger.info("Starting monitoring system...")
        
        # Start heartbeat task
        asyncio.create_task(self.heartbeat_monitor())
        
        # Start process monitor
        asyncio.create_task(self.process_monitor())
        
        # Start message processor
        asyncio.create_task(self.message_processor())
    
    async def heartbeat_monitor(self):
        """Monitor component heartbeats"""
        while True:
            try:
                current_time = time.time()
                
                for component_name, status in self.component_status.items():
                    if status.status == "running" and status.last_heartbeat:
                        time_since_heartbeat = current_time - status.last_heartbeat
                        
                        if time_since_heartbeat > self.config["heartbeat_interval"] * 2:
                            self.logger.warning(f"Component {component_name} heartbeat timeout")
                            await self.handle_component_failure(component_name)
                
                await asyncio.sleep(self.config["heartbeat_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {e}")
    
    async def process_monitor(self):
        """Monitor component processes"""
        while True:
            try:
                for component_name, process in self.processes.items():
                    if process.poll() is not None:  # Process has terminated
                        self.logger.warning(f"Process {component_name} has terminated")
                        await self.handle_component_failure(component_name)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in process monitor: {e}")
    
    async def message_processor(self):
        """Process inter-component messages"""
        while True:
            try:
                message = await self.message_queue.get()
                await self.route_message(message)
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    async def handle_component_failure(self, component_name: str):
        """Handle component failure and restart if possible"""
        self.logger.error(f"Handling failure for component: {component_name}")
        
        status = self.component_status.get(component_name)
        if not status:
            return
        
        status.status = "error"
        
        # Attempt restart based on component type
        if component_name == "cpp_orchestrator":
            await self.start_cpp_orchestrator()
        elif component_name == "go_engine":
            await self.start_go_engine()
        elif component_name == "crystal_interface":
            await self.start_crystal_interface()
    
    async def route_message(self, message: Dict[str, Any]):
        """Route messages between components"""
        target = message.get("target")
        message_type = message.get("type")
        
        if target == "go_engine" and "go_engine" in self.websocket_connections:
            try:
                await self.websocket_connections["go_engine"].send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Error sending message to Go engine: {e}")
        
        elif target == "crystal_interface":
            try:
                # Send HTTP request to Crystal API
                url = f"http://localhost:{self.config['crystal_port']}/api/echo/propagate/default"
                requests.post(url, json=message, timeout=5)
            except Exception as e:
                self.logger.error(f"Error sending message to Crystal interface: {e}")
    
    async def create_integrated_tree(self, content: str) -> Dict[str, Any]:
        """Create a tree using the integrated multi-language system"""
        self.logger.info(f"Creating integrated tree with content: {content[:50]}...")
        
        # Create tree in Python system
        python_root = self.deep_tree_echo.create_tree(content)
        
        # Notify C++ orchestrator
        cpp_message = {
            "target": "cpp_orchestrator",
            "type": "tree_created",
            "content": content,
            "python_echo": python_root.echo_value
        }
        await self.message_queue.put(cpp_message)
        
        # Create corresponding structure in Go engine
        go_message = {
            "type": "echo_propagation",
            "target": "root",
            "parameters": {
                "content": content,
                "python_echo": python_root.echo_value
            },
            "priority": 8,
            "timeout": 5000
        }
        await self.message_queue.put({**go_message, "target": "go_engine"})
        
        # Create chat session in Crystal interface
        try:
            crystal_url = f"http://localhost:{self.config['crystal_port']}/api/chat/sessions"
            crystal_response = requests.post(crystal_url, json={"user_id": "orchestrator"})
            crystal_session = crystal_response.json() if crystal_response.status_code == 200 else None
        except:
            crystal_session = None
        
        return {
            "python_tree": {
                "content": python_root.content,
                "echo_value": python_root.echo_value,
                "emotional_state": python_root.emotional_state.tolist(),
                "spatial_context": {
                    "position": python_root.spatial_context.position,
                    "depth": python_root.spatial_context.depth
                }
            },
            "cpp_notified": True,
            "go_notified": True,
            "crystal_session": crystal_session,
            "integration_timestamp": time.time()
        }
    
    async def propagate_integrated_echoes(self, tree_id: str = "root") -> Dict[str, Any]:
        """Propagate echoes across all language implementations"""
        self.logger.info("Propagating echoes across integrated system...")
        
        results = {}
        
        # Propagate in Python system
        if self.deep_tree_echo.root:
            self.deep_tree_echo.propagate_echoes()
            results["python"] = {
                "status": "completed",
                "root_echo": self.deep_tree_echo.root.echo_value
            }
        
        # Propagate in Go system
        go_propagate_message = {
            "type": "echo_propagation",
            "target": tree_id,
            "priority": 9,
            "timeout": 10000
        }
        await self.message_queue.put({**go_propagate_message, "target": "go_engine"})
        results["go"] = {"status": "requested"}
        
        # Propagate in Crystal system (if session exists)
        try:
            crystal_url = f"http://localhost:{self.config['crystal_port']}/api/echo/propagate/default"
            crystal_response = requests.post(crystal_url, timeout=5)
            results["crystal"] = {
                "status": "completed" if crystal_response.status_code == 200 else "error",
                "response": crystal_response.json() if crystal_response.status_code == 200 else None
            }
        except Exception as e:
            results["crystal"] = {"status": "error", "error": str(e)}
        
        return results
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "orchestrator": {
                "status": "running",
                "components": len(self.component_status),
                "active_processes": len(self.processes),
                "websocket_connections": len(self.websocket_connections)
            },
            "components": {name: asdict(status) for name, status in self.component_status.items()},
            "python_system": {
                "tree_exists": self.deep_tree_echo.root is not None,
                "root_echo": self.deep_tree_echo.root.echo_value if self.deep_tree_echo.root else None
            },
            "integration": {
                "node_llama_cpp": Path("node-llama-cpp").exists(),
                "cpp_compiled": Path("deep-tree-echo").exists(),
                "go_available": subprocess.run(["which", "go"], capture_output=True).returncode == 0,
                "crystal_available": subprocess.run(["which", "crystal"], capture_output=True).returncode == 0
            },
            "timestamp": time.time()
        }
        
        return status
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        self.logger.info("Shutting down Deep Tree Echo integrated system...")
        
        # Close WebSocket connections
        for name, ws in self.websocket_connections.items():
            try:
                await ws.close()
                self.logger.info(f"Closed WebSocket connection: {name}")
            except:
                pass
        
        # Terminate processes
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                self.logger.info(f"Terminated process: {name}")
            except:
                try:
                    process.kill()
                    self.logger.warning(f"Force killed process: {name}")
                except:
                    pass
        
        self.logger.info("=== Deep Tree Echo System Shutdown Complete ===")

# Demonstration and testing functions
async def demonstrate_integration():
    """Demonstrate the integrated multi-language system"""
    print("=== Deep Tree Echo Multi-Language Integration Demo ===")
    
    orchestrator = MultiLanguageOrchestrator()
    
    try:
        # Start all components
        await orchestrator.start_all_components()
        
        # Wait for components to initialize
        await asyncio.sleep(5)
        
        # Create an integrated tree
        tree_result = await orchestrator.create_integrated_tree(
            "Deep Tree Echo - Multi-language cognitive architecture with recursive introspection"
        )
        print("Integrated Tree Creation Result:")
        print(json.dumps(tree_result, indent=2))
        
        # Propagate echoes across all systems
        propagation_result = await orchestrator.propagate_integrated_echoes()
        print("\nEcho Propagation Result:")
        print(json.dumps(propagation_result, indent=2))
        
        # Get system status
        status = await orchestrator.get_system_status()
        print("\nSystem Status:")
        print(json.dumps(status, indent=2))
        
        # Wait a bit to see the system in action
        print("\n=== System Running - Press Ctrl+C to stop ===")
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error during demonstration: {e}")
    finally:
        await orchestrator.shutdown()

# CLI interface
def main():
    """Main entry point for the integrated system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep Tree Echo Multi-Language Integration System")
    parser.add_argument("--demo", action="store_true", help="Run integration demonstration")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--compile", action="store_true", help="Compile components only")
    
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(demonstrate_integration())
    elif args.status:
        orchestrator = MultiLanguageOrchestrator()
        status = asyncio.run(orchestrator.get_system_status())
        print(json.dumps(status, indent=2))
    elif args.compile:
        print("Compiling C++ component...")
        subprocess.run(["g++", "-std=c++17", "-O2", "-pthread", "deep-tree-echo.cpp", "-o", "deep-tree-echo"])
        print("Compilation complete")
    else:
        # Run full system
        asyncio.run(demonstrate_integration())

if __name__ == "__main__":
    main()