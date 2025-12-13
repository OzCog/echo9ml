#!/usr/bin/env python3
"""
Unified Launcher for Deep Tree Echo Multi-Language System

This script provides a unified interface to launch and manage all components
of the Deep Tree Echo persona system:
- C++ orchestrator for neural processing
- Go execution engine for concurrent task handling
- Python integration for coordination and monitoring
"""

import asyncio
import argparse
import json
import logging
import subprocess
import sys
import time
import signal
from pathlib import Path
from typing import Dict, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepTreeEchoLauncher:
    """Unified launcher for the Deep Tree Echo system"""
    
    def __init__(self, verbose=False):
        self.processes = {}
        self.verbose = verbose
        self.running = False
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            
    def check_executables(self):
        """Check if all required executables exist"""
        logger.info("Checking for required executables...")
        
        executables = {
            'cpp': './deep-tree-echo',
            'go': './hyper-echo'
        }
        
        missing = []
        for name, path in executables.items():
            if not Path(path).exists():
                missing.append(f"{name} ({path})")
                logger.error(f"✗ Missing {name} executable: {path}")
            else:
                logger.info(f"✓ Found {name} executable: {path}")
                
        if missing:
            logger.error(f"\nMissing executables: {', '.join(missing)}")
            logger.error("Please compile the components first:")
            logger.error("  C++: g++ -std=c++17 -O2 -o deep-tree-echo deep-tree-echo.cpp")
            logger.error("  Go:  go build -o hyper-echo hyper-echo.go")
            return False
            
        return True
        
    def start_cpp_orchestrator(self, background=False):
        """Start the C++ orchestrator"""
        logger.info("Starting C++ Deep Tree Echo Orchestrator...")
        
        try:
            if background:
                process = subprocess.Popen(
                    ["./deep-tree-echo"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                self.processes['cpp'] = process
                logger.info("✓ C++ orchestrator started in background (PID: %d)", process.pid)
            else:
                result = subprocess.run(
                    ["./deep-tree-echo"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if self.verbose:
                    logger.debug("C++ Output:\n%s", result.stdout)
                    
                if result.returncode == 0:
                    logger.info("✓ C++ orchestrator completed successfully")
                else:
                    logger.error("✗ C++ orchestrator failed with code %d", result.returncode)
                    return False
                    
            return True
            
        except Exception as e:
            logger.error("✗ Failed to start C++ orchestrator: %s", e)
            return False
            
    def start_go_engine(self):
        """Start the Go execution engine"""
        logger.info("Starting Go Hyper-Echo Execution Engine...")
        
        try:
            process = subprocess.Popen(
                ["./hyper-echo"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            self.processes['go'] = process
            
            # Wait for startup and capture initial output
            logger.info("Waiting for Go engine to initialize...")
            time.sleep(2)
            
            if process.poll() is not None:
                output, _ = process.communicate()
                logger.error("✗ Go engine terminated unexpectedly")
                if self.verbose:
                    logger.debug("Output:\n%s", output)
                return False
                
            logger.info("✓ Go execution engine started (PID: %d)", process.pid)
            logger.info("  WebSocket server: ws://localhost:8080/ws")
            logger.info("  Status endpoint: http://localhost:8080/status")
            
            return True
            
        except Exception as e:
            logger.error("✗ Failed to start Go engine: %s", e)
            return False
            
    def monitor_processes(self):
        """Monitor running processes"""
        logger.info("\nMonitoring Deep Tree Echo system...")
        logger.info("Press Ctrl+C to stop\n")
        
        self.running = True
        
        try:
            while self.running:
                # Check process status
                all_running = True
                
                for name, process in self.processes.items():
                    if process.poll() is not None:
                        logger.warning("Process %s has terminated", name)
                        all_running = False
                        
                if not all_running:
                    logger.error("Some processes have terminated. Shutting down.")
                    break
                    
                # Read Go engine output
                if 'go' in self.processes:
                    go_process = self.processes['go']
                    if go_process.stdout:
                        try:
                            line = go_process.stdout.readline()
                            if line and self.verbose:
                                logger.debug("Go: %s", line.strip())
                        except (IOError, OSError) as e:
                            logger.debug("Error reading output: %s", e)
                            pass
                            
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("\nShutdown requested by user")
            
    def stop_all(self):
        """Stop all running processes"""
        logger.info("Stopping all processes...")
        
        self.running = False
        
        for name, process in self.processes.items():
            if process and process.poll() is None:
                logger.info("Terminating %s (PID: %d)...", name, process.pid)
                process.terminate()
                
                try:
                    process.wait(timeout=5)
                    logger.info("✓ %s stopped gracefully", name)
                except subprocess.TimeoutExpired:
                    logger.warning("Force killing %s", name)
                    process.kill()
                    process.wait()
                    
        self.processes.clear()
        logger.info("All processes stopped")
        
    def run_demo(self):
        """Run a demonstration of the system"""
        logger.info("=" * 70)
        logger.info("DEEP TREE ECHO MULTI-LANGUAGE SYSTEM DEMONSTRATION")
        logger.info("=" * 70)
        logger.info("")
        
        # Check prerequisites
        if not self.check_executables():
            return False
            
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 1: C++ Neural Tree Processing")
        logger.info("=" * 70)
        logger.info("")
        
        if not self.start_cpp_orchestrator(background=False):
            return False
            
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 2: Go Concurrent Execution Engine")
        logger.info("=" * 70)
        logger.info("")
        
        if not self.start_go_engine():
            return False
            
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 3: System Monitoring")
        logger.info("=" * 70)
        logger.info("")
        
        # Monitor for a bit
        self.monitor_processes()
        
        # Cleanup
        self.stop_all()
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("Demonstration complete!")
        logger.info("=" * 70)
        
        return True
        
    def run_services(self):
        """Run all services continuously"""
        logger.info("=" * 70)
        logger.info("DEEP TREE ECHO MULTI-LANGUAGE SYSTEM")
        logger.info("=" * 70)
        logger.info("")
        
        # Check prerequisites
        if not self.check_executables():
            return False
            
        logger.info("")
        
        # Start Go engine
        if not self.start_go_engine():
            return False
            
        logger.info("")
        
        # Monitor continuously
        self.monitor_processes()
        
        # Cleanup
        self.stop_all()
        
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Unified launcher for Deep Tree Echo multi-language system'
    )
    
    parser.add_argument(
        '--mode',
        choices=['demo', 'service'],
        default='demo',
        help='Launch mode: demo (one-shot demo) or service (continuous)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = DeepTreeEchoLauncher(verbose=args.verbose)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("\nShutdown signal received")
        launcher.stop_all()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run in selected mode
    try:
        if args.mode == 'demo':
            success = launcher.run_demo()
        else:
            success = launcher.run_services()
            
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error("Launcher error: %s", e)
        launcher.stop_all()
        sys.exit(1)


if __name__ == "__main__":
    main()
