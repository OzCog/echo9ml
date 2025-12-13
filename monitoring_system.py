#!/usr/bin/env python3
"""
Monitoring and Logging Infrastructure for Deep Tree Echo System

This module provides centralized monitoring, logging, and health checking
for all components of the Deep Tree Echo multi-language system.
"""

import asyncio
import json
import logging
import os
import psutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque

from config_manager import DeepTreeEchoConfig


@dataclass
class SystemMetrics:
    """System-level metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int


@dataclass
class ComponentMetrics:
    """Component-level metrics"""
    name: str
    timestamp: float
    pid: Optional[int]
    status: str  # running, stopped, error
    cpu_percent: float
    memory_mb: float
    uptime_seconds: float
    restart_count: int
    last_error: Optional[str]


class MetricsCollector:
    """Collects and stores system and component metrics"""
    
    def __init__(self, history_size: int = 1000):
        self.system_metrics = deque(maxlen=history_size)
        self.component_metrics = {}
        self.start_time = time.time()
        
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        net_io = psutil.net_io_counters()
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=psutil.virtual_memory().percent,
            memory_mb=psutil.virtual_memory().used / 1024 / 1024,
            disk_percent=psutil.disk_usage('/').percent,
            network_bytes_sent=net_io.bytes_sent,
            network_bytes_recv=net_io.bytes_recv
        )
        
        self.system_metrics.append(metrics)
        return metrics
        
    def collect_component_metrics(self, name: str, pid: Optional[int],
                                  status: str, restart_count: int = 0,
                                  last_error: Optional[str] = None) -> ComponentMetrics:
        """Collect metrics for a specific component"""
        cpu_percent = 0.0
        memory_mb = 0.0
        
        if pid:
            try:
                process = psutil.Process(pid)
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_mb = process.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        metrics = ComponentMetrics(
            name=name,
            timestamp=time.time(),
            pid=pid,
            status=status,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            uptime_seconds=time.time() - self.start_time,
            restart_count=restart_count,
            last_error=last_error
        )
        
        if name not in self.component_metrics:
            self.component_metrics[name] = deque(maxlen=1000)
        self.component_metrics[name].append(metrics)
        
        return metrics
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self.start_time,
            'system': None,
            'components': {}
        }
        
        # Latest system metrics
        if self.system_metrics:
            summary['system'] = asdict(self.system_metrics[-1])
            
        # Latest component metrics
        for name, metrics in self.component_metrics.items():
            if metrics:
                summary['components'][name] = asdict(metrics[-1])
                
        return summary
        
    def export_to_file(self, filename: str):
        """Export metrics to JSON file"""
        data = {
            'exported_at': time.time(),
            'system_metrics': [asdict(m) for m in self.system_metrics],
            'component_metrics': {
                name: [asdict(m) for m in metrics]
                for name, metrics in self.component_metrics.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


class HealthChecker:
    """Performs health checks on system components"""
    
    def __init__(self, config: DeepTreeEchoConfig):
        self.config = config
        self.health_status = {}
        
    def check_component(self, name: str, pid: Optional[int]) -> Dict[str, Any]:
        """Check health of a specific component"""
        health = {
            'name': name,
            'timestamp': time.time(),
            'healthy': False,
            'checks': {}
        }
        
        # Check if process is running
        if pid:
            try:
                process = psutil.Process(pid)
                health['checks']['process_running'] = True
                health['checks']['cpu_usage'] = process.cpu_percent(interval=0.1)
                health['checks']['memory_mb'] = process.memory_info().rss / 1024 / 1024
                
                # Check CPU usage
                if health['checks']['cpu_usage'] > 90:
                    health['checks']['cpu_warning'] = True
                    
                # Check memory usage
                memory_limit = self.config.get('performance.memory_limit_mb', 4096)
                if health['checks']['memory_mb'] > memory_limit:
                    health['checks']['memory_warning'] = True
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                health['checks']['process_running'] = False
        else:
            health['checks']['process_running'] = False
            
        # Overall health
        health['healthy'] = health['checks'].get('process_running', False)
        
        self.health_status[name] = health
        return health
        
    def check_all(self, components: Dict[str, Optional[int]]) -> Dict[str, Any]:
        """Check health of all components"""
        results = {}
        
        for name, pid in components.items():
            results[name] = self.check_component(name, pid)
            
        # System-level checks
        system_health = {
            'timestamp': time.time(),
            'checks': {}
        }
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=0.1)
        system_health['checks']['cpu_percent'] = cpu_percent
        system_health['checks']['cpu_ok'] = cpu_percent < 80
        
        # Memory check
        mem = psutil.virtual_memory()
        system_health['checks']['memory_percent'] = mem.percent
        system_health['checks']['memory_ok'] = mem.percent < 80
        
        # Disk check
        disk = psutil.disk_usage('/')
        system_health['checks']['disk_percent'] = disk.percent
        system_health['checks']['disk_ok'] = disk.percent < 90
        
        system_health['healthy'] = all([
            system_health['checks']['cpu_ok'],
            system_health['checks']['memory_ok'],
            system_health['checks']['disk_ok']
        ])
        
        results['system'] = system_health
        
        return results


class DeepTreeEchoMonitor:
    """Main monitoring system for Deep Tree Echo"""
    
    def __init__(self, config: Optional[DeepTreeEchoConfig] = None):
        self.config = config or DeepTreeEchoConfig()
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker(self.config)
        self.logger = self._setup_logging()
        self.running = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = Path(self.config.get('monitoring.log_directory', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('DeepTreeEchoMonitor')
        logger.setLevel(logging.INFO)
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'monitor_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    async def monitor_loop(self, components: Dict[str, Optional[int]],
                          interval: int = 5):
        """Main monitoring loop"""
        self.running = True
        self.logger.info("Starting monitoring loop (interval: %ds)", interval)
        
        while self.running:
            try:
                # Collect metrics
                system_metrics = self.metrics_collector.collect_system_metrics()
                
                for name, pid in components.items():
                    self.metrics_collector.collect_component_metrics(
                        name, pid, 'running' if pid else 'stopped'
                    )
                    
                # Health checks
                health_results = self.health_checker.check_all(components)
                
                # Log summary
                self.logger.info(
                    "System: CPU=%.1f%% MEM=%.1f%% DISK=%.1f%%",
                    system_metrics.cpu_percent,
                    system_metrics.memory_percent,
                    system_metrics.disk_percent
                )
                
                for name, health in health_results.items():
                    if name != 'system':
                        status = "✓" if health['healthy'] else "✗"
                        self.logger.info(
                            "Component %s: %s %s",
                            name, status, health['checks']
                        )
                        
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error("Monitoring error: %s", e)
                await asyncio.sleep(interval)
                
    def stop(self):
        """Stop monitoring"""
        self.logger.info("Stopping monitoring")
        self.running = False
        
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        return {
            'timestamp': time.time(),
            'metrics': self.metrics_collector.get_summary(),
            'health': self.health_checker.health_status
        }
        
    def export_report(self, filename: Optional[str] = None):
        """Export monitoring report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'monitor_report_{timestamp}.json'
            
        report = self.get_status_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info("Exported report to %s", filename)


def main():
    """Main entry point for monitoring system"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Deep Tree Echo monitoring system'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Monitoring interval in seconds'
    )
    
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export metrics to file on exit'
    )
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = DeepTreeEchoMonitor()
    
    # Example: Monitor system without specific components
    components = {}
    
    print("Starting Deep Tree Echo monitoring system...")
    print(f"Monitoring interval: {args.interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    try:
        asyncio.run(monitor.monitor_loop(components, args.interval))
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        
    if args.export:
        monitor.export_report()
        print("Metrics exported")


if __name__ == "__main__":
    main()
