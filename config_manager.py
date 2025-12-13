#!/usr/bin/env python3
"""
Configuration Management for Deep Tree Echo System

This module provides centralized configuration management for all
components of the Deep Tree Echo multi-language system.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DeepTreeEchoConfig:
    """Configuration manager for Deep Tree Echo system"""
    
    # Default configuration
    DEFAULT_CONFIG = {
        "system": {
            "name": "Deep Tree Echo Multi-Language System",
            "version": "1.0.0",
            "description": "Multi-language cognitive architecture with C++, Go, Crystal, and Python"
        },
        "cpp": {
            "executable": "./deep-tree-echo",
            "echo_threshold": 0.75,
            "max_depth": 10,
            "thread_pool_size": 4,
            "enable_inference": True
        },
        "go": {
            "executable": "./hyper-echo",
            "websocket_port": 8080,
            "http_port": 8081,
            "worker_count": 4,
            "buffer_size": 1024,
            "timeout_seconds": 30
        },
        "crystal": {
            "executable": "./crystal-echo",
            "http_port": 5000,
            "enable_websocket": True,
            "session_timeout": 3600
        },
        "python": {
            "log_level": "INFO",
            "monitoring_interval": 5,
            "restart_on_failure": True,
            "max_restart_attempts": 3
        },
        "inference": {
            "model_path": "./models",
            "context_size": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512
        },
        "communication": {
            "websocket_url": "ws://localhost:8080/ws",
            "http_base_url": "http://localhost:8080",
            "crystal_url": "http://localhost:5000",
            "timeout": 10,
            "retry_attempts": 3
        },
        "monitoring": {
            "enable_logging": True,
            "log_directory": "./logs",
            "metrics_enabled": True,
            "metrics_interval": 60,
            "health_check_interval": 30
        },
        "performance": {
            "enable_profiling": False,
            "memory_limit_mb": 4096,
            "cpu_affinity": None,
            "gc_interval": 300
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to custom configuration file (JSON)
        """
        self.config_file = config_file or "deep_tree_echo_config.json"
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load custom configuration if exists
        if Path(self.config_file).exists():
            self.load_config()
        else:
            logger.info("No custom config found, using defaults")
            
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                custom_config = json.load(f)
                
            # Merge with defaults
            self._deep_merge(self.config, custom_config)
            
            logger.info("Loaded configuration from %s", self.config_file)
            return True
            
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            return False
            
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            logger.info("Saved configuration to %s", self.config_file)
            return True
            
        except Exception as e:
            logger.error("Failed to save config: %s", e)
            return False
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'cpp.echo_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'cpp.echo_threshold')
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        keys = key.split('.')
        config = self.config
        
        try:
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
                
            config[keys[-1]] = value
            return True
            
        except Exception as e:
            logger.error("Failed to set config %s: %s", key, e)
            return False
            
    def get_component_config(self, component: str) -> Dict[str, Any]:
        """Get configuration for a specific component"""
        return self.config.get(component, {})
        
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Check required executables exist
        for component in ['cpp', 'go']:
            exe_path = self.get(f"{component}.executable")
            if not Path(exe_path).exists():
                errors.append(f"Missing {component} executable: {exe_path}")
                
        # Check port conflicts
        ports = []
        for component in ['go', 'crystal']:
            for port_type in ['websocket_port', 'http_port']:
                port = self.get(f"{component}.{port_type}")
                if port and port in ports:
                    errors.append(f"Port conflict: {port} used multiple times")
                if port:
                    ports.append(port)
                    
        # Check log directory
        log_dir = self.get('monitoring.log_directory')
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            
        if errors:
            for error in errors:
                logger.error("Config validation error: %s", error)
            return False
            
        logger.info("Configuration validated successfully")
        return True
        
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Recursively merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
        
    def export_env_vars(self) -> Dict[str, str]:
        """Export configuration as environment variables"""
        env_vars = {}
        
        def flatten(obj, prefix=''):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}_{key}".upper() if prefix else key.upper()
                    flatten(value, new_key)
            else:
                env_vars[f"DTE_{prefix}"] = str(obj)
                
        flatten(self.config)
        return env_vars
        
    def __str__(self) -> str:
        """String representation"""
        return json.dumps(self.config, indent=2)


def create_default_config(output_file: str = "deep_tree_echo_config.json"):
    """Create a default configuration file"""
    config = DeepTreeEchoConfig()
    config.config_file = output_file
    
    if config.save_config():
        print(f"Created default configuration: {output_file}")
        print("\nDefault configuration:")
        print(config)
        return True
    else:
        print(f"Failed to create configuration file: {output_file}")
        return False


def main():
    """Main entry point for configuration management"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Deep Tree Echo configuration management'
    )
    
    parser.add_argument(
        '--create-default',
        action='store_true',
        help='Create default configuration file'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate current configuration'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show current configuration'
    )
    
    parser.add_argument(
        '--config-file',
        default='deep_tree_echo_config.json',
        help='Configuration file path'
    )
    
    args = parser.parse_args()
    
    if args.create_default:
        create_default_config(args.config_file)
        
    elif args.validate:
        config = DeepTreeEchoConfig(args.config_file)
        if config.validate():
            print("✓ Configuration is valid")
        else:
            print("✗ Configuration has errors")
            
    elif args.show:
        config = DeepTreeEchoConfig(args.config_file)
        print(config)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
