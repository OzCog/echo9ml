"""
Test Suite for Phase 7: Production Deployment & Real-Time Optimization

Comprehensive testing of production deployment infrastructure, performance
monitoring, load balancing, and auto-scaling components.
"""

import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

# Import the components to test
try:
    from phase7_production_deployment import (
        ProductionDeploymentOrchestrator,
        RealTimePerformanceMonitor,
        CognitiveLoadBalancer,
        AutoScaler,
        ProductionConfigurationManager,
        DeploymentEnvironment,
        ServiceHealth,
        PerformanceMetrics,
        ServiceConfiguration
    )
    PHASE7_AVAILABLE = True
except ImportError as e:
    PHASE7_AVAILABLE = False
    print(f"Phase 7 components not available: {e}")

@unittest.skipUnless(PHASE7_AVAILABLE, "Phase 7 components not available")
class TestProductionDeploymentOrchestrator(unittest.TestCase):
    """Test production deployment orchestrator"""
    
    def setUp(self):
        """Set up test environment"""
        self.orchestrator = ProductionDeploymentOrchestrator(
            environment=DeploymentEnvironment.DEVELOPMENT
        )
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        self.assertIsNotNone(self.orchestrator)
        self.assertEqual(self.orchestrator.environment, DeploymentEnvironment.DEVELOPMENT)
        self.assertIsInstance(self.orchestrator.deployment_config, dict)
        self.assertGreater(len(self.orchestrator.deployment_config), 0)
    
    def test_deployment_config_creation(self):
        """Test deployment configuration creation"""
        config = self.orchestrator.deployment_config
        
        # Check required services exist
        required_services = [
            "cognitive_gateway", "reasoning_engine", "memory_manager",
            "attention_allocator", "tensor_processor"
        ]
        
        for service in required_services:
            self.assertIn(service, config)
            self.assertIsInstance(config[service], ServiceConfiguration)
            self.assertGreater(config[service].replicas, 0)
            self.assertGreater(config[service].port, 0)
    
    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    async def test_validate_deployment_environment(self, mock_disk, mock_memory, mock_cpu):
        """Test deployment environment validation"""
        # Mock sufficient resources
        mock_cpu.return_value = 8
        mock_memory.return_value.total = 16 * 1024**3  # 16GB
        mock_disk.return_value.free = 100 * 1024**3   # 100GB
        
        result = await self.orchestrator._validate_deployment_environment()
        self.assertTrue(result)
        
        # Mock insufficient resources
        mock_cpu.return_value = 2  # Below minimum
        result = await self.orchestrator._validate_deployment_environment()
        self.assertFalse(result)
    
    async def test_deploy_service(self):
        """Test individual service deployment"""
        config = ServiceConfiguration(
            service_name="test-service",
            replicas=2,
            cpu_limit=1.0,
            memory_limit_mb=512,
            port=9000
        )
        
        await self.orchestrator._deploy_service("test-service", config)
        
        # Verify service was deployed
        self.assertIn("test-service", self.orchestrator.services)
        service_info = self.orchestrator.services["test-service"]
        self.assertEqual(len(service_info["instances"]), 2)
        self.assertEqual(service_info["health_status"], ServiceHealth.HEALTHY)
    
    async def test_service_health_check(self):
        """Test service health checking"""
        # Deploy a test service first
        config = ServiceConfiguration("test-service", replicas=2)
        await self.orchestrator._deploy_service("test-service", config)
        
        # Check health of existing service
        health = await self.orchestrator._check_service_health("test-service")
        self.assertTrue(health)
        
        # Check health of non-existent service
        health = await self.orchestrator._check_service_health("non-existent")
        self.assertFalse(health)
    
    def test_get_deployment_status(self):
        """Test deployment status reporting"""
        # Add a test service
        self.orchestrator.services["test-service"] = {
            "status": "running",
            "health_status": ServiceHealth.HEALTHY,
            "instances": [{"id": "test-1"}],
            "start_time": time.time()
        }
        
        status = self.orchestrator.get_deployment_status()
        
        self.assertIn("environment", status)
        self.assertIn("services", status)
        self.assertIn("overall_health", status)
        self.assertIn("test-service", status["services"])

@unittest.skipUnless(PHASE7_AVAILABLE, "Phase 7 components not available")
class TestRealTimePerformanceMonitor(unittest.TestCase):
    """Test real-time performance monitoring"""
    
    def setUp(self):
        """Set up test environment"""
        self.monitor = RealTimePerformanceMonitor()
    
    def test_initialization(self):
        """Test monitor initialization"""
        self.assertIsNotNone(self.monitor)
        self.assertFalse(self.monitor.monitoring_active)
        self.assertEqual(len(self.monitor.metrics_history), 0)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.net_io_counters')
    @patch('psutil.disk_io_counters')
    async def test_collect_metrics(self, mock_disk, mock_net, mock_memory, mock_cpu):
        """Test metrics collection"""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value.percent = 60.0
        
        mock_net_counters = MagicMock()
        mock_net_counters.bytes_sent = 1000
        mock_net_counters.bytes_recv = 2000
        mock_net.return_value = mock_net_counters
        
        mock_disk_counters = MagicMock()
        mock_disk_counters.read_bytes = 3000
        mock_disk_counters.write_bytes = 4000
        mock_disk.return_value = mock_disk_counters
        
        metrics = await self.monitor._collect_metrics()
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.cpu_usage, 50.0)
        self.assertEqual(metrics.memory_usage, 60.0)
        self.assertGreater(metrics.cognitive_load, 0)
        self.assertGreater(metrics.attention_allocation_efficiency, 0)
    
    def test_performance_metrics_to_dict(self):
        """Test performance metrics serialization"""
        metrics = PerformanceMetrics(
            cpu_usage=75.0,
            memory_usage=80.0,
            network_io=(1000, 2000),
            cognitive_load=0.6
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertIn("cpu_usage", metrics_dict)
        self.assertIn("memory_usage", metrics_dict)
        self.assertIn("cognitive_load", metrics_dict)
        self.assertEqual(metrics_dict["cpu_usage"], 75.0)
        self.assertEqual(metrics_dict["network_io"]["sent"], 1000)
    
    def test_get_active_alerts(self):
        """Test alert generation"""
        # Add high-load metrics
        high_load_metrics = PerformanceMetrics(
            cpu_usage=95.0,
            memory_usage=90.0,
            cognitive_load=0.9,
            error_rate=0.15
        )
        self.monitor.metrics_history.append(high_load_metrics)
        
        alerts = self.monitor._get_active_alerts()
        
        self.assertGreater(len(alerts), 0)
        self.assertTrue(any("High CPU usage" in alert for alert in alerts))
        self.assertTrue(any("High memory usage" in alert for alert in alerts))
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Add some test metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                cpu_usage=50.0 + i * 10,
                memory_usage=40.0 + i * 5,
                cognitive_load=0.5 + i * 0.1
            )
            self.monitor.metrics_history.append(metrics)
        
        summary = self.monitor.get_performance_summary()
        
        self.assertIn("current_metrics", summary)
        self.assertIn("averages_5min", summary)
        self.assertIn("alerts", summary)

@unittest.skipUnless(PHASE7_AVAILABLE, "Phase 7 components not available")
class TestCognitiveLoadBalancer(unittest.TestCase):
    """Test cognitive load balancer"""
    
    def setUp(self):
        """Set up test environment"""
        self.load_balancer = CognitiveLoadBalancer()
    
    def test_initialization(self):
        """Test load balancer initialization"""
        self.assertIsNotNone(self.load_balancer)
        self.assertFalse(self.load_balancer.balancing_active)
        self.assertIsInstance(self.load_balancer.service_loads, dict)
        self.assertIsInstance(self.load_balancer.routing_table, dict)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    async def test_update_service_loads(self, mock_memory, mock_cpu):
        """Test service load updates"""
        mock_cpu.return_value = 60.0
        mock_memory.return_value.percent = 70.0
        
        await self.load_balancer._update_service_loads()
        
        self.assertGreater(len(self.load_balancer.service_loads), 0)
        for service, load in self.load_balancer.service_loads.items():
            self.assertGreaterEqual(load, 0.0)
            self.assertLessEqual(load, 1.0)
    
    async def test_optimize_routing(self):
        """Test routing optimization"""
        # Set up some service loads
        self.load_balancer.service_loads = {
            "reasoning_engine_1": 0.3,
            "reasoning_engine_2": 0.7,
            "memory_manager_1": 0.5,
            "memory_manager_2": 0.2
        }
        
        await self.load_balancer._optimize_routing()
        
        # Check that routing table was created
        self.assertIn("reasoning", self.load_balancer.routing_table)
        self.assertIn("memory", self.load_balancer.routing_table)
        
        # Check that services are sorted by load
        reasoning_services = self.load_balancer.routing_table["reasoning"]
        if len(reasoning_services) > 1:
            self.assertTrue(
                self.load_balancer.service_loads[reasoning_services[0]] <= 
                self.load_balancer.service_loads[reasoning_services[1]]
            )
    
    def test_route_request(self):
        """Test request routing"""
        # Set up routing table
        self.load_balancer.routing_table = {
            "reasoning": ["reasoning_engine_1", "reasoning_engine_2"],
            "memory": ["memory_manager_1"]
        }
        
        # Test valid routing
        service = self.load_balancer.route_request("reasoning")
        self.assertEqual(service, "reasoning_engine_1")
        
        service = self.load_balancer.route_request("memory")
        self.assertEqual(service, "memory_manager_1")
        
        # Test invalid routing
        service = self.load_balancer.route_request("nonexistent")
        self.assertIsNone(service)
    
    def test_get_load_balance_status(self):
        """Test load balance status reporting"""
        self.load_balancer.service_loads = {"test_service": 0.5}
        self.load_balancer.routing_table = {"test": ["test_service"]}
        self.load_balancer.balancing_active = True
        
        status = self.load_balancer.get_load_balance_status()
        
        self.assertIn("service_loads", status)
        self.assertIn("routing_table", status)
        self.assertIn("balancing_active", status)
        self.assertTrue(status["balancing_active"])

@unittest.skipUnless(PHASE7_AVAILABLE, "Phase 7 components not available")
class TestAutoScaler(unittest.TestCase):
    """Test auto-scaling system"""
    
    def setUp(self):
        """Set up test environment"""
        self.auto_scaler = AutoScaler()
    
    def test_initialization(self):
        """Test auto-scaler initialization"""
        self.assertIsNotNone(self.auto_scaler)
        self.assertFalse(self.auto_scaler.scaling_active)
        self.assertIsInstance(self.auto_scaler.scaling_policies, dict)
        self.assertGreater(len(self.auto_scaler.scaling_policies), 0)
    
    def test_scaling_policies(self):
        """Test scaling policy configuration"""
        policies = self.auto_scaler.scaling_policies
        
        for service, policy in policies.items():
            self.assertIn("min_replicas", policy)
            self.assertIn("max_replicas", policy)
            self.assertIn("target_cpu_utilization", policy)
            self.assertIn("scale_up_threshold", policy)
            self.assertIn("scale_down_threshold", policy)
            
            self.assertGreaterEqual(policy["min_replicas"], 1)
            self.assertGreater(policy["max_replicas"], policy["min_replicas"])
    
    def test_calculate_recommended_replicas(self):
        """Test replica count calculation"""
        policy = {
            "min_replicas": 1,
            "max_replicas": 5,
            "target_cpu_utilization": 70,
            "target_cognitive_load": 0.7,
            "scale_up_threshold": 80,
            "scale_down_threshold": 40
        }
        
        # Test scale up scenario
        high_load_metrics = {
            "cpu_usage": 85.0,
            "cognitive_load": 0.8
        }
        
        recommended = self.auto_scaler._calculate_recommended_replicas(
            "test_service", high_load_metrics, policy
        )
        self.assertGreaterEqual(recommended, 2)  # Should scale up
        
        # Test scale down scenario
        low_load_metrics = {
            "cpu_usage": 30.0,
            "cognitive_load": 0.3
        }
        
        recommended = self.auto_scaler._calculate_recommended_replicas(
            "test_service", low_load_metrics, policy
        )
        self.assertLessEqual(recommended, 2)  # Should scale down or stay same
    
    def test_get_scaling_reason(self):
        """Test scaling reason generation"""
        policy = {
            "scale_up_threshold": 80,
            "scale_down_threshold": 40,
            "target_cognitive_load": 0.7
        }
        
        # High CPU reason
        high_cpu_metrics = {"cpu_usage": 85.0, "cognitive_load": 0.5}
        reason = self.auto_scaler._get_scaling_reason(high_cpu_metrics, policy)
        self.assertIn("High CPU usage", reason)
        
        # Low CPU reason
        low_cpu_metrics = {"cpu_usage": 30.0, "cognitive_load": 0.3}
        reason = self.auto_scaler._get_scaling_reason(low_cpu_metrics, policy)
        self.assertIn("Low CPU usage", reason)
    
    def test_get_scaling_status(self):
        """Test scaling status reporting"""
        # Add some test scaling history
        test_action = {
            "timestamp": time.time(),
            "service": "test_service",
            "action": "scale_up",
            "from_replicas": 2,
            "to_replicas": 3,
            "reason": "High CPU usage"
        }
        self.auto_scaler.scaling_history.append(test_action)
        
        status = self.auto_scaler.get_scaling_status()
        
        self.assertIn("scaling_active", status)
        self.assertIn("policies", status)
        self.assertIn("recent_actions", status)
        self.assertIn("total_scaling_actions", status)
        self.assertEqual(status["total_scaling_actions"], 1)

@unittest.skipUnless(PHASE7_AVAILABLE, "Phase 7 components not available")
class TestProductionConfigurationManager(unittest.TestCase):
    """Test production configuration manager"""
    
    def setUp(self):
        """Set up test environment"""
        # Use a temporary file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        self.config_manager = ProductionConfigurationManager(self.config_path)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test configuration manager initialization"""
        self.assertIsNotNone(self.config_manager)
        self.assertIsInstance(self.config_manager.config, dict)
        self.assertGreater(len(self.config_manager.config), 0)
    
    def test_default_configuration_creation(self):
        """Test default configuration creation"""
        config = self.config_manager.config
        
        # Check required sections exist
        required_sections = [
            "deployment", "monitoring", "scaling", "security", "performance"
        ]
        
        for section in required_sections:
            self.assertIn(section, config)
            self.assertIsInstance(config[section], dict)
    
    def test_configuration_persistence(self):
        """Test configuration saving and loading"""
        # Modify configuration
        original_log_level = self.config_manager.config["monitoring"]["log_level"]
        self.config_manager.config["monitoring"]["log_level"] = "DEBUG"
        self.config_manager.save_configuration()
        
        # Create new manager to test loading
        new_manager = ProductionConfigurationManager(self.config_path)
        self.assertEqual(new_manager.config["monitoring"]["log_level"], "DEBUG")
        self.assertNotEqual(new_manager.config["monitoring"]["log_level"], original_log_level)
    
    def test_configuration_updates(self):
        """Test configuration updates"""
        updates = {
            "new_setting": "test_value",
            "log_level": "WARNING"
        }
        
        self.config_manager.update_configuration("monitoring", updates)
        
        monitoring_config = self.config_manager.get_configuration("monitoring")
        self.assertEqual(monitoring_config["new_setting"], "test_value")
        self.assertEqual(monitoring_config["log_level"], "WARNING")
    
    def test_get_configuration(self):
        """Test configuration retrieval"""
        # Get entire configuration
        full_config = self.config_manager.get_configuration()
        self.assertIsInstance(full_config, dict)
        self.assertIn("deployment", full_config)
        
        # Get specific section
        deployment_config = self.config_manager.get_configuration("deployment")
        self.assertIsInstance(deployment_config, dict)
        self.assertIn("environment", deployment_config)
        
        # Get non-existent section
        empty_config = self.config_manager.get_configuration("nonexistent")
        self.assertEqual(empty_config, {})

class TestPhase7Integration(unittest.TestCase):
    """Integration tests for Phase 7 components"""
    
    def setUp(self):
        """Set up integration test environment"""
        if not PHASE7_AVAILABLE:
            self.skipTest("Phase 7 components not available")
    
    async def test_full_deployment_simulation(self):
        """Test full deployment simulation"""
        # Initialize orchestrator
        orchestrator = ProductionDeploymentOrchestrator(
            environment=DeploymentEnvironment.DEVELOPMENT
        )
        
        # Mock system resources to pass validation
        with patch('psutil.cpu_count', return_value=8), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value.total = 16 * 1024**3
            mock_disk.return_value.free = 100 * 1024**3
            
            # Test deployment validation
            validation_result = await orchestrator._validate_deployment_environment()
            self.assertTrue(validation_result)
    
    async def test_monitoring_integration(self):
        """Test monitoring system integration"""
        monitor = RealTimePerformanceMonitor()
        
        # Mock system metrics
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.net_io_counters') as mock_net, \
             patch('psutil.disk_io_counters') as mock_disk:
            
            # Set up mocks
            mock_memory.return_value.percent = 60.0
            mock_net.return_value.bytes_sent = 1000
            mock_net.return_value.bytes_recv = 2000
            mock_disk.return_value.read_bytes = 3000
            mock_disk.return_value.write_bytes = 4000
            
            # Collect metrics
            metrics = await monitor._collect_metrics()
            self.assertIsInstance(metrics, PerformanceMetrics)
            
            # Test metrics analysis
            monitor.metrics_history.append(metrics)
            await monitor._analyze_performance_trends()
    
    def test_load_balancer_integration(self):
        """Test load balancer integration"""
        load_balancer = CognitiveLoadBalancer()
        
        # Set up test scenario
        load_balancer.service_loads = {
            "reasoning_engine_1": 0.3,
            "reasoning_engine_2": 0.8,
            "memory_manager_1": 0.5
        }
        
        # Test routing
        asyncio.run(load_balancer._optimize_routing())
        
        # Verify routing works
        service = load_balancer.route_request("reasoning")
        self.assertIsNotNone(service)
        self.assertIn("reasoning_engine", service)
    
    def test_configuration_integration(self):
        """Test configuration system integration"""
        temp_dir = tempfile.mkdtemp()
        config_path = os.path.join(temp_dir, "integration_test_config.json")
        
        try:
            # Create configuration manager
            config_manager = ProductionConfigurationManager(config_path)
            
            # Test configuration flow
            original_config = config_manager.get_configuration()
            self.assertIsInstance(original_config, dict)
            
            # Update configuration
            config_manager.update_configuration("deployment", {
                "test_setting": "integration_test"
            })
            
            # Verify update
            deployment_config = config_manager.get_configuration("deployment")
            self.assertEqual(deployment_config["test_setting"], "integration_test")
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)

# Test runner for async tests
class AsyncTestRunner:
    """Test runner for async test methods"""
    
    @staticmethod
    def run_async_test(test_method):
        """Run an async test method"""
        return asyncio.run(test_method())

# Main test execution
def run_phase7_tests():
    """Run all Phase 7 tests"""
    if not PHASE7_AVAILABLE:
        print("Phase 7 components not available - skipping tests")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestProductionDeploymentOrchestrator,
        TestRealTimePerformanceMonitor,
        TestCognitiveLoadBalancer,
        TestAutoScaler,
        TestProductionConfigurationManager,
        TestPhase7Integration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_phase7_tests()
    print(f"\nPhase 7 Tests {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)