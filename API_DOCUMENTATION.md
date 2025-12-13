# Deep Tree Echo System - API Documentation

Complete API reference for all components of the Deep Tree Echo multi-language persona system.

## Table of Contents

1. [C++ Orchestrator API](#cpp-orchestrator-api)
2. [Go Execution Engine API](#go-execution-engine-api)
3. [Python Integration API](#python-integration-api)
4. [Configuration API](#configuration-api)
5. [Monitoring API](#monitoring-api)

---

## C++ Orchestrator API

### DeepTreeEchoOrchestrator Class

Core orchestration class for neural tree processing and echo propagation.

#### Constructor

```cpp
DeepTreeEchoOrchestrator(double echo_threshold = 0.75, int max_depth = 10)
```

**Parameters:**
- `echo_threshold`: Minimum echo value to propagate (0.0 - 1.0)
- `max_depth`: Maximum tree depth for propagation

**Example:**
```cpp
DeepTreeEchoOrchestrator orchestrator(0.75, 10);
```

#### Methods

##### create_cognitive_tree()

```cpp
void create_cognitive_tree()
```

Creates the initial cognitive tree structure with predefined nodes.

**Nodes Created:**
- Root: "Deep Tree Echo - Recursive, Adaptive, Integrative"
- Echo State Networks integration
- P-System hierarchical structure
- Cognitive architecture bridging
- Memory management and retrieval
- Emotional state processing
- Spatial awareness and 3D context

##### propagate_echo()

```cpp
void propagate_echo(std::shared_ptr<TreeNode> node)
```

Recursively propagates echo values through the tree structure.

**Parameters:**
- `node`: Starting node for propagation

##### analyze_echo_patterns()

```cpp
EchoPatternMetrics analyze_echo_patterns()
```

Analyzes patterns across the neural tree.

**Returns:** `EchoPatternMetrics` containing:
- `echo_variance`: Variance in echo values
- `emotional_coherence`: Emotional state coherence
- `resonance_depth`: Depth of resonance in tree
- `spatial_distribution`: Spatial pattern distribution

##### integrate_with_llama()

```cpp
std::string integrate_with_llama(const std::string& prompt)
```

Integrates with LLAMA inference engine for cognitive processing.

**Parameters:**
- `prompt`: Input prompt for inference

**Returns:** Inference result string

---

## Go Execution Engine API

### HyperEchoEngine

High-performance concurrent execution engine with WebSocket support.

#### Initialization

```go
engine := NewHyperEchoEngine(workerCount int)
```

**Parameters:**
- `workerCount`: Number of concurrent workers (default: 4)

**Example:**
```go
engine := NewHyperEchoEngine(4)
engine.Start()
```

#### WebSocket Endpoints

##### Connection

```
ws://localhost:8080/ws
```

Connect to the WebSocket server for real-time communication.

**Example (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

##### Send Command

Send JSON messages to execute commands:

```json
{
    "type": "execute",
    "node_id": "cognitive",
    "content": "Process cognitive patterns",
    "priority": 1
}
```

**Message Fields:**
- `type`: Message type ("execute", "query", "status")
- `node_id`: Target node identifier
- `content`: Command content
- `priority`: Execution priority (0-10)

##### Status Response

Server sends status updates:

```json
{
    "type": "status",
    "timestamp": 1234567890,
    "workers": 4,
    "queue_length": 2,
    "total_nodes": 4,
    "average_echo": 0.75
}
```

#### HTTP Endpoints

##### GET /status

Get current engine status.

**Response:**
```json
{
    "is_running": true,
    "workers": 4,
    "queue_length": 0,
    "total_nodes": 4,
    "average_echo": 0.723,
    "active_connections": 2
}
```

#### Methods

##### QueueNode

```go
func (e *HyperEchoEngine) QueueNode(node *EchoNode) error
```

Queue a node for execution.

**Parameters:**
- `node`: EchoNode to execute

**Returns:** Error if queue is full

##### AnalyzePatterns

```go
func (e *HyperEchoEngine) AnalyzePatterns() map[string]float64
```

Analyze hyper-patterns across all nodes.

**Returns:** Map of pattern metrics:
- `cognitive_load`
- `execution_density`
- `resonance_coherence`
- `spatial_distribution`
- `emotional_harmony`
- `temporal_flow`

---

## Python Integration API

### MultiLanguageOrchestrator Class

Unified orchestration interface for managing all system components.

#### Initialization

```python
from deep_tree_echo_integration import MultiLanguageOrchestrator

orchestrator = MultiLanguageOrchestrator()
```

#### Methods

##### start_monitoring_system()

```python
async def start_monitoring_system() -> Dict[str, Any]
```

Start all system components and monitoring.

**Returns:** Status dictionary

**Example:**
```python
import asyncio

async def main():
    orchestrator = MultiLanguageOrchestrator()
    status = await orchestrator.start_monitoring_system()
    print(status)

asyncio.run(main())
```

##### get_system_status()

```python
async def get_system_status() -> Dict[str, Any]
```

Get current status of all components.

**Returns:**
```python
{
    'timestamp': 1234567890.0,
    'components': {
        'cpp': {'status': 'running', 'pid': 1234},
        'go': {'status': 'running', 'pid': 1235}
    },
    'health': 'healthy'
}
```

##### stop_all_components()

```python
async def stop_all_components()
```

Gracefully stop all running components.

---

## Configuration API

### DeepTreeEchoConfig Class

Configuration management system.

#### Initialization

```python
from config_manager import DeepTreeEchoConfig

config = DeepTreeEchoConfig("config.json")
```

#### Methods

##### get()

```python
def get(key: str, default: Any = None) -> Any
```

Get configuration value using dot notation.

**Example:**
```python
echo_threshold = config.get('cpp.echo_threshold')  # 0.75
port = config.get('go.websocket_port')  # 8080
```

##### set()

```python
def set(key: str, value: Any) -> bool
```

Set configuration value.

**Example:**
```python
config.set('go.worker_count', 8)
config.save_config()
```

##### validate()

```python
def validate() -> bool
```

Validate configuration for correctness.

**Returns:** True if valid, False otherwise

**Example:**
```python
if config.validate():
    print("Configuration is valid")
else:
    print("Configuration has errors")
```

---

## Monitoring API

### DeepTreeEchoMonitor Class

System monitoring and health checking.

#### Initialization

```python
from monitoring_system import DeepTreeEchoMonitor

monitor = DeepTreeEchoMonitor()
```

#### Methods

##### monitor_loop()

```python
async def monitor_loop(
    components: Dict[str, Optional[int]],
    interval: int = 5
)
```

Start monitoring loop.

**Parameters:**
- `components`: Dictionary of component names to PIDs
- `interval`: Monitoring interval in seconds

**Example:**
```python
import asyncio

async def main():
    monitor = DeepTreeEchoMonitor()
    components = {'cpp': 1234, 'go': 1235}
    await monitor.monitor_loop(components, interval=5)

asyncio.run(main())
```

##### get_status_report()

```python
def get_status_report() -> Dict[str, Any]
```

Get comprehensive status report.

**Returns:**
```python
{
    'timestamp': 1234567890.0,
    'metrics': {
        'system': {
            'cpu_percent': 25.3,
            'memory_percent': 45.2,
            'disk_percent': 60.1
        },
        'components': {
            'cpp': {'cpu_percent': 10.5, 'memory_mb': 125.3},
            'go': {'cpu_percent': 15.2, 'memory_mb': 89.7}
        }
    },
    'health': {...}
}
```

---

## Launch API

### DeepTreeEchoLauncher Class

Unified system launcher.

#### Initialization

```python
from launch_unified_system import DeepTreeEchoLauncher

launcher = DeepTreeEchoLauncher(verbose=True)
```

#### Methods

##### run_demo()

```python
def run_demo() -> bool
```

Run demonstration mode (one-shot).

**Returns:** True if successful

**Example:**
```python
launcher = DeepTreeEchoLauncher()
success = launcher.run_demo()
```

##### run_services()

```python
def run_services() -> bool
```

Run all services continuously.

**Returns:** True if successful

**Example:**
```python
launcher = DeepTreeEchoLauncher()
launcher.run_services()
```

---

## Data Structures

### SpatialContext

3D spatial awareness structure.

```cpp
struct SpatialContext {
    std::tuple<double, double, double> position;
    std::tuple<double, double, double> orientation;
    double scale;
    double depth;
    double field_of_view;
    std::map<std::string, double> spatial_relations;
};
```

### EmotionalState

7-dimensional emotional vector.

```cpp
struct EmotionalState {
    std::array<double, 7> emotions;
    double dominance;
    double activation;
    double valence;
};
```

### EchoNode (Go)

```go
type EchoNode struct {
    ID             string
    Content        string
    EchoValue      float64
    Children       []*EchoNode
    Metadata       map[string]interface{}
    EmotionalState EmotionalState
    SpatialCtx     SpatialContext
    Priority       int
}
```

---

## Usage Examples

### Complete Workflow Example

```python
#!/usr/bin/env python3
"""Complete workflow example"""

import asyncio
from launch_unified_system import DeepTreeEchoLauncher
from config_manager import DeepTreeEchoConfig
from monitoring_system import DeepTreeEchoMonitor

async def main():
    # 1. Load configuration
    config = DeepTreeEchoConfig()
    if not config.validate():
        print("Configuration invalid!")
        return
        
    # 2. Create launcher
    launcher = DeepTreeEchoLauncher(verbose=True)
    
    # 3. Check prerequisites
    if not launcher.check_executables():
        print("Missing executables!")
        return
        
    # 4. Start monitoring
    monitor = DeepTreeEchoMonitor(config)
    
    # 5. Run system
    success = launcher.run_demo()
    
    if success:
        print("System running successfully!")
        
        # 6. Get status report
        report = monitor.get_status_report()
        print(f"Status: {report}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Error Handling

All APIs include error handling. Common error codes:

- **ENOENT**: Executable not found
- **EADDRINUSE**: Port already in use
- **ETIMEDOUT**: Operation timed out
- **ECONNREFUSED**: Connection refused

Example error handling:

```python
try:
    orchestrator = MultiLanguageOrchestrator()
    await orchestrator.start_monitoring_system()
except FileNotFoundError as e:
    print(f"Executable not found: {e}")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Version Information

- API Version: 1.0.0
- Last Updated: 2025-12-13
- Compatibility: Deep Tree Echo System v1.0.0
