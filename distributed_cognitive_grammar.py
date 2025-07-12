"""
Distributed Network of Agentic Cognitive Grammar for OpenCoq/echo9ml

This module implements the distributed cognitive grammar system as specified in the issue:
- Distributed Agentic Kernel (Echo9ML Node)
- Hypergraph Representation (AtomSpace Integration)
- GGML Tensor Kernel (Custom Shapes)
- Communication Substrate (Async Messaging/IPC)
- Attention Allocation (ECAN-inspired Module)
- Symbolic Reasoning (PLN/Pattern Matcher)
- Adaptive Learning (MOSES Evolutionary Search)

Based on the architectural specification and existing components.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from collections import defaultdict, deque

# Import existing components
try:
    from swarmprotocol import MessageBroker, RLAgent
    from memory_management import HypergraphMemory, MemoryNode, MemoryType
    from echoself_introspection import AdaptiveAttentionAllocator
    from ecan_attention_allocator import ECANAttentionAllocator, ResourceType, TaskPriority
    IMPORTS_AVAILABLE = True
    ECAN_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    ECAN_AVAILABLE = False
    logging.warning("Some imports not available, using mock implementations")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of cognitive agents in the distributed network"""
    ECHO9ML_NODE = "echo9ml_node"
    HYPERGRAPH_MANAGER = "hypergraph_manager"
    TENSOR_KERNEL = "tensor_kernel"
    ATTENTION_ALLOCATOR = "attention_allocator"
    SYMBOLIC_REASONER = "symbolic_reasoner"
    ADAPTIVE_LEARNER = "adaptive_learner"

class MessageType(Enum):
    """Types of messages in the cognitive grammar network"""
    HYPERGRAPH_FRAGMENT = "hypergraph_fragment"
    TENSOR_UPDATE = "tensor_update"
    ATTENTION_ALLOCATION = "attention_allocation"
    REASONING_QUERY = "reasoning_query"
    REASONING_RESULT = "reasoning_result"
    LEARNING_UPDATE = "learning_update"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"

@dataclass
class HypergraphFragment:
    """Hypergraph knowledge fragment for exchange between agents"""
    id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source_agent: str = ""
    semantic_weight: float = 1.0

@dataclass
class TensorShape:
    """GGML tensor shape specification"""
    dimensions: Tuple[int, ...]
    dtype: str = "float32"
    semantic_mapping: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.semantic_mapping:
            # Default semantic mapping for cognitive dimensions
            self.semantic_mapping = {
                "persona_id": 0,
                "trait_id": 1,
                "time": 2,
                "context": 3,
                "valence": 4
            }

@dataclass
class CognitiveMessage:
    """Message structure for inter-agent communication"""
    message_id: str
    message_type: MessageType
    sender_id: str
    receiver_id: Optional[str] = None  # None for broadcast
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # 1=low, 5=high
    requires_response: bool = False

class DistributedCognitiveAgent:
    """Base class for distributed cognitive agents"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, 
                 broker: Optional[Any] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.broker = broker or (MessageBroker() if IMPORTS_AVAILABLE else None)
        self.inbox = None
        self.hypergraph_memory = HypergraphMemory() if IMPORTS_AVAILABLE else None
        self.tensor_shapes: Dict[str, TensorShape] = {}
        
        # Enhanced ECAN Attention Allocation
        if ECAN_AVAILABLE:
            self.ecan_allocator = ECANAttentionAllocator(agent_id, initial_av=1000.0)
            self.attention_allocator = None
            self.logger.info(f"Initialized ECAN attention allocator for {agent_id}")
        else:
            self.attention_allocator = AdaptiveAttentionAllocator() if IMPORTS_AVAILABLE else None
            self.ecan_allocator = None
            
        self.running = False
        self.last_heartbeat = time.time()
        self.peers: Set[str] = set()
        
        # Initialize basic tensor shapes
        self._initialize_tensor_shapes()
        
        logger.info(f"Initialized {agent_type.value} agent: {agent_id}")
    
    def _initialize_tensor_shapes(self):
        """Initialize default tensor shapes for cognitive operations"""
        # Persona tensor shape (prime factorization for flexibility)
        self.tensor_shapes["persona"] = TensorShape(
            dimensions=(3, 7, 13, 5, 2),  # personas x traits x time x context x valence
            semantic_mapping={
                "persona_id": 0,
                "trait_id": 1, 
                "time": 2,
                "context": 3,
                "valence": 4
            }
        )
        
        # Attention tensor shape
        self.tensor_shapes["attention"] = TensorShape(
            dimensions=(10, 10),  # source_nodes x target_nodes
            semantic_mapping={
                "source": 0,
                "target": 1
            }
        )
        
        # Memory tensor shape
        self.tensor_shapes["memory"] = TensorShape(
            dimensions=(100, 8, 5),  # memory_nodes x memory_types x salience_levels
            semantic_mapping={
                "memory_node": 0,
                "memory_type": 1,
                "salience": 2
            }
        )
    
    async def start(self):
        """Start the cognitive agent"""
        self.running = True
        if self.broker:
            self.inbox = self.broker.subscribe()
        
        # Start main processing loop
        tasks = [
            self._process_messages(),
            self._heartbeat_loop(),
            self._cognitive_processing_loop()
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the cognitive agent"""
        self.running = False
        logger.info(f"Stopping agent {self.agent_id}")
    
    async def _process_messages(self):
        """Process incoming messages"""
        while self.running:
            try:
                if self.inbox:
                    message_data = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
                    message = json.loads(message_data)
                    await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await asyncio.sleep(0.1)
    
    async def _handle_message(self, message_data: Dict[str, Any]):
        """Handle incoming message"""
        try:
            message = CognitiveMessage(**message_data)
            
            # Skip own messages
            if message.sender_id == self.agent_id:
                return
            
            # Add sender to peers
            self.peers.add(message.sender_id)
            
            # Route message based on type
            if message.message_type == MessageType.HYPERGRAPH_FRAGMENT:
                await self._handle_hypergraph_fragment(message)
            elif message.message_type == MessageType.TENSOR_UPDATE:
                await self._handle_tensor_update(message)
            elif message.message_type == MessageType.ATTENTION_ALLOCATION:
                await self._handle_attention_allocation(message)
            elif message.message_type == MessageType.REASONING_QUERY:
                await self._handle_reasoning_query(message)
            elif message.message_type == MessageType.REASONING_RESULT:
                await self._handle_reasoning_result(message)
            elif message.message_type == MessageType.LEARNING_UPDATE:
                await self._handle_learning_update(message)
            elif message.message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(message)
            elif message.message_type == MessageType.DISCOVERY:
                await self._handle_discovery(message)
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_hypergraph_fragment(self, message: CognitiveMessage):
        """Handle hypergraph fragment update"""
        if not self.hypergraph_memory:
            return
            
        fragment_data = message.payload.get("fragment", {})
        fragment = HypergraphFragment(**fragment_data)
        
        # Integrate fragment into local hypergraph
        for node_data in fragment.nodes:
            if "id" in node_data and "content" in node_data:
                node = MemoryNode(
                    id=node_data["id"],
                    content=node_data["content"],
                    memory_type=MemoryType.SEMANTIC,
                    source=f"agent_{fragment.source_agent}",
                    salience=node_data.get("salience", 0.5)
                )
                self.hypergraph_memory.add_node(node)
        
        logger.info(f"Integrated hypergraph fragment from {fragment.source_agent}")
    
    async def _handle_tensor_update(self, message: CognitiveMessage):
        """Handle tensor update message"""
        tensor_data = message.payload.get("tensor", {})
        tensor_type = tensor_data.get("type", "unknown")
        
        # Update local tensor shapes if needed
        if tensor_type in self.tensor_shapes:
            shape_data = tensor_data.get("shape", {})
            if shape_data:
                self.tensor_shapes[tensor_type] = TensorShape(**shape_data)
        
        logger.info(f"Updated tensor {tensor_type} from {message.sender_id}")
    
    async def _handle_attention_allocation(self, message: CognitiveMessage):
        """Handle attention allocation message with ECAN support"""
        allocation_data = message.payload.get("allocation", {})
        
        if self.ecan_allocator:
            # ECAN-based attention handling
            if "attention_atoms" in allocation_data:
                # Process shared attention atoms
                for atom_data in allocation_data["attention_atoms"]:
                    self.ecan_allocator.create_attention_atom(
                        content=atom_data.get("content", f"shared_from_{message.sender_id}"),
                        initial_sti=atom_data.get("sti", 0.3),
                        initial_av=atom_data.get("av", 5.0)
                    )
                    
            if "resource_bids" in allocation_data:
                # Process resource sharing requests
                for bid_data in allocation_data["resource_bids"]:
                    # Consider external bids for resource trading
                    logger.info(f"Received resource bid from {message.sender_id}: {bid_data}")
                    
            # Run an ECAN attention cycle to process new information
            await self.ecan_allocator.run_attention_cycle()
            
        elif self.attention_allocator:
            # Fallback to basic attention allocation
            current_load = allocation_data.get("load", 0.5)
            recent_activity = allocation_data.get("activity", 0.5)
            
            threshold = self.attention_allocator.adaptive_attention(
                current_load, recent_activity
            )
            
            logger.info(f"Updated attention threshold to {threshold:.3f}")
        
        logger.debug(f"Processed attention allocation from {message.sender_id}")
    
    async def _handle_reasoning_query(self, message: CognitiveMessage):
        """Handle reasoning query"""
        query_data = message.payload.get("query", {})
        
        # Simple pattern matching for now
        if self.hypergraph_memory:
            results = self.hypergraph_memory.search_nodes(
                query_data.get("pattern", ""),
                max_results=query_data.get("max_results", 10)
            )
            
            # Send result back
            response = CognitiveMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.REASONING_RESULT,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                payload={
                    "query_id": message.message_id,
                    "results": [node.to_dict() for node in results]
                }
            )
            
            await self._send_message(response)
    
    async def _handle_reasoning_result(self, message: CognitiveMessage):
        """Handle reasoning result"""
        results = message.payload.get("results", [])
        logger.info(f"Received {len(results)} reasoning results from {message.sender_id}")
    
    async def _handle_learning_update(self, message: CognitiveMessage):
        """Handle learning update"""
        update_data = message.payload.get("update", {})
        learning_type = update_data.get("type", "unknown")
        
        logger.info(f"Received learning update ({learning_type}) from {message.sender_id}")
    
    async def _handle_heartbeat(self, message: CognitiveMessage):
        """Handle heartbeat message"""
        self.last_heartbeat = time.time()
        logger.debug(f"Heartbeat from {message.sender_id}")
    
    async def _handle_discovery(self, message: CognitiveMessage):
        """Handle peer discovery"""
        peer_info = message.payload.get("peer_info", {})
        peer_id = peer_info.get("agent_id", "unknown")
        
        self.peers.add(peer_id)
        logger.info(f"Discovered peer agent: {peer_id}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self.running:
            try:
                heartbeat = CognitiveMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.HEARTBEAT,
                    sender_id=self.agent_id,
                    payload={
                        "timestamp": time.time(),
                        "agent_type": self.agent_type.value,
                        "status": "active"
                    }
                )
                
                await self._send_message(heartbeat)
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _cognitive_processing_loop(self):
        """Main cognitive processing loop"""
        while self.running:
            try:
                # Agent-specific processing
                await self._process_cognitive_state()
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in cognitive processing: {e}")
                await asyncio.sleep(1)
    
    async def _process_cognitive_state(self):
        """Process cognitive state with ECAN support"""
        if self.ecan_allocator:
            # Run ECAN attention cycle
            cycle_result = await self.ecan_allocator.run_attention_cycle()
            
            # Share high-attention atoms with other agents occasionally
            if cycle_result["attention_spreads"] > 0 and len(self.peers) > 0:
                await self._share_attention_atoms()
                
            # Share resource availability if needed
            if cycle_result["allocations_processed"] > 0:
                await self._share_resource_status()
    
    async def _share_attention_atoms(self):
        """Share high-attention atoms with other agents"""
        if not self.ecan_allocator or not self.peers:
            return
            
        # Find high-attention atoms to share
        high_attention_atoms = [
            {
                "atom_id": atom.atom_id,
                "content": str(atom.content),
                "sti": atom.sti,
                "av": atom.av,
                "age": atom.age
            }
            for atom in self.ecan_allocator.attention_atoms.values()
            if atom.sti > 0.5  # Only share high-attention atoms
        ]
        
        if high_attention_atoms:
            attention_message = CognitiveMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ATTENTION_ALLOCATION,
                sender_id=self.agent_id,
                payload={
                    "allocation": {
                        "attention_atoms": high_attention_atoms,
                        "timestamp": time.time(),
                        "sharing_reason": "high_attention_spread"
                    }
                }
            )
            
            await self._send_message(attention_message)
            logger.debug(f"Shared {len(high_attention_atoms)} attention atoms with network")
    
    async def _share_resource_status(self):
        """Share resource availability status with other agents"""
        if not self.ecan_allocator or not self.peers:
            return
            
        metrics = self.ecan_allocator.get_performance_metrics()
        
        resource_message = CognitiveMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ATTENTION_ALLOCATION,
            sender_id=self.agent_id,
            payload={
                "allocation": {
                    "resource_utilization": metrics["resource_utilization"],
                    "available_av": metrics["current_av"],
                    "performance_metrics": {
                        "task_completion_rate": metrics["total_tasks_completed"],
                        "av_efficiency": metrics["av_efficiency"],
                        "resource_efficiency": metrics["resource_efficiency"]
                    },
                    "timestamp": time.time()
                }
            }
        )
        
        await self._send_message(resource_message)
        logger.debug("Shared resource status with network")
    
    async def _send_message(self, message: CognitiveMessage):
        """Send message to other agents"""
        if self.broker:
            message_data = {
                "message_id": message.message_id,
                "message_type": message.message_type.value,
                "sender_id": message.sender_id,
                "receiver_id": message.receiver_id,
                "payload": message.payload,
                "timestamp": message.timestamp,
                "priority": message.priority,
                "requires_response": message.requires_response
            }
            
            await self.broker.publish(message_data)
    
    async def broadcast_hypergraph_fragment(self, fragment: HypergraphFragment):
        """Broadcast hypergraph fragment to network"""
        message = CognitiveMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HYPERGRAPH_FRAGMENT,
            sender_id=self.agent_id,
            payload={
                "fragment": {
                    "id": fragment.id,
                    "nodes": fragment.nodes,
                    "edges": fragment.edges,
                    "metadata": fragment.metadata,
                    "timestamp": fragment.timestamp,
                    "source_agent": fragment.source_agent,
                    "semantic_weight": fragment.semantic_weight
                }
            }
        )
        
        await self._send_message(message)
    
    async def query_distributed_reasoning(self, pattern: str, max_results: int = 10):
        """Query distributed reasoning network"""
        message = CognitiveMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REASONING_QUERY,
            sender_id=self.agent_id,
            payload={
                "query": {
                    "pattern": pattern,
                    "max_results": max_results
                }
            },
            requires_response=True
        )
        
        await self._send_message(message)

class Echo9MLNode(DistributedCognitiveAgent):
    """Main Echo9ML cognitive agent node"""
    
    def __init__(self, agent_id: str, broker: Optional[Any] = None):
        super().__init__(agent_id, AgentType.ECHO9ML_NODE, broker)
        self.persona_data = {}
        self.evolution_history = []
    
    async def _process_cognitive_state(self):
        """Process Echo9ML node cognitive state"""
        # Update persona evolution
        current_time = time.time()
        
        # Create hypergraph fragment from current state
        if self.hypergraph_memory and len(self.hypergraph_memory.nodes) > 0:
            # Sample some nodes for sharing
            sample_nodes = list(self.hypergraph_memory.nodes.values())[:5]
            
            fragment = HypergraphFragment(
                id=str(uuid.uuid4()),
                nodes=[node.to_dict() for node in sample_nodes],
                edges=[],
                source_agent=self.agent_id,
                semantic_weight=0.8
            )
            
            await self.broadcast_hypergraph_fragment(fragment)

class DistributedCognitiveNetwork:
    """Manages the distributed cognitive grammar network"""
    
    def __init__(self):
        self.broker = MessageBroker() if IMPORTS_AVAILABLE else None
        self.agents: Dict[str, DistributedCognitiveAgent] = {}
        self.running = False
    
    def add_agent(self, agent: DistributedCognitiveAgent):
        """Add agent to network"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent {agent.agent_id} to network")
    
    async def start_network(self):
        """Start the distributed cognitive network"""
        self.running = True
        
        # Start all agents
        agent_tasks = []
        for agent in self.agents.values():
            agent_tasks.append(agent.start())
        
        # Start network coordination
        coordination_task = self._coordination_loop()
        
        # Run all tasks
        await asyncio.gather(*agent_tasks, coordination_task)
    
    async def stop_network(self):
        """Stop the distributed cognitive network"""
        self.running = False
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        logger.info("Stopped distributed cognitive network")
    
    async def _coordination_loop(self):
        """Network coordination loop"""
        while self.running:
            try:
                # Periodic network health checks
                await self._check_network_health()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(10)
    
    async def _check_network_health(self):
        """Check network health and connectivity"""
        active_agents = [agent for agent in self.agents.values() if agent.running]
        logger.info(f"Network health: {len(active_agents)}/{len(self.agents)} agents active")
        
        # Trigger discovery messages
        for agent in active_agents:
            discovery_message = CognitiveMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.DISCOVERY,
                sender_id=agent.agent_id,
                payload={
                    "peer_info": {
                        "agent_id": agent.agent_id,
                        "agent_type": agent.agent_type.value,
                        "peer_count": len(agent.peers)
                    }
                }
            )
            
            await agent._send_message(discovery_message)

# Example usage and integration
async def create_distributed_network():
    """Create a sample distributed cognitive network"""
    network = DistributedCognitiveNetwork()
    
    # Create different types of agents
    echo_node = Echo9MLNode("echo_node_1", network.broker)
    
    # Add agents to network
    network.add_agent(echo_node)
    
    return network

# Main execution
if __name__ == "__main__":
    async def main():
        logger.info("Starting distributed cognitive grammar network...")
        
        network = await create_distributed_network()
        
        try:
            await network.start_network()
        except KeyboardInterrupt:
            logger.info("Shutting down network...")
            await network.stop_network()
    
    asyncio.run(main())