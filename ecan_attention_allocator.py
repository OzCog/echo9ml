"""
ECAN-style Economic Attention Allocation and Resource Kernel

This module implements the Economic Cognitive Attention Network (ECAN) inspired
attention allocation system with bidding, trading, and resource scheduling for
the echo9ml distributed cognitive grammar network.

Key features:
- Economic bidding mechanisms for attention resources
- Dynamic resource trading between cognitive agents
- Real-world task scheduling with resource constraints
- Attention spreading with economic incentives
- Performance benchmarking and resource allocation tracking
"""

import asyncio
import time
import logging
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of cognitive resources in ECAN system"""
    ATTENTION = "attention"
    MEMORY = "memory" 
    PROCESSING = "processing"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    REASONING = "reasoning"

class TaskPriority(Enum):
    """Task priority levels for scheduling"""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1

@dataclass
class ResourceBid:
    """Bid for cognitive resources in ECAN market"""
    bid_id: str
    agent_id: str
    resource_type: ResourceType
    amount: float  # Resource units requested
    price: float   # Economic value offered
    priority: TaskPriority
    task_id: str
    timestamp: float = field(default_factory=time.time)
    deadline: Optional[float] = None

@dataclass
class AttentionAtom:
    """Cognitive atom with ECAN attention values"""
    atom_id: str
    content: Any
    sti: float = 0.0  # Short-term importance
    lti: float = 0.0  # Long-term importance
    vlti: float = 0.0  # Very long-term importance
    av: float = 100.0  # Activation value (economic currency)
    rent: float = 0.0  # Attention rent paid
    age: int = 0
    last_accessed: float = field(default_factory=time.time)

@dataclass
class CognitiveTask:
    """Real-world cognitive task for scheduling"""
    task_id: str
    task_type: str
    description: str
    required_resources: Dict[ResourceType, float]
    priority: TaskPriority
    deadline: Optional[float] = None
    estimated_duration: float = 1.0
    agent_id: str = ""
    status: str = "pending"  # pending, running, completed, failed
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    results: Dict[str, Any] = field(default_factory=dict)

class ECANAttentionAllocator:
    """
    ECAN-style Economic Attention Allocation system
    
    Implements economic mechanisms for attention allocation including:
    - Bidding and trading for attention resources
    - Dynamic pricing based on supply and demand
    - Attention spreading through economic incentives
    - Resource scheduling with economic constraints
    """
    
    def __init__(self, agent_id: str, initial_av: float = 1000.0):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
        # Economic parameters
        self.total_av = initial_av  # Total activation value available
        self.available_av = initial_av
        self.attention_atoms: Dict[str, AttentionAtom] = {}
        
        # Resource management
        self.resource_capacity: Dict[ResourceType, float] = {
            ResourceType.ATTENTION: 100.0,
            ResourceType.MEMORY: 100.0,
            ResourceType.PROCESSING: 100.0,
            ResourceType.COMMUNICATION: 50.0,
            ResourceType.LEARNING: 25.0,
            ResourceType.REASONING: 75.0
        }
        self.resource_usage: Dict[ResourceType, float] = {
            rt: 0.0 for rt in ResourceType
        }
        
        # Bidding and trading
        self.pending_bids: Dict[str, ResourceBid] = {}
        self.active_trades: Dict[str, Dict[str, Any]] = {}
        
        # Task scheduling
        self.task_queue: deque = deque()
        self.running_tasks: Dict[str, CognitiveTask] = {}
        self.completed_tasks: List[CognitiveTask] = []
        
        # Performance tracking
        self.metrics = {
            "total_tasks_completed": 0,
            "average_task_duration": 0.0,
            "resource_utilization": {rt.value: 0.0 for rt in ResourceType},
            "attention_spread_events": 0,
            "economic_transactions": 0,
            "av_earned": 0.0,
            "av_spent": 0.0
        }
        
        # Attention spreading configuration
        self.spreading_threshold = 0.1
        self.spreading_factor = 0.8
        self.rent_collection_interval = 1.0
        
        self.logger.info(f"Initialized ECAN attention allocator for {agent_id}")

    def create_attention_atom(self, content: Any, initial_sti: float = 0.0, 
                            initial_av: float = 10.0) -> AttentionAtom:
        """Create a new attention atom with economic properties"""
        atom_id = str(uuid.uuid4())
        atom = AttentionAtom(
            atom_id=atom_id,
            content=content,
            sti=initial_sti,
            av=initial_av
        )
        self.attention_atoms[atom_id] = atom
        self.logger.debug(f"Created attention atom {atom_id} with AV={initial_av}")
        return atom

    def update_attention_values(self, atom_id: str, sti_delta: float = 0.0, 
                              lti_delta: float = 0.0) -> bool:
        """Update attention values for an atom with economic constraints"""
        if atom_id not in self.attention_atoms:
            return False
            
        atom = self.attention_atoms[atom_id]
        
        # Economic cost for attention increases
        cost = max(0, sti_delta + lti_delta) * 0.1
        if cost > self.available_av:
            self.logger.warning(f"Insufficient AV for attention update: need {cost}, have {self.available_av}")
            return False
            
        atom.sti += sti_delta
        atom.lti += lti_delta
        atom.last_accessed = time.time()
        self.available_av -= cost
        
        self.logger.debug(f"Updated atom {atom_id}: STI={atom.sti:.2f}, LTI={atom.lti:.2f}, cost={cost:.2f}")
        return True

    def bid_for_resources(self, resource_type: ResourceType, amount: float, 
                         price: float, task_id: str, priority: TaskPriority,
                         deadline: Optional[float] = None) -> str:
        """Submit a bid for cognitive resources"""
        bid_id = str(uuid.uuid4())
        bid = ResourceBid(
            bid_id=bid_id,
            agent_id=self.agent_id,
            resource_type=resource_type,
            amount=amount,
            price=price,
            priority=priority,
            task_id=task_id,
            deadline=deadline
        )
        
        self.pending_bids[bid_id] = bid
        self.logger.info(f"Submitted bid {bid_id} for {amount} {resource_type.value} at price {price}")
        return bid_id

    def process_resource_allocation(self) -> Dict[str, bool]:
        """Process pending bids and allocate resources based on economic criteria"""
        allocations = {}
        
        # Sort bids by priority and price
        sorted_bids = sorted(
            self.pending_bids.values(),
            key=lambda b: (b.priority.value, b.price),
            reverse=True
        )
        
        for bid in sorted_bids:
            # Check resource availability
            available = self.resource_capacity[bid.resource_type] - self.resource_usage[bid.resource_type]
            
            if available >= bid.amount and self.available_av >= bid.price:
                # Allocate resources
                self.resource_usage[bid.resource_type] += bid.amount
                self.available_av -= bid.price
                allocations[bid.bid_id] = True
                
                # Remove from pending
                del self.pending_bids[bid.bid_id]
                
                self.metrics["economic_transactions"] += 1
                self.metrics["av_spent"] += bid.price
                
                self.logger.info(f"Allocated {bid.amount} {bid.resource_type.value} for bid {bid.bid_id}")
            else:
                allocations[bid.bid_id] = False
                self.logger.debug(f"Cannot allocate resources for bid {bid.bid_id}: insufficient resources or AV")
        
        return allocations

    def schedule_task(self, task: CognitiveTask) -> bool:
        """Schedule a cognitive task with resource requirements"""
        # Check if we have sufficient resources
        can_schedule = True
        for resource_type, required in task.required_resources.items():
            available = self.resource_capacity[resource_type] - self.resource_usage[resource_type]
            if available < required:
                can_schedule = False
                break
        
        if can_schedule:
            # Reserve resources
            for resource_type, required in task.required_resources.items():
                self.resource_usage[resource_type] += required
            
            task.agent_id = self.agent_id
            task.status = "running"
            task.started_at = time.time()
            self.running_tasks[task.task_id] = task
            
            self.logger.info(f"Scheduled task {task.task_id}: {task.description}")
            return True
        else:
            # Add to queue for later scheduling
            task.status = "pending"
            self.task_queue.append(task)
            self.logger.info(f"Queued task {task.task_id}: {task.description}")
            return False

    async def execute_task(self, task: CognitiveTask) -> Dict[str, Any]:
        """Execute a cognitive task with simulated processing"""
        start_time = time.time()
        
        # Simulate task execution with resource consumption
        await asyncio.sleep(min(task.estimated_duration, 0.1))  # Cap simulation time
        
        # Simulate different types of cognitive tasks
        results = {}
        if task.task_type == "reasoning":
            results = {
                "reasoning_steps": np.random.randint(3, 10),
                "conclusion_confidence": np.random.uniform(0.6, 0.95),
                "patterns_found": np.random.randint(1, 5)
            }
        elif task.task_type == "memory_retrieval":
            results = {
                "memories_retrieved": np.random.randint(5, 20),
                "relevance_score": np.random.uniform(0.7, 0.95),
                "retrieval_time": time.time() - start_time
            }
        elif task.task_type == "attention_focusing":
            results = {
                "attention_targets": np.random.randint(2, 8),
                "focus_strength": np.random.uniform(0.8, 1.0),
                "distractors_filtered": np.random.randint(10, 50)
            }
        elif task.task_type == "learning":
            results = {
                "patterns_learned": np.random.randint(1, 5),
                "learning_rate": np.random.uniform(0.1, 0.3),
                "knowledge_integration": np.random.uniform(0.6, 0.9)
            }
        
        # Economic reward for task completion
        reward = task.priority.value * 10.0
        self.available_av += reward
        self.metrics["av_earned"] += reward
        
        completion_time = time.time()
        task.completed_at = completion_time
        task.status = "completed"
        task.results = results
        
        # Release resources
        for resource_type, required in task.required_resources.items():
            self.resource_usage[resource_type] -= required
            self.resource_usage[resource_type] = max(0, self.resource_usage[resource_type])
        
        # Move to completed tasks
        if task.task_id in self.running_tasks:
            del self.running_tasks[task.task_id]
        self.completed_tasks.append(task)
        
        # Update metrics
        self.metrics["total_tasks_completed"] += 1
        duration = completion_time - task.started_at
        self.metrics["average_task_duration"] = (
            (self.metrics["average_task_duration"] * (self.metrics["total_tasks_completed"] - 1) + duration) /
            self.metrics["total_tasks_completed"]
        )
        
        self.logger.info(f"Completed task {task.task_id} in {duration:.2f}s, earned {reward} AV")
        return results

    def spread_attention(self, source_atom_id: str, target_atoms: List[str], 
                        spreading_amount: float = None) -> Dict[str, float]:
        """Spread attention from source atom to targets with economic constraints"""
        if source_atom_id not in self.attention_atoms:
            return {}
        
        source_atom = self.attention_atoms[source_atom_id]
        
        # Calculate spreading amount based on economic model
        if spreading_amount is None:
            spreading_amount = min(source_atom.sti * self.spreading_factor, source_atom.av * 0.1)
        
        # Economic cost for attention spreading
        cost_per_target = spreading_amount / len(target_atoms) if target_atoms else 0
        total_cost = cost_per_target * len(target_atoms)
        
        if total_cost > self.available_av:
            self.logger.warning(f"Insufficient AV for attention spreading: need {total_cost}, have {self.available_av}")
            return {}
        
        spread_results = {}
        
        for target_id in target_atoms:
            if target_id in self.attention_atoms:
                target_atom = self.attention_atoms[target_id]
                
                # Transfer attention with economic accounting
                target_atom.sti += cost_per_target
                target_atom.av += cost_per_target
                spread_results[target_id] = cost_per_target
                
                self.logger.debug(f"Spread {cost_per_target:.2f} attention from {source_atom_id} to {target_id}")
        
        # Reduce source attention and AV
        source_atom.sti -= spreading_amount
        source_atom.av -= spreading_amount
        self.available_av -= total_cost
        
        self.metrics["attention_spread_events"] += 1
        
        return spread_results

    def collect_attention_rent(self) -> float:
        """Collect rent from attention atoms based on their activation"""
        total_rent = 0.0
        
        for atom in self.attention_atoms.values():
            # Calculate rent based on STI and age
            rent = atom.sti * 0.01 * (1 + atom.age * 0.001)
            atom.rent += rent
            atom.av -= rent
            total_rent += rent
            atom.age += 1
            
            # Remove atoms with negative AV (bankruptcy)
            if atom.av <= 0:
                self.logger.debug(f"Removing bankrupt atom {atom.atom_id}")
        
        # Remove bankrupt atoms
        self.attention_atoms = {
            aid: atom for aid, atom in self.attention_atoms.items() 
            if atom.av > 0
        }
        
        self.available_av += total_rent
        return total_rent

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for benchmarking"""
        # Update resource utilization metrics
        for resource_type in ResourceType:
            utilization = self.resource_usage[resource_type] / self.resource_capacity[resource_type]
            self.metrics["resource_utilization"][resource_type.value] = utilization
        
        # Add current state metrics
        current_metrics = {
            **self.metrics,
            "current_av": self.available_av,
            "total_av": self.total_av,
            "active_atoms": len(self.attention_atoms),
            "pending_bids": len(self.pending_bids),
            "running_tasks": len(self.running_tasks),
            "queued_tasks": len(self.task_queue),
            "av_efficiency": (self.metrics["av_earned"] - self.metrics["av_spent"]) / max(1, self.metrics["av_spent"]),
            "resource_efficiency": sum(self.metrics["resource_utilization"].values()) / len(ResourceType)
        }
        
        return current_metrics

    async def run_attention_cycle(self) -> Dict[str, Any]:
        """Run one cycle of ECAN attention allocation"""
        cycle_start = time.time()
        
        # 1. Process resource allocation for pending bids
        allocations = self.process_resource_allocation()
        
        # 2. Try to schedule queued tasks
        scheduled_count = 0
        while self.task_queue:
            task = self.task_queue.popleft()
            if self.schedule_task(task):
                scheduled_count += 1
            else:
                self.task_queue.appendleft(task)
                break
        
        # 3. Execute running tasks (simulate some completion)
        completed_tasks = []
        for task_id, task in list(self.running_tasks.items()):
            # Simulate task completion probability
            if np.random.random() < 0.3:  # 30% chance per cycle
                results = await self.execute_task(task)
                completed_tasks.append(task_id)
        
        # 4. Collect attention rent
        rent_collected = self.collect_attention_rent()
        
        # 5. Perform attention spreading
        spread_events = 0
        for atom_id, atom in list(self.attention_atoms.items()):
            if atom.sti > self.spreading_threshold:
                # Find targets for spreading (simplified)
                targets = [aid for aid in self.attention_atoms.keys() if aid != atom_id][:3]
                if targets:
                    self.spread_attention(atom_id, targets)
                    spread_events += 1
        
        cycle_duration = time.time() - cycle_start
        
        return {
            "cycle_duration": cycle_duration,
            "allocations_processed": len(allocations),
            "tasks_scheduled": scheduled_count,
            "tasks_completed": len(completed_tasks),
            "rent_collected": rent_collected,
            "attention_spreads": spread_events,
            "performance_metrics": self.get_performance_metrics()
        }

# Utility functions for creating real-world cognitive tasks

def create_reasoning_task(description: str, complexity: float = 0.5) -> CognitiveTask:
    """Create a reasoning task with appropriate resource requirements"""
    return CognitiveTask(
        task_id=str(uuid.uuid4()),
        task_type="reasoning",
        description=description,
        required_resources={
            ResourceType.REASONING: 20.0 * complexity,
            ResourceType.ATTENTION: 15.0 * complexity,
            ResourceType.MEMORY: 10.0 * complexity
        },
        priority=TaskPriority.HIGH if complexity > 0.7 else TaskPriority.NORMAL,
        estimated_duration=complexity * 2.0
    )

def create_memory_task(description: str, memory_load: float = 0.5) -> CognitiveTask:
    """Create a memory retrieval task"""
    return CognitiveTask(
        task_id=str(uuid.uuid4()),
        task_type="memory_retrieval",
        description=description,
        required_resources={
            ResourceType.MEMORY: 30.0 * memory_load,
            ResourceType.ATTENTION: 10.0 * memory_load,
            ResourceType.PROCESSING: 15.0 * memory_load
        },
        priority=TaskPriority.NORMAL,
        estimated_duration=memory_load * 1.5
    )

def create_attention_task(description: str, focus_intensity: float = 0.5) -> CognitiveTask:
    """Create an attention focusing task"""
    return CognitiveTask(
        task_id=str(uuid.uuid4()),
        task_type="attention_focusing",
        description=description,
        required_resources={
            ResourceType.ATTENTION: 40.0 * focus_intensity,
            ResourceType.PROCESSING: 20.0 * focus_intensity
        },
        priority=TaskPriority.HIGH,
        estimated_duration=focus_intensity * 1.0
    )

def create_learning_task(description: str, learning_complexity: float = 0.5) -> CognitiveTask:
    """Create a learning task"""
    return CognitiveTask(
        task_id=str(uuid.uuid4()),
        task_type="learning",
        description=description,
        required_resources={
            ResourceType.LEARNING: 25.0 * learning_complexity,
            ResourceType.MEMORY: 20.0 * learning_complexity,
            ResourceType.PROCESSING: 15.0 * learning_complexity,
            ResourceType.ATTENTION: 10.0 * learning_complexity
        },
        priority=TaskPriority.NORMAL,
        estimated_duration=learning_complexity * 3.0
    )