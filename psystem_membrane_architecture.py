"""
P-System Inspired Membrane Architecture for Frame Problem Resolution

This module implements a membrane computing inspired architecture to address
the frame problem in distributed cognitive systems. It provides nested
membrane structures that can dynamically form boundaries and contexts
for cognitive processing.

Key Features:
- Hierarchical membrane structure with nested contexts
- Dynamic boundary formation based on semantic similarity
- Context-sensitive processing rules
- Membrane permeability for knowledge transfer
- Self-organizing membrane topology
- Frame problem mitigation through context isolation
"""

import time
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MembraneType(Enum):
    """Types of membranes in the P-System architecture"""
    ELEMENTARY = "elementary"  # Leaf membrane containing objects
    COMPOSITE = "composite"    # Contains other membranes
    SKIN = "skin"             # Outermost membrane
    COMMUNICATION = "communication"  # Specialized for inter-membrane transfer
    CONTEXT = "context"       # Defines semantic context boundaries

class ObjectType(Enum):
    """Types of objects that can exist within membranes"""
    ATOM = "atom"               # Basic symbolic atom
    LINK = "link"              # Connection between atoms
    TENSOR = "tensor"          # Tensor representation
    PATTERN = "pattern"        # Cognitive pattern
    RULE = "rule"             # Processing rule
    ATTENTION = "attention"    # Attention allocation object

class PermeabilityType(Enum):
    """Membrane permeability types"""
    IMPERMEABLE = "impermeable"  # No transfer allowed
    SELECTIVE = "selective"      # Conditional transfer
    PERMEABLE = "permeable"      # Free transfer
    DIRECTIONAL = "directional"  # One-way transfer

@dataclass
class MembraneObject:
    """Object that exists within a membrane"""
    object_id: str
    object_type: ObjectType
    content: Dict[str, Any]
    semantic_tags: Set[str] = field(default_factory=set)
    creation_time: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    mobility: float = 1.0  # 0.0 = immobile, 1.0 = fully mobile
    
    def __post_init__(self):
        if not self.object_id:
            self.object_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary representation"""
        return {
            "object_id": self.object_id,
            "object_type": self.object_type.value,
            "content": self.content,
            "semantic_tags": list(self.semantic_tags),
            "creation_time": self.creation_time,
            "last_modified": self.last_modified,
            "mobility": self.mobility
        }

@dataclass
class MembraneRule:
    """Processing rule within a membrane"""
    rule_id: str
    rule_type: str  # "evolution", "communication", "dissolution", "division"
    conditions: Dict[str, Any]  # Conditions for rule activation
    actions: List[Dict[str, Any]]  # Actions to perform
    priority: int = 1  # Higher priority rules execute first
    active: bool = True
    execution_count: int = 0
    
    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = str(uuid.uuid4())

@dataclass
class MembranePermeability:
    """Permeability configuration for a membrane"""
    permeability_type: PermeabilityType
    allowed_object_types: Set[ObjectType] = field(default_factory=set)
    semantic_filters: Set[str] = field(default_factory=set)
    size_limits: Dict[str, int] = field(default_factory=dict)
    directional_rules: Dict[str, str] = field(default_factory=dict)  # membrane_id -> direction

class CognitiveMembrane:
    """Individual membrane in the P-System architecture"""
    
    def __init__(self, membrane_id: str, membrane_type: MembraneType, 
                 parent_id: Optional[str] = None):
        self.membrane_id = membrane_id
        self.membrane_type = membrane_type
        self.parent_id = parent_id
        self.children: Set[str] = set()
        self.objects: Dict[str, MembraneObject] = {}
        self.rules: Dict[str, MembraneRule] = {}
        self.permeability = MembranePermeability(PermeabilityType.SELECTIVE)
        self.semantic_context: Dict[str, Any] = {}
        self.creation_time = time.time()
        self.last_activity = time.time()
        self.activity_level = 0.0
        
        # Frame problem mitigation
        self.frame_constraints: Set[str] = set()  # What should NOT change
        self.change_scope: Set[str] = set()       # What CAN change
        self.isolation_level = 0.5  # 0.0 = no isolation, 1.0 = complete isolation
        
        logger.info(f"Created membrane {membrane_id} of type {membrane_type.value}")
    
    def add_object(self, obj: MembraneObject) -> bool:
        """Add an object to the membrane"""
        if self._can_accept_object(obj):
            self.objects[obj.object_id] = obj
            self._update_activity()
            logger.debug(f"Added object {obj.object_id} to membrane {self.membrane_id}")
            return True
        return False
    
    def remove_object(self, object_id: str) -> Optional[MembraneObject]:
        """Remove an object from the membrane"""
        obj = self.objects.pop(object_id, None)
        if obj:
            self._update_activity()
            logger.debug(f"Removed object {object_id} from membrane {self.membrane_id}")
        return obj
    
    def add_rule(self, rule: MembraneRule) -> bool:
        """Add a processing rule to the membrane"""
        self.rules[rule.rule_id] = rule
        logger.debug(f"Added rule {rule.rule_id} to membrane {self.membrane_id}")
        return True
    
    def execute_rules(self, max_iterations: int = 10) -> List[Dict[str, Any]]:
        """Execute all applicable rules in the membrane"""
        execution_log = []
        
        for iteration in range(max_iterations):
            applicable_rules = self._get_applicable_rules()
            if not applicable_rules:
                break
            
            # Sort by priority
            applicable_rules.sort(key=lambda r: r.priority, reverse=True)
            
            for rule in applicable_rules:
                if self._execute_rule(rule):
                    execution_log.append({
                        "iteration": iteration,
                        "rule_id": rule.rule_id,
                        "rule_type": rule.rule_type,
                        "timestamp": time.time()
                    })
                    rule.execution_count += 1
        
        return execution_log
    
    def set_frame_constraints(self, constraints: Set[str]):
        """Set frame constraints (what should not change)"""
        self.frame_constraints = constraints
        logger.debug(f"Set frame constraints for membrane {self.membrane_id}: {constraints}")
    
    def set_change_scope(self, scope: Set[str]):
        """Set change scope (what can change)"""
        self.change_scope = scope
        logger.debug(f"Set change scope for membrane {self.membrane_id}: {scope}")
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current membrane context"""
        return {
            "membrane_id": self.membrane_id,
            "membrane_type": self.membrane_type.value,
            "object_count": len(self.objects),
            "rule_count": len(self.rules),
            "activity_level": self.activity_level,
            "semantic_context": self.semantic_context,
            "frame_constraints": list(self.frame_constraints),
            "change_scope": list(self.change_scope),
            "isolation_level": self.isolation_level
        }
    
    def _can_accept_object(self, obj: MembraneObject) -> bool:
        """Check if membrane can accept the object"""
        if self.permeability.permeability_type == PermeabilityType.IMPERMEABLE:
            return False
        
        if self.permeability.permeability_type == PermeabilityType.SELECTIVE:
            # Check object type filter
            if (self.permeability.allowed_object_types and 
                obj.object_type not in self.permeability.allowed_object_types):
                return False
            
            # Check semantic filter
            if (self.permeability.semantic_filters and 
                not obj.semantic_tags.intersection(self.permeability.semantic_filters)):
                return False
        
        return True
    
    def _get_applicable_rules(self) -> List[MembraneRule]:
        """Get rules that can be executed in current state"""
        applicable = []
        
        for rule in self.rules.values():
            if rule.active and self._check_rule_conditions(rule):
                applicable.append(rule)
        
        return applicable
    
    def _check_rule_conditions(self, rule: MembraneRule) -> bool:
        """Check if rule conditions are satisfied"""
        conditions = rule.conditions
        
        # Check object-based conditions
        if "required_objects" in conditions:
            required_types = conditions["required_objects"]
            current_types = [obj.object_type.value for obj in self.objects.values()]
            if not all(req_type in current_types for req_type in required_types):
                return False
        
        # Check semantic conditions
        if "semantic_requirements" in conditions:
            required_tags = set(conditions["semantic_requirements"])
            all_tags = set()
            for obj in self.objects.values():
                all_tags.update(obj.semantic_tags)
            if not required_tags.issubset(all_tags):
                return False
        
        # Check activity level conditions
        if "min_activity" in conditions:
            if self.activity_level < conditions["min_activity"]:
                return False
        
        return True
    
    def _execute_rule(self, rule: MembraneRule) -> bool:
        """Execute a specific rule"""
        try:
            for action in rule.actions:
                action_type = action.get("type", "")
                
                if action_type == "object_evolution":
                    self._execute_object_evolution(action)
                elif action_type == "object_communication":
                    self._execute_object_communication(action)
                elif action_type == "membrane_division":
                    self._execute_membrane_division(action)
                elif action_type == "context_update":
                    self._execute_context_update(action)
                elif action_type == "frame_enforcement":
                    self._execute_frame_enforcement(action)
                
            self._update_activity()
            return True
        
        except Exception as e:
            logger.error(f"Error executing rule {rule.rule_id}: {e}")
            return False
    
    def _execute_object_evolution(self, action: Dict[str, Any]):
        """Execute object evolution action"""
        target_objects = action.get("target_objects", [])
        evolution_type = action.get("evolution_type", "generic")
        
        for obj_id in target_objects:
            if obj_id in self.objects:
                obj = self.objects[obj_id]
                
                if evolution_type == "semantic_drift":
                    # Add random semantic tag
                    new_tag = f"evolved_{int(time.time())}"
                    obj.semantic_tags.add(new_tag)
                elif evolution_type == "content_modification":
                    # Modify object content
                    if "weight" in obj.content:
                        obj.content["weight"] *= 1.1  # Increase weight
                
                obj.last_modified = time.time()
    
    def _execute_object_communication(self, action: Dict[str, Any]):
        """Execute object communication action (placeholder)"""
        # This would interface with the membrane system to transfer objects
        source_objects = action.get("source_objects", [])
        target_membrane = action.get("target_membrane", "")
        
        # Mark objects for potential transfer
        for obj_id in source_objects:
            if obj_id in self.objects:
                obj = self.objects[obj_id]
                obj.semantic_tags.add("transfer_candidate")
    
    def _execute_membrane_division(self, action: Dict[str, Any]):
        """Execute membrane division action (placeholder)"""
        # This would create new membranes
        division_type = action.get("division_type", "binary")
        logger.info(f"Membrane division requested: {division_type}")
    
    def _execute_context_update(self, action: Dict[str, Any]):
        """Execute context update action"""
        updates = action.get("updates", {})
        for key, value in updates.items():
            self.semantic_context[key] = value
    
    def _execute_frame_enforcement(self, action: Dict[str, Any]):
        """Execute frame constraint enforcement"""
        constraint_type = action.get("constraint_type", "preservation")
        
        if constraint_type == "preservation":
            # Preserve objects tagged with frame constraints
            preserved_objects = action.get("preserve_objects", [])
            for obj_id in preserved_objects:
                if obj_id in self.objects:
                    self.objects[obj_id].mobility = 0.0  # Make immobile
        
        elif constraint_type == "isolation":
            # Increase isolation level
            isolation_increase = action.get("isolation_increase", 0.1)
            self.isolation_level = min(1.0, self.isolation_level + isolation_increase)
    
    def _update_activity(self):
        """Update membrane activity level"""
        current_time = time.time()
        time_since_last = current_time - self.last_activity
        
        # Activity decay over time
        decay_factor = max(0.0, 1.0 - time_since_last / 60.0)  # Decay over 1 minute
        self.activity_level *= decay_factor
        
        # Increase activity due to current event
        activity_boost = 0.1
        self.activity_level = min(1.0, self.activity_level + activity_boost)
        
        self.last_activity = current_time

class PSystemMembraneArchitecture:
    """Main P-System membrane architecture for frame problem resolution"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.membranes: Dict[str, CognitiveMembrane] = {}
        self.membrane_hierarchy: Dict[str, Set[str]] = defaultdict(set)
        self.skin_membrane_id = None
        self.communication_channels: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.global_context: Dict[str, Any] = {}
        
        # Frame problem resolution state
        self.frame_states: Dict[str, Dict[str, Any]] = {}
        self.change_history: List[Dict[str, Any]] = []
        
        # Create skin membrane
        self._create_skin_membrane()
        
        logger.info(f"Initialized P-System membrane architecture for agent {agent_id}")
    
    def create_membrane(self, membrane_type: MembraneType, 
                       parent_id: Optional[str] = None,
                       membrane_id: Optional[str] = None) -> str:
        """Create a new membrane in the architecture"""
        if membrane_id is None:
            membrane_id = f"{self.agent_id}_membrane_{len(self.membranes)}"
        
        if parent_id and parent_id not in self.membranes:
            raise ValueError(f"Parent membrane {parent_id} does not exist")
        
        membrane = CognitiveMembrane(membrane_id, membrane_type, parent_id)
        self.membranes[membrane_id] = membrane
        
        if parent_id:
            self.membrane_hierarchy[parent_id].add(membrane_id)
            self.membranes[parent_id].children.add(membrane_id)
        
        logger.info(f"Created membrane {membrane_id} of type {membrane_type.value}")
        return membrane_id
    
    def dissolve_membrane(self, membrane_id: str) -> bool:
        """Dissolve a membrane and redistribute its contents"""
        if membrane_id not in self.membranes:
            return False
        
        membrane = self.membranes[membrane_id]
        parent_id = membrane.parent_id
        
        # Move objects to parent membrane
        if parent_id and parent_id in self.membranes:
            parent_membrane = self.membranes[parent_id]
            for obj in membrane.objects.values():
                parent_membrane.add_object(obj)
        
        # Remove from hierarchy
        if parent_id:
            self.membrane_hierarchy[parent_id].discard(membrane_id)
            self.membranes[parent_id].children.discard(membrane_id)
        
        # Remove children relationships
        for child_id in membrane.children:
            if child_id in self.membranes:
                self.membranes[child_id].parent_id = parent_id
                if parent_id:
                    self.membrane_hierarchy[parent_id].add(child_id)
        
        del self.membranes[membrane_id]
        logger.info(f"Dissolved membrane {membrane_id}")
        return True
    
    def add_object_to_membrane(self, membrane_id: str, obj: MembraneObject) -> bool:
        """Add an object to a specific membrane"""
        if membrane_id not in self.membranes:
            return False
        
        membrane = self.membranes[membrane_id]
        success = membrane.add_object(obj)
        
        if success:
            self._record_change("object_added", {
                "membrane_id": membrane_id,
                "object_id": obj.object_id,
                "object_type": obj.object_type.value
            })
        
        return success
    
    def transfer_object(self, object_id: str, source_membrane_id: str, 
                       target_membrane_id: str) -> bool:
        """Transfer an object between membranes"""
        if (source_membrane_id not in self.membranes or 
            target_membrane_id not in self.membranes):
            return False
        
        source_membrane = self.membranes[source_membrane_id]
        target_membrane = self.membranes[target_membrane_id]
        
        # Check if transfer is allowed
        if not self._can_transfer_object(object_id, source_membrane, target_membrane):
            return False
        
        # Perform transfer
        obj = source_membrane.remove_object(object_id)
        if obj and target_membrane.add_object(obj):
            self._record_change("object_transferred", {
                "object_id": object_id,
                "source_membrane": source_membrane_id,
                "target_membrane": target_membrane_id
            })
            return True
        
        return False
    
    def execute_membrane_rules(self, membrane_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Execute rules in specified membrane(s)"""
        execution_results = {}
        
        if membrane_id:
            if membrane_id in self.membranes:
                execution_results[membrane_id] = self.membranes[membrane_id].execute_rules()
        else:
            # Execute rules in all membranes
            for mem_id, membrane in self.membranes.items():
                execution_results[mem_id] = membrane.execute_rules()
        
        return execution_results
    
    def set_membrane_context(self, membrane_id: str, context: Dict[str, Any]) -> bool:
        """Set semantic context for a membrane"""
        if membrane_id not in self.membranes:
            return False
        
        membrane = self.membranes[membrane_id]
        membrane.semantic_context.update(context)
        
        # Create context boundary rules for frame problem resolution
        self._create_context_boundary_rules(membrane_id, context)
        
        return True
    
    def isolate_membrane_context(self, membrane_id: str, isolation_level: float = 0.8):
        """Isolate a membrane context to prevent frame problem"""
        if membrane_id not in self.membranes:
            return False
        
        membrane = self.membranes[membrane_id]
        membrane.isolation_level = isolation_level
        
        # Update permeability based on isolation level
        if isolation_level > 0.7:
            membrane.permeability.permeability_type = PermeabilityType.SELECTIVE
        elif isolation_level > 0.9:
            membrane.permeability.permeability_type = PermeabilityType.IMPERMEABLE
        
        # Create isolation rules
        isolation_rule = MembraneRule(
            rule_id=f"isolation_{membrane_id}",
            rule_type="frame_enforcement",
            conditions={"min_activity": 0.1},
            actions=[{
                "type": "frame_enforcement",
                "constraint_type": "isolation",
                "isolation_increase": 0.1
            }],
            priority=10
        )
        
        membrane.add_rule(isolation_rule)
        
        logger.info(f"Isolated membrane {membrane_id} with level {isolation_level}")
        return True
    
    def create_context_membrane(self, context_definition: Dict[str, Any], 
                              parent_id: Optional[str] = None) -> str:
        """Create a specialized context membrane for frame problem resolution"""
        membrane_id = self.create_membrane(MembraneType.CONTEXT, parent_id)
        membrane = self.membranes[membrane_id]
        
        # Set context-specific configuration
        membrane.semantic_context = context_definition.copy()
        
        # Define frame constraints based on context
        if "frame_constraints" in context_definition:
            membrane.set_frame_constraints(set(context_definition["frame_constraints"]))
        
        if "change_scope" in context_definition:
            membrane.set_change_scope(set(context_definition["change_scope"]))
        
        # Create context-specific rules
        self._create_context_specific_rules(membrane_id, context_definition)
        
        logger.info(f"Created context membrane {membrane_id} with context: {context_definition}")
        return membrane_id
    
    def get_membrane_state(self, membrane_id: str) -> Optional[Dict[str, Any]]:
        """Get complete state of a membrane"""
        if membrane_id not in self.membranes:
            return None
        
        membrane = self.membranes[membrane_id]
        
        return {
            "membrane_id": membrane_id,
            "membrane_type": membrane.membrane_type.value,
            "parent_id": membrane.parent_id,
            "children": list(membrane.children),
            "objects": {obj_id: obj.to_dict() for obj_id, obj in membrane.objects.items()},
            "rules": {rule_id: {
                "rule_type": rule.rule_type,
                "active": rule.active,
                "execution_count": rule.execution_count
            } for rule_id, rule in membrane.rules.items()},
            "context_summary": membrane.get_context_summary(),
            "activity_level": membrane.activity_level,
            "isolation_level": membrane.isolation_level
        }
    
    def get_architecture_overview(self) -> Dict[str, Any]:
        """Get overview of entire membrane architecture"""
        membrane_stats = {}
        for mem_id, membrane in self.membranes.items():
            membrane_stats[mem_id] = {
                "type": membrane.membrane_type.value,
                "object_count": len(membrane.objects),
                "rule_count": len(membrane.rules),
                "activity_level": membrane.activity_level,
                "isolation_level": membrane.isolation_level
            }
        
        return {
            "agent_id": self.agent_id,
            "total_membranes": len(self.membranes),
            "membrane_hierarchy": dict(self.membrane_hierarchy),
            "membrane_stats": membrane_stats,
            "change_history_length": len(self.change_history),
            "global_context": self.global_context
        }
    
    def _create_skin_membrane(self):
        """Create the outermost skin membrane"""
        self.skin_membrane_id = f"{self.agent_id}_skin"
        skin_membrane = CognitiveMembrane(self.skin_membrane_id, MembraneType.SKIN)
        skin_membrane.permeability.permeability_type = PermeabilityType.SELECTIVE
        self.membranes[self.skin_membrane_id] = skin_membrane
    
    def _can_transfer_object(self, object_id: str, source_membrane: CognitiveMembrane, 
                           target_membrane: CognitiveMembrane) -> bool:
        """Check if object transfer is allowed between membranes"""
        if object_id not in source_membrane.objects:
            return False
        
        obj = source_membrane.objects[object_id]
        
        # Check object mobility
        if obj.mobility <= 0.0:
            return False
        
        # Check target membrane permeability
        if not target_membrane._can_accept_object(obj):
            return False
        
        # Check frame constraints
        if object_id in source_membrane.frame_constraints:
            return False
        
        # Check isolation levels
        if (source_membrane.isolation_level > 0.8 or 
            target_membrane.isolation_level > 0.8):
            return False
        
        return True
    
    def _record_change(self, change_type: str, change_data: Dict[str, Any]):
        """Record a change for frame problem analysis"""
        change_record = {
            "timestamp": time.time(),
            "change_type": change_type,
            "change_data": change_data,
            "agent_id": self.agent_id
        }
        
        self.change_history.append(change_record)
        
        # Keep only recent history
        max_history = 1000
        if len(self.change_history) > max_history:
            self.change_history = self.change_history[-max_history:]
    
    def _create_context_boundary_rules(self, membrane_id: str, context: Dict[str, Any]):
        """Create rules to maintain context boundaries"""
        membrane = self.membranes[membrane_id]
        
        # Rule to prevent context drift
        context_preservation_rule = MembraneRule(
            rule_id=f"context_preserve_{membrane_id}",
            rule_type="context_enforcement",
            conditions={"min_activity": 0.05},
            actions=[{
                "type": "frame_enforcement",
                "constraint_type": "preservation",
                "preserve_objects": list(context.get("core_objects", []))
            }],
            priority=8
        )
        
        membrane.add_rule(context_preservation_rule)
    
    def _create_context_specific_rules(self, membrane_id: str, context_definition: Dict[str, Any]):
        """Create rules specific to the context definition"""
        membrane = self.membranes[membrane_id]
        
        # Rule for context-specific object evolution
        if "evolution_rules" in context_definition:
            for rule_def in context_definition["evolution_rules"]:
                evolution_rule = MembraneRule(
                    rule_id=f"evolution_{membrane_id}_{rule_def.get('name', 'default')}",
                    rule_type="object_evolution",
                    conditions=rule_def.get("conditions", {}),
                    actions=rule_def.get("actions", []),
                    priority=rule_def.get("priority", 5)
                )
                membrane.add_rule(evolution_rule)
        
        # Rule for maintaining semantic coherence
        coherence_rule = MembraneRule(
            rule_id=f"coherence_{membrane_id}",
            rule_type="semantic_coherence",
            conditions={"semantic_requirements": context_definition.get("required_tags", [])},
            actions=[{
                "type": "context_update",
                "updates": {"coherence_check": True}
            }],
            priority=6
        )
        
        membrane.add_rule(coherence_rule)

# Example usage and integration
if __name__ == "__main__":
    # Create P-System architecture
    psystem = PSystemMembraneArchitecture("test_agent")
    
    # Create context membranes
    reasoning_context = {
        "semantic_focus": "logical_reasoning",
        "frame_constraints": ["logical_axioms", "inference_rules"],
        "change_scope": ["temporary_conclusions", "working_memory"],
        "required_tags": ["reasoning", "logic"]
    }
    
    reasoning_membrane_id = psystem.create_context_membrane(
        reasoning_context, 
        psystem.skin_membrane_id
    )
    
    creativity_context = {
        "semantic_focus": "creative_thinking",
        "frame_constraints": ["core_concepts"],
        "change_scope": ["associations", "novel_combinations"],
        "required_tags": ["creativity", "imagination"]
    }
    
    creativity_membrane_id = psystem.create_context_membrane(
        creativity_context,
        psystem.skin_membrane_id
    )
    
    # Add objects to membranes
    reasoning_object = MembraneObject(
        object_id="logical_rule_1",
        object_type=ObjectType.RULE,
        content={"rule": "modus_ponens", "strength": 0.9},
        semantic_tags={"reasoning", "logic", "inference"}
    )
    
    creative_object = MembraneObject(
        object_id="creative_pattern_1",
        object_type=ObjectType.PATTERN,
        content={"pattern": "metaphor_generation", "novelty": 0.8},
        semantic_tags={"creativity", "metaphor", "association"}
    )
    
    psystem.add_object_to_membrane(reasoning_membrane_id, reasoning_object)
    psystem.add_object_to_membrane(creativity_membrane_id, creative_object)
    
    # Isolate reasoning context to prevent interference
    psystem.isolate_membrane_context(reasoning_membrane_id, isolation_level=0.8)
    
    # Execute rules
    execution_results = psystem.execute_membrane_rules()
    
    # Print architecture overview
    overview = psystem.get_architecture_overview()
    print(f"P-System Architecture Overview:")
    print(f"  Total membranes: {overview['total_membranes']}")
    print(f"  Change history: {overview['change_history_length']} events")
    
    for mem_id, stats in overview['membrane_stats'].items():
        print(f"  Membrane {mem_id}: {stats['type']}, "
              f"objects={stats['object_count']}, "
              f"isolation={stats['isolation_level']:.2f}")