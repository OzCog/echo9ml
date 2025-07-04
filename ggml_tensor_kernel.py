"""
GGML Tensor Kernel Integration for Echo9ML Distributed Cognitive Grammar

This module provides integration points for GGML tensor operations
in the distributed cognitive grammar system. It defines tensor shapes,
custom operations, and semantic mappings for cognitive processing.

Based on the specification in echo9ml.md for tensor customization.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TensorOperationType(Enum):
    """Types of tensor operations for cognitive processing"""
    PERSONA_EVOLVE = "persona_evolve"
    ATTENTION_SPREAD = "attention_spread"
    MEMORY_CONSOLIDATE = "memory_consolidate"
    REASONING_PROPAGATE = "reasoning_propagate"
    LEARNING_ADAPT = "learning_adapt"

@dataclass
class TensorMetadata:
    """Metadata for cognitive tensors"""
    cognitive_dimension: str
    semantic_weight: float = 1.0
    temporal_context: Optional[str] = None
    source_agent: Optional[str] = None
    creation_time: float = field(default_factory=time.time)

@dataclass
class CognitiveTensor:
    """Cognitive tensor with semantic meaning"""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    data: Optional[List[float]] = None
    metadata: TensorMetadata = field(default_factory=lambda: TensorMetadata("unknown"))
    
    def __post_init__(self):
        if self.data is None:
            # Initialize with zeros
            size = 1
            for dim in self.shape:
                size *= dim
            self.data = [0.0] * size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tensor to dictionary for serialization"""
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "data": self.data,
            "metadata": {
                "cognitive_dimension": self.metadata.cognitive_dimension,
                "semantic_weight": self.metadata.semantic_weight,
                "temporal_context": self.metadata.temporal_context,
                "source_agent": self.metadata.source_agent,
                "creation_time": self.metadata.creation_time
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveTensor':
        """Create tensor from dictionary"""
        metadata = TensorMetadata(**data.get("metadata", {}))
        return cls(
            name=data["name"],
            shape=tuple(data["shape"]),
            dtype=data["dtype"],
            data=data.get("data"),
            metadata=metadata
        )

class GGMLTensorKernel:
    """GGML tensor kernel for distributed cognitive operations"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.tensors: Dict[str, CognitiveTensor] = {}
        self.tensor_shapes: Dict[str, Tuple[int, ...]] = {}
        self.custom_operations: Dict[str, callable] = {}
        
        # Initialize default tensor shapes
        self._initialize_tensor_shapes()
        
        # Register custom operations
        self._register_custom_operations()
        
        logger.info(f"Initialized GGML tensor kernel for agent {agent_id}")
    
    def _initialize_tensor_shapes(self):
        """Initialize tensor shapes based on echo9ml.md specification"""
        # Prime factorization for evolutionary flexibility
        self.tensor_shapes.update({
            # Persona tensor: [persona_id, trait_id, time, context, valence]
            "persona": (3, 7, 13, 5, 2),  # 3x7x13x5x2 = 2730 elements
            
            # Memory tensor: [memory_node, memory_type, salience, temporal, relational]
            "memory": (101, 8, 5, 7, 3),  # 101x8x5x7x3 = 84,840 elements
            
            # Attention tensor: [source, target, strength, context, decay]
            "attention": (17, 17, 11, 7, 2),  # 17x17x11x7x2 = 44,506 elements
            
            # Reasoning tensor: [premise, conclusion, confidence, context, rule_type]
            "reasoning": (23, 23, 9, 5, 4),  # 23x23x9x5x4 = 18,900 elements
            
            # Learning tensor: [experience, adaptation, weight, context, meta]
            "learning": (19, 13, 7, 5, 3),  # 19x13x7x5x3 = 17,745 elements
        })
    
    def _register_custom_operations(self):
        """Register custom GGML operations for cognitive processing"""
        self.custom_operations.update({
            TensorOperationType.PERSONA_EVOLVE: self._persona_evolve_op,
            TensorOperationType.ATTENTION_SPREAD: self._attention_spread_op,
            TensorOperationType.MEMORY_CONSOLIDATE: self._memory_consolidate_op,
            TensorOperationType.REASONING_PROPAGATE: self._reasoning_propagate_op,
            TensorOperationType.LEARNING_ADAPT: self._learning_adapt_op
        })
    
    def create_tensor(self, name: str, tensor_type: str, 
                     cognitive_dimension: str, semantic_weight: float = 1.0) -> CognitiveTensor:
        """Create a new cognitive tensor"""
        if tensor_type not in self.tensor_shapes:
            raise ValueError(f"Unknown tensor type: {tensor_type}")
        
        shape = self.tensor_shapes[tensor_type]
        metadata = TensorMetadata(
            cognitive_dimension=cognitive_dimension,
            semantic_weight=semantic_weight,
            source_agent=self.agent_id
        )
        
        tensor = CognitiveTensor(
            name=name,
            shape=shape,
            dtype="float32",
            metadata=metadata
        )
        
        self.tensors[name] = tensor
        logger.info(f"Created tensor {name} with shape {shape}")
        return tensor
    
    def get_tensor(self, name: str) -> Optional[CognitiveTensor]:
        """Get tensor by name"""
        return self.tensors.get(name)
    
    def update_tensor(self, name: str, data: List[float]) -> bool:
        """Update tensor data"""
        if name not in self.tensors:
            return False
        
        tensor = self.tensors[name]
        expected_size = 1
        for dim in tensor.shape:
            expected_size *= dim
        
        if len(data) != expected_size:
            logger.error(f"Data size mismatch for tensor {name}: expected {expected_size}, got {len(data)}")
            return False
        
        tensor.data = data
        logger.info(f"Updated tensor {name} with new data")
        return True
    
    def execute_operation(self, operation_type: TensorOperationType, 
                         input_tensors: List[str], output_tensor: str,
                         **kwargs) -> bool:
        """Execute custom tensor operation"""
        if operation_type not in self.custom_operations:
            logger.error(f"Unknown operation type: {operation_type}")
            return False
        
        # Validate input tensors exist
        for tensor_name in input_tensors:
            if tensor_name not in self.tensors:
                logger.error(f"Input tensor {tensor_name} not found")
                return False
        
        try:
            # Execute operation
            result = self.custom_operations[operation_type](
                input_tensors, output_tensor, **kwargs
            )
            
            if result:
                logger.info(f"Executed operation {operation_type.value} successfully")
            else:
                logger.error(f"Operation {operation_type.value} failed")
            
            return result
        
        except Exception as e:
            logger.error(f"Error executing operation {operation_type.value}: {e}")
            return False
    
    def _persona_evolve_op(self, input_tensors: List[str], output_tensor: str, 
                          learning_rate: float = 0.1, **kwargs) -> bool:
        """
        Custom GGML operation for persona evolution
        
        Implements the persona evolution mechanism from echo9ml.md:
        - Apply evolutionary rules: selection, mutation, attention reweighting
        - Update persona traits based on experience history
        """
        if not input_tensors:
            return False
        
        persona_tensor = self.tensors.get(input_tensors[0])
        if not persona_tensor:
            return False
        
        # Simple evolution: apply learning rate to modify persona traits
        if persona_tensor.data:
            evolved_data = []
            for i, value in enumerate(persona_tensor.data):
                # Apply stochastic evolution with learning rate
                import random
                evolution_factor = 1.0 + (random.random() - 0.5) * learning_rate
                evolved_value = value * evolution_factor
                evolved_data.append(max(0.0, min(1.0, evolved_value)))  # Clamp to [0,1]
            
            # Create or update output tensor
            if output_tensor not in self.tensors:
                self.tensors[output_tensor] = CognitiveTensor(
                    name=output_tensor,
                    shape=persona_tensor.shape,
                    dtype=persona_tensor.dtype,
                    data=evolved_data,
                    metadata=TensorMetadata(
                        cognitive_dimension="persona_evolution",
                        semantic_weight=persona_tensor.metadata.semantic_weight,
                        source_agent=self.agent_id
                    )
                )
            else:
                self.tensors[output_tensor].data = evolved_data
            
            return True
        
        return False
    
    def _attention_spread_op(self, input_tensors: List[str], output_tensor: str,
                           decay_factor: float = 0.8, **kwargs) -> bool:
        """
        Custom GGML operation for attention spreading
        
        Implements attention allocation across cognitive networks
        """
        if not input_tensors:
            return False
        
        attention_tensor = self.tensors.get(input_tensors[0])
        if not attention_tensor or not attention_tensor.data:
            return False
        
        # Implement attention spreading with decay
        spread_data = []
        for i, value in enumerate(attention_tensor.data):
            # Apply decay factor for attention spreading
            spread_value = value * decay_factor
            spread_data.append(spread_value)
        
        # Create or update output tensor
        if output_tensor not in self.tensors:
            self.tensors[output_tensor] = CognitiveTensor(
                name=output_tensor,
                shape=attention_tensor.shape,
                dtype=attention_tensor.dtype,
                data=spread_data,
                metadata=TensorMetadata(
                    cognitive_dimension="attention_spread",
                    semantic_weight=attention_tensor.metadata.semantic_weight,
                    source_agent=self.agent_id
                )
            )
        else:
            self.tensors[output_tensor].data = spread_data
        
        return True
    
    def _memory_consolidate_op(self, input_tensors: List[str], output_tensor: str,
                             consolidation_threshold: float = 0.7, **kwargs) -> bool:
        """
        Custom GGML operation for memory consolidation
        
        Consolidates memory representations based on salience and connections
        """
        if not input_tensors:
            return False
        
        memory_tensor = self.tensors.get(input_tensors[0])
        if not memory_tensor or not memory_tensor.data:
            return False
        
        # Simple consolidation: enhance values above threshold
        consolidated_data = []
        for value in memory_tensor.data:
            if value > consolidation_threshold:
                consolidated_value = min(1.0, value * 1.2)  # Enhance by 20%
            else:
                consolidated_value = value * 0.9  # Decay by 10%
            consolidated_data.append(consolidated_value)
        
        # Create or update output tensor
        if output_tensor not in self.tensors:
            self.tensors[output_tensor] = CognitiveTensor(
                name=output_tensor,
                shape=memory_tensor.shape,
                dtype=memory_tensor.dtype,
                data=consolidated_data,
                metadata=TensorMetadata(
                    cognitive_dimension="memory_consolidation",
                    semantic_weight=memory_tensor.metadata.semantic_weight,
                    source_agent=self.agent_id
                )
            )
        else:
            self.tensors[output_tensor].data = consolidated_data
        
        return True
    
    def _reasoning_propagate_op(self, input_tensors: List[str], output_tensor: str,
                              confidence_threshold: float = 0.5, **kwargs) -> bool:
        """
        Custom GGML operation for reasoning propagation
        
        Propagates reasoning patterns across cognitive networks
        """
        if not input_tensors:
            return False
        
        reasoning_tensor = self.tensors.get(input_tensors[0])
        if not reasoning_tensor or not reasoning_tensor.data:
            return False
        
        # Simple reasoning propagation
        propagated_data = []
        for value in reasoning_tensor.data:
            if value > confidence_threshold:
                # Propagate with confidence
                propagated_value = min(1.0, value + 0.1)
            else:
                # Decay uncertain reasoning
                propagated_value = max(0.0, value - 0.05)
            propagated_data.append(propagated_value)
        
        # Create or update output tensor
        if output_tensor not in self.tensors:
            self.tensors[output_tensor] = CognitiveTensor(
                name=output_tensor,
                shape=reasoning_tensor.shape,
                dtype=reasoning_tensor.dtype,
                data=propagated_data,
                metadata=TensorMetadata(
                    cognitive_dimension="reasoning_propagation",
                    semantic_weight=reasoning_tensor.metadata.semantic_weight,
                    source_agent=self.agent_id
                )
            )
        else:
            self.tensors[output_tensor].data = propagated_data
        
        return True
    
    def _learning_adapt_op(self, input_tensors: List[str], output_tensor: str,
                          adaptation_rate: float = 0.05, **kwargs) -> bool:
        """
        Custom GGML operation for adaptive learning
        
        Implements MOSES-style evolutionary search for cognitive adaptation
        """
        if not input_tensors:
            return False
        
        learning_tensor = self.tensors.get(input_tensors[0])
        if not learning_tensor or not learning_tensor.data:
            return False
        
        # Simple adaptive learning
        adapted_data = []
        import random
        
        for value in learning_tensor.data:
            # Apply random variation with adaptation rate
            variation = (random.random() - 0.5) * adaptation_rate
            adapted_value = value + variation
            adapted_data.append(max(0.0, min(1.0, adapted_value)))  # Clamp to [0,1]
        
        # Create or update output tensor
        if output_tensor not in self.tensors:
            self.tensors[output_tensor] = CognitiveTensor(
                name=output_tensor,
                shape=learning_tensor.shape,
                dtype=learning_tensor.dtype,
                data=adapted_data,
                metadata=TensorMetadata(
                    cognitive_dimension="adaptive_learning",
                    semantic_weight=learning_tensor.metadata.semantic_weight,
                    source_agent=self.agent_id
                )
            )
        else:
            self.tensors[output_tensor].data = adapted_data
        
        return True
    
    def get_tensor_info(self, name: str) -> Dict[str, Any]:
        """Get tensor information"""
        tensor = self.tensors.get(name)
        if not tensor:
            return {}
        
        return {
            "name": tensor.name,
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "size": len(tensor.data) if tensor.data else 0,
            "cognitive_dimension": tensor.metadata.cognitive_dimension,
            "semantic_weight": tensor.metadata.semantic_weight,
            "source_agent": tensor.metadata.source_agent,
            "creation_time": tensor.metadata.creation_time
        }
    
    def list_tensors(self) -> List[str]:
        """List all tensor names"""
        return list(self.tensors.keys())
    
    def export_tensor_catalog(self) -> Dict[str, Any]:
        """Export tensor catalog for sharing with other agents"""
        catalog = {
            "agent_id": self.agent_id,
            "tensor_shapes": self.tensor_shapes,
            "tensors": {
                name: tensor.to_dict() for name, tensor in self.tensors.items()
            },
            "export_time": time.time()
        }
        
        return catalog
    
    def import_tensor_catalog(self, catalog: Dict[str, Any]) -> bool:
        """Import tensor catalog from another agent"""
        try:
            source_agent = catalog.get("agent_id", "unknown")
            
            # Import tensor shapes
            for shape_name, shape in catalog.get("tensor_shapes", {}).items():
                if shape_name not in self.tensor_shapes:
                    self.tensor_shapes[shape_name] = tuple(shape)
            
            # Import tensors
            for tensor_name, tensor_data in catalog.get("tensors", {}).items():
                imported_tensor = CognitiveTensor.from_dict(tensor_data)
                # Prefix with source agent to avoid conflicts
                prefixed_name = f"{source_agent}_{tensor_name}"
                self.tensors[prefixed_name] = imported_tensor
            
            logger.info(f"Imported tensor catalog from {source_agent}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing tensor catalog: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Create tensor kernel
    kernel = GGMLTensorKernel("test_agent")
    
    # Create some tensors
    persona_tensor = kernel.create_tensor("persona_base", "persona", "persona_traits")
    attention_tensor = kernel.create_tensor("attention_base", "attention", "attention_allocation")
    
    # Execute operations
    kernel.execute_operation(
        TensorOperationType.PERSONA_EVOLVE,
        ["persona_base"],
        "persona_evolved",
        learning_rate=0.1
    )
    
    kernel.execute_operation(
        TensorOperationType.ATTENTION_SPREAD,
        ["attention_base"],
        "attention_spread",
        decay_factor=0.8
    )
    
    # Print tensor information
    print("Tensor Catalog:")
    for tensor_name in kernel.list_tensors():
        info = kernel.get_tensor_info(tensor_name)
        print(f"  {tensor_name}: {info}")
    
    # Export catalog
    catalog = kernel.export_tensor_catalog()
    print(f"\nExported catalog with {len(catalog['tensors'])} tensors")