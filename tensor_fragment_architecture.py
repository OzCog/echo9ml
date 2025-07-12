"""
Tensor Fragment Architecture for Distributed Cognitive Grammar

This module enhances the existing tensor operations to support fragment sharing
and distributed processing in the cognitive grammar network. It implements the
tensor fragment architecture requirements for Phase 1.

Key Features:
- Tensor fragment serialization and sharing
- Prime factorization optimization for evolutionary flexibility  
- Distributed tensor operations across cognitive agents
- Fragment reconstruction and validation
- Semantic tensor mappings for cognitive dimensions
"""

import numpy as np
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import existing components
try:
    from ggml_tensor_kernel import GGMLTensorKernel, CognitiveTensor, TensorMetadata, TensorOperationType
    from ko6ml_atomspace_adapter import Ko6mlPrimitive, Ko6mlPrimitiveType
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    logging.warning("Some tensor components not available")

logger = logging.getLogger(__name__)

class FragmentCompressionType(Enum):
    """Types of tensor fragment compression"""
    NONE = "none"
    SPARSE = "sparse"
    QUANTIZED = "quantized"
    DELTA = "delta"
    PRIME_FACTORIZED = "prime_factorized"

class FragmentSharingMode(Enum):
    """Modes for sharing tensor fragments"""
    FULL_FRAGMENT = "full_fragment"
    SPARSE_UPDATES = "sparse_updates"
    GRADIENT_ONLY = "gradient_only"
    SEMANTIC_ONLY = "semantic_only"

@dataclass
class TensorFragment:
    """Serializable tensor fragment for distributed sharing"""
    fragment_id: str
    source_agent: str
    tensor_name: str
    shape: Tuple[int, ...]
    semantic_dimensions: List[str]
    data: List[float]
    metadata: Dict[str, Any]
    compression_type: FragmentCompressionType = FragmentCompressionType.NONE
    sharing_mode: FragmentSharingMode = FragmentSharingMode.FULL_FRAGMENT
    creation_time: float = field(default_factory=time.time)
    version: int = 1
    checksum: str = field(default="")
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for fragment integrity"""
        content = f"{self.shape}{self.data}{self.version}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fragment to dictionary for serialization"""
        return {
            "fragment_id": self.fragment_id,
            "source_agent": self.source_agent,
            "tensor_name": self.tensor_name,
            "shape": self.shape,
            "semantic_dimensions": self.semantic_dimensions,
            "data": self.data,
            "metadata": self.metadata,
            "compression_type": self.compression_type.value,
            "sharing_mode": self.sharing_mode.value,
            "creation_time": self.creation_time,
            "version": self.version,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TensorFragment':
        """Create fragment from dictionary"""
        return cls(
            fragment_id=data["fragment_id"],
            source_agent=data["source_agent"],
            tensor_name=data["tensor_name"],
            shape=tuple(data["shape"]),
            semantic_dimensions=data["semantic_dimensions"],
            data=data["data"],
            metadata=data["metadata"],
            compression_type=FragmentCompressionType(data.get("compression_type", "none")),
            sharing_mode=FragmentSharingMode(data.get("sharing_mode", "full_fragment")),
            creation_time=data.get("creation_time", time.time()),
            version=data.get("version", 1),
            checksum=data.get("checksum", "")
        )
    
    def validate_integrity(self) -> bool:
        """Validate fragment integrity using checksum"""
        expected_checksum = self._calculate_checksum()
        return self.checksum == expected_checksum
    
    def get_fragment_size(self) -> int:
        """Get fragment size in elements"""
        return len(self.data)
    
    def get_memory_footprint(self) -> int:
        """Estimate memory footprint in bytes"""
        # Rough estimate: float32 = 4 bytes + metadata overhead
        return len(self.data) * 4 + len(json.dumps(self.metadata)) + 1024

@dataclass 
class FragmentOperation:
    """Operation to be performed on tensor fragments"""
    operation_id: str
    operation_type: TensorOperationType
    source_fragments: List[str]
    target_fragment: str
    parameters: Dict[str, Any]
    execution_time: float = field(default_factory=time.time)
    status: str = "pending"
    result_fragment: Optional[str] = None

class DistributedTensorKernel:
    """Enhanced tensor kernel with fragment sharing capabilities"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.fragments: Dict[str, TensorFragment] = {}
        self.shared_fragments: Dict[str, TensorFragment] = {}
        self.pending_operations: List[FragmentOperation] = []
        self.operation_history: List[FragmentOperation] = []
        
        # Prime factorization tensor shapes (from echo9ml architecture)
        self.tensor_shape_catalog = {
            "persona": {
                "shape": (3, 7, 13, 5, 2),
                "semantic_dimensions": ["persona_id", "trait_id", "time", "context", "valence"],
                "prime_factors": [3, 7, 13, 5, 2],
                "total_elements": 2730,
                "evolutionary_flexibility": "high"
            },
            "memory": {
                "shape": (101, 8, 5, 7, 3),
                "semantic_dimensions": ["memory_node", "memory_type", "salience", "temporal", "relational"],
                "prime_factors": [101, 8, 5, 7, 3],  # 101 is prime for diversity
                "total_elements": 84840,
                "evolutionary_flexibility": "very_high"
            },
            "attention": {
                "shape": (17, 17, 11, 7, 2),
                "semantic_dimensions": ["source", "target", "strength", "context", "decay"],
                "prime_factors": [17, 17, 11, 7, 2],  # Square matrix for attention spreading
                "total_elements": 44506,
                "evolutionary_flexibility": "high"
            },
            "reasoning": {
                "shape": (23, 5, 7, 3, 2),
                "semantic_dimensions": ["pattern_id", "complexity", "temporal", "context", "validity"],
                "prime_factors": [23, 5, 7, 3, 2],
                "total_elements": 4830,
                "evolutionary_flexibility": "medium"
            },
            "agent_state": {
                "shape": (13, 11, 7, 5, 3),
                "semantic_dimensions": ["agent_id", "component", "temporal", "context", "activity"],
                "prime_factors": [13, 11, 7, 5, 3],
                "total_elements": 15015,
                "evolutionary_flexibility": "high"
            },
            "hypergraph": {
                "shape": (19, 17, 5, 3, 2),
                "semantic_dimensions": ["node_id", "edge_id", "relation", "weight", "direction"],
                "prime_factors": [19, 17, 5, 3, 2],
                "total_elements": 9690,
                "evolutionary_flexibility": "high"
            }
        }
        
        # Fragment compression algorithms
        self.compression_algorithms = {
            FragmentCompressionType.SPARSE: self._compress_sparse,
            FragmentCompressionType.QUANTIZED: self._compress_quantized,
            FragmentCompressionType.DELTA: self._compress_delta,
            FragmentCompressionType.PRIME_FACTORIZED: self._compress_prime_factorized
        }
        
        # Fragment sharing protocols
        self.sharing_protocols = {
            FragmentSharingMode.FULL_FRAGMENT: self._share_full_fragment,
            FragmentSharingMode.SPARSE_UPDATES: self._share_sparse_updates,
            FragmentSharingMode.GRADIENT_ONLY: self._share_gradient_only,
            FragmentSharingMode.SEMANTIC_ONLY: self._share_semantic_only
        }
    
    def create_tensor_fragment(self, tensor_type: str, data: Optional[List[float]] = None, 
                             metadata: Optional[Dict[str, Any]] = None) -> TensorFragment:
        """Create a new tensor fragment of specified type"""
        if tensor_type not in self.tensor_shape_catalog:
            raise ValueError(f"Unknown tensor type: {tensor_type}")
        
        catalog_entry = self.tensor_shape_catalog[tensor_type]
        shape = catalog_entry["shape"]
        semantic_dimensions = catalog_entry["semantic_dimensions"]
        
        # Initialize data if not provided
        if data is None:
            total_elements = catalog_entry["total_elements"]
            data = [0.0] * total_elements
        
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {
                "tensor_type": tensor_type,
                "evolutionary_flexibility": catalog_entry["evolutionary_flexibility"],
                "prime_factors": catalog_entry["prime_factors"]
            }
        
        fragment_id = f"{self.agent_id}_{tensor_type}_{int(time.time() * 1000)}"
        
        fragment = TensorFragment(
            fragment_id=fragment_id,
            source_agent=self.agent_id,
            tensor_name=tensor_type,
            shape=shape,
            semantic_dimensions=semantic_dimensions,
            data=data,
            metadata=metadata
        )
        
        self.fragments[fragment_id] = fragment
        logger.info(f"Created tensor fragment {fragment_id} of type {tensor_type}")
        
        return fragment
    
    def share_fragment(self, fragment_id: str, target_agents: List[str], 
                      sharing_mode: FragmentSharingMode = FragmentSharingMode.FULL_FRAGMENT) -> Dict[str, Any]:
        """Share tensor fragment with target agents"""
        if fragment_id not in self.fragments:
            raise ValueError(f"Fragment {fragment_id} not found")
        
        fragment = self.fragments[fragment_id]
        protocol = self.sharing_protocols.get(sharing_mode, self._share_full_fragment)
        
        shared_data = protocol(fragment)
        
        sharing_result = {
            "fragment_id": fragment_id,
            "source_agent": self.agent_id,
            "target_agents": target_agents,
            "sharing_mode": sharing_mode.value,
            "shared_data": shared_data,
            "timestamp": time.time(),
            "data_size": len(json.dumps(shared_data))
        }
        
        logger.info(f"Shared fragment {fragment_id} with {len(target_agents)} agents using {sharing_mode.value}")
        
        return sharing_result
    
    def receive_fragment(self, shared_data: Dict[str, Any], source_agent: str) -> str:
        """Receive and integrate shared tensor fragment"""
        fragment = TensorFragment.from_dict(shared_data)
        
        # Validate fragment integrity
        if not fragment.validate_integrity():
            raise ValueError(f"Fragment integrity validation failed for {fragment.fragment_id}")
        
        # Store in shared fragments
        self.shared_fragments[fragment.fragment_id] = fragment
        
        logger.info(f"Received fragment {fragment.fragment_id} from {source_agent}")
        
        return fragment.fragment_id
    
    def merge_fragments(self, fragment_ids: List[str], merge_strategy: str = "weighted_average") -> TensorFragment:
        """Merge multiple tensor fragments using specified strategy"""
        if not fragment_ids:
            raise ValueError("No fragments to merge")
        
        # Get fragments to merge
        fragments_to_merge = []
        for fid in fragment_ids:
            if fid in self.fragments:
                fragments_to_merge.append(self.fragments[fid])
            elif fid in self.shared_fragments:
                fragments_to_merge.append(self.shared_fragments[fid])
            else:
                raise ValueError(f"Fragment {fid} not found")
        
        # Validate compatibility
        first_fragment = fragments_to_merge[0]
        for fragment in fragments_to_merge[1:]:
            if fragment.shape != first_fragment.shape:
                raise ValueError("Cannot merge fragments with different shapes")
            if fragment.tensor_name != first_fragment.tensor_name:
                raise ValueError("Cannot merge fragments of different tensor types")
        
        # Perform merge based on strategy
        if merge_strategy == "weighted_average":
            merged_data = self._merge_weighted_average(fragments_to_merge)
        elif merge_strategy == "maximum":
            merged_data = self._merge_maximum(fragments_to_merge)
        elif merge_strategy == "consensus":
            merged_data = self._merge_consensus(fragments_to_merge)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Create merged fragment
        merged_fragment_id = f"{self.agent_id}_merged_{int(time.time() * 1000)}"
        merged_fragment = TensorFragment(
            fragment_id=merged_fragment_id,
            source_agent=self.agent_id,
            tensor_name=first_fragment.tensor_name,
            shape=first_fragment.shape,
            semantic_dimensions=first_fragment.semantic_dimensions,
            data=merged_data,
            metadata={
                "merge_strategy": merge_strategy,
                "source_fragments": fragment_ids,
                "merge_timestamp": time.time()
            }
        )
        
        self.fragments[merged_fragment_id] = merged_fragment
        logger.info(f"Merged {len(fragment_ids)} fragments into {merged_fragment_id}")
        
        return merged_fragment
    
    def execute_distributed_operation(self, operation: FragmentOperation) -> bool:
        """Execute distributed tensor operation"""
        try:
            operation.status = "executing"
            
            # Get source fragments
            source_fragments = []
            for fid in operation.source_fragments:
                if fid in self.fragments:
                    source_fragments.append(self.fragments[fid])
                elif fid in self.shared_fragments:
                    source_fragments.append(self.shared_fragments[fid])
                else:
                    raise ValueError(f"Source fragment {fid} not found")
            
            # Execute operation based on type
            if operation.operation_type == TensorOperationType.PERSONA_EVOLVE:
                result_fragment = self._execute_persona_evolve(source_fragments, operation.parameters)
            elif operation.operation_type == TensorOperationType.ATTENTION_SPREAD:
                result_fragment = self._execute_attention_spread(source_fragments, operation.parameters)
            elif operation.operation_type == TensorOperationType.MEMORY_CONSOLIDATE:
                result_fragment = self._execute_memory_consolidate(source_fragments, operation.parameters)
            elif operation.operation_type == TensorOperationType.REASONING_PROPAGATE:
                result_fragment = self._execute_reasoning_propagate(source_fragments, operation.parameters)
            elif operation.operation_type == TensorOperationType.HYPERGRAPH_ENCODE:
                result_fragment = self._execute_hypergraph_encode(source_fragments, operation.parameters)
            else:
                raise ValueError(f"Unknown operation type: {operation.operation_type}")
            
            # Store result
            self.fragments[result_fragment.fragment_id] = result_fragment
            operation.result_fragment = result_fragment.fragment_id
            operation.status = "completed"
            
            logger.info(f"Executed operation {operation.operation_id} successfully")
            return True
            
        except Exception as e:
            operation.status = f"failed: {str(e)}"
            logger.error(f"Operation {operation.operation_id} failed: {e}")
            return False
        
        finally:
            self.operation_history.append(operation)
    
    def get_tensor_documentation(self) -> Dict[str, Any]:
        """Get comprehensive tensor architecture documentation"""
        return {
            "tensor_shape_catalog": self.tensor_shape_catalog,
            "supported_operations": [op.value for op in TensorOperationType],
            "compression_types": [comp.value for comp in FragmentCompressionType],
            "sharing_modes": [mode.value for mode in FragmentSharingMode],
            "agent_id": self.agent_id,
            "active_fragments": len(self.fragments),
            "shared_fragments": len(self.shared_fragments),
            "pending_operations": len(self.pending_operations),
            "completed_operations": len([op for op in self.operation_history if op.status == "completed"]),
            "prime_factorization_benefits": {
                "evolutionary_flexibility": "Prime factors allow easy tensor reshaping during evolution",
                "computational_efficiency": "Prime dimensions optimize memory access patterns",
                "distributed_processing": "Prime factors enable efficient fragment distribution",
                "genetic_algorithms": "Prime shapes support crossover and mutation operations"
            }
        }
    
    # Compression algorithms
    def _compress_sparse(self, fragment: TensorFragment) -> Dict[str, Any]:
        """Compress fragment using sparse representation"""
        non_zero_indices = []
        non_zero_values = []
        
        for i, value in enumerate(fragment.data):
            if abs(value) > 1e-6:  # Small threshold for zero
                non_zero_indices.append(i)
                non_zero_values.append(value)
        
        return {
            "indices": non_zero_indices,
            "values": non_zero_values,
            "shape": fragment.shape,
            "compression_ratio": len(non_zero_values) / len(fragment.data)
        }
    
    def _compress_quantized(self, fragment: TensorFragment) -> Dict[str, Any]:
        """Compress fragment using quantization"""
        # Simple 8-bit quantization
        min_val = min(fragment.data) if fragment.data else 0.0
        max_val = max(fragment.data) if fragment.data else 1.0
        range_val = max_val - min_val if max_val != min_val else 1.0
        
        quantized_data = []
        for value in fragment.data:
            normalized = (value - min_val) / range_val
            quantized = int(normalized * 255)
            quantized_data.append(max(0, min(255, quantized)))
        
        return {
            "quantized_data": quantized_data,
            "min_value": min_val,
            "max_value": max_val,
            "compression_ratio": 0.25  # 8-bit vs 32-bit
        }
    
    def _compress_delta(self, fragment: TensorFragment) -> Dict[str, Any]:
        """Compress fragment using delta encoding"""
        if not fragment.data:
            return {"deltas": [], "base_value": 0.0}
        
        base_value = fragment.data[0]
        deltas = [base_value]
        
        for i in range(1, len(fragment.data)):
            delta = fragment.data[i] - fragment.data[i-1]
            deltas.append(delta)
        
        return {
            "deltas": deltas,
            "base_value": base_value,
            "compression_ratio": 0.8  # Typically good compression for smooth data
        }
    
    def _compress_prime_factorized(self, fragment: TensorFragment) -> Dict[str, Any]:
        """Compress fragment using prime factorization structure"""
        # Exploit prime factorization structure for compression
        shape = fragment.shape
        if not all(self._is_prime_or_power_of_prime(dim) for dim in shape[:3]):
            return {"error": "Not suitable for prime factorization compression"}
        
        # Group data by prime factor boundaries
        groups = []
        stride = 1
        for dim in reversed(shape):
            stride *= dim
        
        # Simple grouping by first prime dimension
        group_size = len(fragment.data) // shape[0] if shape[0] > 0 else len(fragment.data)
        for i in range(0, len(fragment.data), group_size):
            group = fragment.data[i:i+group_size]
            groups.append(group)
        
        return {
            "groups": groups,
            "group_size": group_size,
            "shape": shape,
            "compression_ratio": 0.9  # Modest compression
        }
    
    def _is_prime_or_power_of_prime(self, n: int) -> bool:
        """Check if number is prime or power of prime"""
        if n < 2:
            return False
        
        # Find smallest prime factor
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                # Check if n is a power of i
                temp = n
                while temp % i == 0:
                    temp //= i
                return temp == 1
        
        return True  # n is prime
    
    # Sharing protocols
    def _share_full_fragment(self, fragment: TensorFragment) -> Dict[str, Any]:
        """Share complete fragment"""
        return fragment.to_dict()
    
    def _share_sparse_updates(self, fragment: TensorFragment) -> Dict[str, Any]:
        """Share only sparse updates"""
        compressed = self._compress_sparse(fragment)
        return {
            "fragment_id": fragment.fragment_id,
            "tensor_name": fragment.tensor_name,
            "shape": fragment.shape,
            "sparse_data": compressed,
            "metadata": fragment.metadata
        }
    
    def _share_gradient_only(self, fragment: TensorFragment) -> Dict[str, Any]:
        """Share only gradient information"""
        # Calculate simple gradient (difference from zero)
        gradients = [abs(value) for value in fragment.data]
        
        return {
            "fragment_id": fragment.fragment_id,
            "tensor_name": fragment.tensor_name,
            "gradients": gradients,
            "metadata": fragment.metadata
        }
    
    def _share_semantic_only(self, fragment: TensorFragment) -> Dict[str, Any]:
        """Share only semantic metadata"""
        return {
            "fragment_id": fragment.fragment_id,
            "tensor_name": fragment.tensor_name,
            "shape": fragment.shape,
            "semantic_dimensions": fragment.semantic_dimensions,
            "metadata": fragment.metadata,
            "data_summary": {
                "mean": sum(fragment.data) / len(fragment.data) if fragment.data else 0.0,
                "min": min(fragment.data) if fragment.data else 0.0,
                "max": max(fragment.data) if fragment.data else 0.0,
                "non_zero_count": sum(1 for x in fragment.data if abs(x) > 1e-6)
            }
        }
    
    # Merge strategies
    def _merge_weighted_average(self, fragments: List[TensorFragment]) -> List[float]:
        """Merge fragments using weighted average"""
        if not fragments:
            return []
        
        data_length = len(fragments[0].data)
        merged_data = [0.0] * data_length
        total_weight = 0.0
        
        for fragment in fragments:
            weight = fragment.metadata.get("weight", 1.0)
            total_weight += weight
            
            for i in range(data_length):
                merged_data[i] += fragment.data[i] * weight
        
        # Normalize by total weight
        if total_weight > 0:
            merged_data = [value / total_weight for value in merged_data]
        
        return merged_data
    
    def _merge_maximum(self, fragments: List[TensorFragment]) -> List[float]:
        """Merge fragments taking maximum values"""
        if not fragments:
            return []
        
        data_length = len(fragments[0].data)
        merged_data = list(fragments[0].data)
        
        for fragment in fragments[1:]:
            for i in range(data_length):
                merged_data[i] = max(merged_data[i], fragment.data[i])
        
        return merged_data
    
    def _merge_consensus(self, fragments: List[TensorFragment]) -> List[float]:
        """Merge fragments using consensus (median)"""
        if not fragments:
            return []
        
        data_length = len(fragments[0].data)
        merged_data = []
        
        for i in range(data_length):
            values = [fragment.data[i] for fragment in fragments]
            values.sort()
            
            # Take median
            n = len(values)
            if n % 2 == 0:
                median = (values[n//2 - 1] + values[n//2]) / 2.0
            else:
                median = values[n//2]
            
            merged_data.append(median)
        
        return merged_data
    
    # Operation implementations
    def _execute_persona_evolve(self, fragments: List[TensorFragment], parameters: Dict[str, Any]) -> TensorFragment:
        """Execute persona evolution operation"""
        if not fragments:
            raise ValueError("No fragments provided for persona evolution")
        
        fragment = fragments[0]
        learning_rate = parameters.get("learning_rate", 0.01)
        
        # Simple evolution: add small random changes weighted by learning rate
        evolved_data = []
        for value in fragment.data:
            # Add small evolution step
            evolution_delta = (hash(str(value)) % 100 - 50) / 50000.0  # Deterministic "random"
            evolved_value = value + evolution_delta * learning_rate
            evolved_data.append(max(-1.0, min(1.0, evolved_value)))  # Clamp to [-1, 1]
        
        result_id = f"{self.agent_id}_persona_evolved_{int(time.time() * 1000)}"
        return TensorFragment(
            fragment_id=result_id,
            source_agent=self.agent_id,
            tensor_name=fragment.tensor_name,
            shape=fragment.shape,
            semantic_dimensions=fragment.semantic_dimensions,
            data=evolved_data,
            metadata={
                "operation": "persona_evolve",
                "learning_rate": learning_rate,
                "source_fragment": fragment.fragment_id
            }
        )
    
    def _execute_attention_spread(self, fragments: List[TensorFragment], parameters: Dict[str, Any]) -> TensorFragment:
        """Execute attention spreading operation"""
        if not fragments:
            raise ValueError("No fragments provided for attention spreading")
        
        fragment = fragments[0]
        spread_factor = parameters.get("spread_factor", 0.1)
        
        # Simple attention spreading: smooth neighboring values
        spread_data = list(fragment.data)
        for i in range(1, len(spread_data) - 1):
            neighbor_avg = (spread_data[i-1] + spread_data[i+1]) / 2.0
            spread_data[i] = spread_data[i] * (1 - spread_factor) + neighbor_avg * spread_factor
        
        result_id = f"{self.agent_id}_attention_spread_{int(time.time() * 1000)}"
        return TensorFragment(
            fragment_id=result_id,
            source_agent=self.agent_id,
            tensor_name=fragment.tensor_name,
            shape=fragment.shape,
            semantic_dimensions=fragment.semantic_dimensions,
            data=spread_data,
            metadata={
                "operation": "attention_spread",
                "spread_factor": spread_factor,
                "source_fragment": fragment.fragment_id
            }
        )
    
    def _execute_memory_consolidate(self, fragments: List[TensorFragment], parameters: Dict[str, Any]) -> TensorFragment:
        """Execute memory consolidation operation"""
        if len(fragments) < 2:
            raise ValueError("Need at least 2 fragments for memory consolidation")
        
        consolidation_strength = parameters.get("consolidation_strength", 0.5)
        
        # Merge fragments with weighted average based on consolidation strength
        fragment1, fragment2 = fragments[0], fragments[1]
        consolidated_data = []
        
        for i in range(len(fragment1.data)):
            consolidated_value = (fragment1.data[i] * (1 - consolidation_strength) + 
                                fragment2.data[i] * consolidation_strength)
            consolidated_data.append(consolidated_value)
        
        result_id = f"{self.agent_id}_memory_consolidated_{int(time.time() * 1000)}"
        return TensorFragment(
            fragment_id=result_id,
            source_agent=self.agent_id,
            tensor_name=fragment1.tensor_name,
            shape=fragment1.shape,
            semantic_dimensions=fragment1.semantic_dimensions,
            data=consolidated_data,
            metadata={
                "operation": "memory_consolidate",
                "consolidation_strength": consolidation_strength,
                "source_fragments": [f.fragment_id for f in fragments[:2]]
            }
        )
    
    def _execute_reasoning_propagate(self, fragments: List[TensorFragment], parameters: Dict[str, Any]) -> TensorFragment:
        """Execute reasoning propagation operation"""
        if not fragments:
            raise ValueError("No fragments provided for reasoning propagation")
        
        fragment = fragments[0]
        propagation_steps = parameters.get("propagation_steps", 3)
        
        # Simple reasoning propagation: iterative smoothing
        propagated_data = list(fragment.data)
        
        for step in range(propagation_steps):
            new_data = list(propagated_data)
            for i in range(len(propagated_data)):
                # Influence from neighbors (circular)
                prev_idx = (i - 1) % len(propagated_data)
                next_idx = (i + 1) % len(propagated_data)
                influence = (propagated_data[prev_idx] + propagated_data[next_idx]) / 2.0
                new_data[i] = propagated_data[i] * 0.8 + influence * 0.2
            propagated_data = new_data
        
        result_id = f"{self.agent_id}_reasoning_propagated_{int(time.time() * 1000)}"
        return TensorFragment(
            fragment_id=result_id,
            source_agent=self.agent_id,
            tensor_name=fragment.tensor_name,
            shape=fragment.shape,
            semantic_dimensions=fragment.semantic_dimensions,
            data=propagated_data,
            metadata={
                "operation": "reasoning_propagate",
                "propagation_steps": propagation_steps,
                "source_fragment": fragment.fragment_id
            }
        )
    
    def _execute_hypergraph_encode(self, fragments: List[TensorFragment], parameters: Dict[str, Any]) -> TensorFragment:
        """Execute hypergraph encoding operation"""
        if not fragments:
            raise ValueError("No fragments provided for hypergraph encoding")
        
        fragment = fragments[0]
        encoding_density = parameters.get("encoding_density", 0.3)
        
        # Create hypergraph encoding tensor
        hypergraph_shape = self.tensor_shape_catalog["hypergraph"]["shape"]
        hypergraph_elements = self.tensor_shape_catalog["hypergraph"]["total_elements"]
        
        # Sample from source fragment to populate hypergraph tensor
        encoded_data = []
        source_data = fragment.data
        
        for i in range(hypergraph_elements):
            # Sample from source with some transformation
            source_idx = i % len(source_data)
            base_value = source_data[source_idx]
            
            # Add encoding transformation
            encoded_value = base_value * encoding_density + (1 - encoding_density) * 0.5
            encoded_data.append(encoded_value)
        
        result_id = f"{self.agent_id}_hypergraph_encoded_{int(time.time() * 1000)}"
        return TensorFragment(
            fragment_id=result_id,
            source_agent=self.agent_id,
            tensor_name="hypergraph",
            shape=hypergraph_shape,
            semantic_dimensions=self.tensor_shape_catalog["hypergraph"]["semantic_dimensions"],
            data=encoded_data,
            metadata={
                "operation": "hypergraph_encode",
                "encoding_density": encoding_density,
                "source_fragment": fragment.fragment_id
            }
        )

def create_distributed_tensor_kernel(agent_id: str) -> DistributedTensorKernel:
    """Factory function to create distributed tensor kernel"""
    return DistributedTensorKernel(agent_id)

# Example usage and testing
if __name__ == "__main__":
    print("Tensor Fragment Architecture Test")
    print("=" * 40)
    
    # Create distributed tensor kernel
    kernel = create_distributed_tensor_kernel("test_agent")
    
    # Create tensor fragments
    persona_fragment = kernel.create_tensor_fragment("persona")
    memory_fragment = kernel.create_tensor_fragment("memory")
    attention_fragment = kernel.create_tensor_fragment("attention")
    
    print(f"Created {len(kernel.fragments)} tensor fragments")
    
    # Test fragment sharing
    sharing_result = kernel.share_fragment(
        persona_fragment.fragment_id, 
        ["agent_2", "agent_3"], 
        FragmentSharingMode.SPARSE_UPDATES
    )
    print(f"Shared fragment with data size: {sharing_result['data_size']} bytes")
    
    # Test fragment operations
    operation = FragmentOperation(
        operation_id="test_op_1",
        operation_type=TensorOperationType.PERSONA_EVOLVE,
        source_fragments=[persona_fragment.fragment_id],
        target_fragment="evolved_persona",
        parameters={"learning_rate": 0.05}
    )
    
    success = kernel.execute_distributed_operation(operation)
    print(f"Operation executed successfully: {success}")
    
    # Get documentation
    docs = kernel.get_tensor_documentation()
    print(f"\nTensor Architecture Documentation:")
    print(f"Active fragments: {docs['active_fragments']}")
    print(f"Completed operations: {docs['completed_operations']}")
    print(f"Supported tensor types: {list(docs['tensor_shape_catalog'].keys())}")