"""
Ko6ml ↔ AtomSpace Translation Adapter for Cognitive Primitives

This module provides bidirectional translation between ko6ml cognitive primitives
and AtomSpace hypergraph patterns. It implements the foundation for Phase 1 of
the Distributed Agentic Cognitive Grammar Network.

Key Features:
- Modular Scheme-inspired adapters for grammar translation
- Round-trip translation validation
- Cognitive primitive mapping to hypergraph structures
- Tensor shape integration with prime factorization
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import existing components
try:
    from symbolic_reasoning import Atom, Link, TruthValue, SymbolicAtomSpace
    from echo9ml import PersonaTraitType, PersonaKernel
    from ggml_tensor_kernel import CognitiveTensor, TensorMetadata
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    logging.warning("Some components not available for import")

logger = logging.getLogger(__name__)

class Ko6mlPrimitiveType(Enum):
    """Ko6ml cognitive primitive types"""
    AGENT_STATE = "agent_state"
    MEMORY_FRAGMENT = "memory_fragment"
    REASONING_PATTERN = "reasoning_pattern"
    ATTENTION_ALLOCATION = "attention_allocation"
    PERSONA_TRAIT = "persona_trait"
    HYPERGRAPH_NODE = "hypergraph_node"
    HYPERGRAPH_LINK = "hypergraph_link"
    TENSOR_FRAGMENT = "tensor_fragment"

@dataclass
class Ko6mlPrimitive:
    """Ko6ml cognitive primitive representation"""
    primitive_id: str
    primitive_type: Ko6mlPrimitiveType
    content: Dict[str, Any]
    truth_value: Optional[Tuple[float, float]] = None  # (strength, confidence)
    tensor_signature: Optional[Tuple[int, ...]] = None
    creation_time: float = field(default_factory=time.time)
    
    def to_scheme_expr(self) -> str:
        """Convert to Scheme-like expression"""
        tv_str = ""
        if self.truth_value:
            tv_str = f" (tv {self.truth_value[0]:.3f} {self.truth_value[1]:.3f})"
        
        tensor_str = ""
        if self.tensor_signature:
            tensor_str = f" (tensor-shape {' '.join(map(str, self.tensor_signature))})"
            
        return f"({self.primitive_type.value} {self.primitive_id} {json.dumps(self.content)}{tv_str}{tensor_str})"

@dataclass
class AtomSpaceFragment:
    """AtomSpace hypergraph fragment"""
    atoms: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    fragment_id: str
    source_primitive: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "fragment_id": self.fragment_id,
            "atoms": self.atoms,
            "links": self.links,
            "source_primitive": self.source_primitive
        }

class Ko6mlAtomSpaceAdapter:
    """Bidirectional translator between ko6ml primitives and AtomSpace hypergraphs"""
    
    def __init__(self):
        self.primitive_mappings: Dict[Ko6mlPrimitiveType, callable] = {}
        self.atomspace_mappings: Dict[str, callable] = {}
        self.translation_history: List[Dict[str, Any]] = []
        
        # Initialize translation mappings
        self._initialize_primitive_mappings()
        self._initialize_atomspace_mappings()
        
        # Prime factorization tensor shapes from echo9ml
        self.tensor_shapes = {
            "persona": (3, 7, 13, 5, 2),  # 2730 elements
            "memory": (101, 8, 5, 7, 3),  # 84,840 elements  
            "attention": (17, 17, 11, 7, 2),  # 44,506 elements
            "reasoning": (23, 5, 7, 3, 2),  # 4,830 elements
            "agent_state": (13, 11, 7, 5, 3)  # 15,015 elements
        }
    
    def _initialize_primitive_mappings(self):
        """Initialize ko6ml primitive to AtomSpace mappings"""
        self.primitive_mappings = {
            Ko6mlPrimitiveType.AGENT_STATE: self._map_agent_state_to_atomspace,
            Ko6mlPrimitiveType.MEMORY_FRAGMENT: self._map_memory_fragment_to_atomspace,
            Ko6mlPrimitiveType.REASONING_PATTERN: self._map_reasoning_pattern_to_atomspace,
            Ko6mlPrimitiveType.ATTENTION_ALLOCATION: self._map_attention_allocation_to_atomspace,
            Ko6mlPrimitiveType.PERSONA_TRAIT: self._map_persona_trait_to_atomspace,
            Ko6mlPrimitiveType.HYPERGRAPH_NODE: self._map_hypergraph_node_to_atomspace,
            Ko6mlPrimitiveType.HYPERGRAPH_LINK: self._map_hypergraph_link_to_atomspace,
            Ko6mlPrimitiveType.TENSOR_FRAGMENT: self._map_tensor_fragment_to_atomspace
        }
    
    def _initialize_atomspace_mappings(self):
        """Initialize AtomSpace to ko6ml primitive mappings"""
        self.atomspace_mappings = {
            "ConceptNode": self._map_concept_node_to_primitive,
            "PredicateNode": self._map_predicate_node_to_primitive,
            "InheritanceLink": self._map_inheritance_link_to_primitive,
            "SimilarityLink": self._map_similarity_link_to_primitive,
            "EvaluationLink": self._map_evaluation_link_to_primitive,
            "AndLink": self._map_and_link_to_primitive,
            "OrLink": self._map_or_link_to_primitive
        }
    
    def ko6ml_to_atomspace(self, primitive: Ko6mlPrimitive) -> AtomSpaceFragment:
        """Translate ko6ml primitive to AtomSpace hypergraph fragment"""
        logger.info(f"Translating ko6ml primitive {primitive.primitive_id} to AtomSpace")
        
        if primitive.primitive_type not in self.primitive_mappings:
            raise ValueError(f"Unknown primitive type: {primitive.primitive_type}")
        
        translator = self.primitive_mappings[primitive.primitive_type]
        fragment = translator(primitive)
        
        # Record translation
        self.translation_history.append({
            "direction": "ko6ml_to_atomspace",
            "primitive_id": primitive.primitive_id,
            "fragment_id": fragment.fragment_id,
            "timestamp": time.time()
        })
        
        return fragment
    
    def atomspace_to_ko6ml(self, fragment: AtomSpaceFragment) -> List[Ko6mlPrimitive]:
        """Translate AtomSpace hypergraph fragment to ko6ml primitives"""
        logger.info(f"Translating AtomSpace fragment {fragment.fragment_id} to ko6ml")
        
        primitives = []
        
        # Process atoms
        for atom in fragment.atoms:
            atom_type = atom.get("type", "ConceptNode")
            if atom_type in self.atomspace_mappings:
                translator = self.atomspace_mappings[atom_type]
                primitive = translator(atom, fragment)
                primitives.append(primitive)
        
        # Process links
        for link in fragment.links:
            link_type = link.get("type", "InheritanceLink")
            if link_type in self.atomspace_mappings:
                translator = self.atomspace_mappings[link_type]
                primitive = translator(link, fragment)
                primitives.append(primitive)
        
        # Record translation
        for primitive in primitives:
            self.translation_history.append({
                "direction": "atomspace_to_ko6ml",
                "fragment_id": fragment.fragment_id,
                "primitive_id": primitive.primitive_id,
                "timestamp": time.time()
            })
        
        return primitives
    
    def validate_round_trip(self, original_primitive: Ko6mlPrimitive) -> Dict[str, Any]:
        """Validate round-trip translation ko6ml → AtomSpace → ko6ml"""
        logger.info(f"Validating round-trip for primitive {original_primitive.primitive_id}")
        
        try:
            # Forward translation
            atomspace_fragment = self.ko6ml_to_atomspace(original_primitive)
            
            # Backward translation
            recovered_primitives = self.atomspace_to_ko6ml(atomspace_fragment)
            
            # Find the most similar recovered primitive
            best_match = None
            best_similarity = 0.0
            
            for recovered in recovered_primitives:
                similarity = self._calculate_primitive_similarity(original_primitive, recovered)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = recovered
            
            success = best_similarity >= 0.8  # 80% similarity threshold
            
            return {
                "success": success,
                "similarity": best_similarity,
                "original": original_primitive,
                "recovered": best_match,
                "atomspace_fragment": atomspace_fragment,
                "all_recovered": recovered_primitives
            }
            
        except Exception as e:
            logger.error(f"Round-trip validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "original": original_primitive
            }
    
    def _calculate_primitive_similarity(self, p1: Ko6mlPrimitive, p2: Ko6mlPrimitive) -> float:
        """Calculate similarity between two ko6ml primitives"""
        if p1.primitive_type != p2.primitive_type:
            return 0.0
        
        # Check content similarity
        content_keys1 = set(p1.content.keys())
        content_keys2 = set(p2.content.keys())
        
        if not content_keys1 or not content_keys2:
            return 0.5
        
        key_overlap = len(content_keys1.intersection(content_keys2))
        key_total = len(content_keys1.union(content_keys2))
        
        key_similarity = key_overlap / key_total if key_total > 0 else 0.0
        
        # Check truth value similarity if both have them
        tv_similarity = 1.0
        if p1.truth_value and p2.truth_value:
            strength_diff = abs(p1.truth_value[0] - p2.truth_value[0])
            confidence_diff = abs(p1.truth_value[1] - p2.truth_value[1])
            tv_similarity = 1.0 - (strength_diff + confidence_diff) / 2.0
        
        return (key_similarity + tv_similarity) / 2.0
    
    # Ko6ml to AtomSpace translation methods
    def _map_agent_state_to_atomspace(self, primitive: Ko6mlPrimitive) -> AtomSpaceFragment:
        """Map agent state primitive to AtomSpace representation"""
        agent_id = primitive.content.get("agent_id", "unknown_agent")
        state_data = primitive.content.get("state", {})
        
        atoms = [{
            "id": f"agent_{agent_id}",
            "type": "ConceptNode",
            "name": f"Agent_{agent_id}",
            "truth_value": primitive.truth_value or (0.9, 0.8)
        }]
        
        links = []
        
        # Create state property links
        for key, value in state_data.items():
            property_atom = {
                "id": f"property_{key}",
                "type": "PredicateNode", 
                "name": key,
                "truth_value": (0.8, 0.9)
            }
            atoms.append(property_atom)
            
            value_atom = {
                "id": f"value_{key}_{value}",
                "type": "ConceptNode",
                "name": str(value),
                "truth_value": (0.7, 0.8)
            }
            atoms.append(value_atom)
            
            # Evaluation link
            eval_link = {
                "id": f"eval_{agent_id}_{key}",
                "type": "EvaluationLink",
                "outgoing": [f"property_{key}", f"agent_{agent_id}", f"value_{key}_{value}"],
                "truth_value": (0.8, 0.8)
            }
            links.append(eval_link)
        
        return AtomSpaceFragment(
            atoms=atoms,
            links=links,
            fragment_id=f"agent_state_{primitive.primitive_id}",
            source_primitive=primitive.primitive_id
        )
    
    def _map_memory_fragment_to_atomspace(self, primitive: Ko6mlPrimitive) -> AtomSpaceFragment:
        """Map memory fragment to AtomSpace representation"""
        memory_content = primitive.content.get("content", "")
        memory_type = primitive.content.get("type", "episodic")
        salience = primitive.content.get("salience", 0.5)
        
        atoms = [{
            "id": f"memory_{primitive.primitive_id}",
            "type": "ConceptNode",
            "name": f"Memory_{primitive.primitive_id}",
            "truth_value": (salience, 0.8)
        }, {
            "id": f"memory_type_{memory_type}",
            "type": "ConceptNode",
            "name": memory_type,
            "truth_value": (0.9, 0.9)
        }]
        
        links = [{
            "id": f"memory_inheritance_{primitive.primitive_id}",
            "type": "InheritanceLink",
            "outgoing": [f"memory_{primitive.primitive_id}", f"memory_type_{memory_type}"],
            "truth_value": (0.8, 0.8)
        }]
        
        return AtomSpaceFragment(
            atoms=atoms,
            links=links,
            fragment_id=f"memory_{primitive.primitive_id}",
            source_primitive=primitive.primitive_id
        )
    
    def _map_reasoning_pattern_to_atomspace(self, primitive: Ko6mlPrimitive) -> AtomSpaceFragment:
        """Map reasoning pattern to AtomSpace representation"""
        pattern_type = primitive.content.get("pattern_type", "inference")
        premises = primitive.content.get("premises", [])
        conclusion = primitive.content.get("conclusion", "")
        
        atoms = [{
            "id": f"reasoning_{primitive.primitive_id}",
            "type": "PredicateNode",
            "name": f"ReasoningPattern_{pattern_type}",
            "truth_value": primitive.truth_value or (0.8, 0.7)
        }]
        
        links = []
        
        # Add premise atoms and links
        for i, premise in enumerate(premises):
            premise_atom = {
                "id": f"premise_{primitive.primitive_id}_{i}",
                "type": "ConceptNode",
                "name": str(premise),
                "truth_value": (0.7, 0.8)
            }
            atoms.append(premise_atom)
        
        return AtomSpaceFragment(
            atoms=atoms,
            links=links,
            fragment_id=f"reasoning_{primitive.primitive_id}",
            source_primitive=primitive.primitive_id
        )
    
    def _map_attention_allocation_to_atomspace(self, primitive: Ko6mlPrimitive) -> AtomSpaceFragment:
        """Map attention allocation to AtomSpace representation"""
        allocation = primitive.content.get("allocation", {})
        total_attention = primitive.content.get("total_attention", 100.0)
        
        atoms = [{
            "id": f"attention_system_{primitive.primitive_id}",
            "type": "ConceptNode", 
            "name": "AttentionSystem",
            "truth_value": (0.9, 0.9)
        }]
        
        links = []
        
        for item, attention_value in allocation.items():
            item_atom = {
                "id": f"attention_item_{item}",
                "type": "ConceptNode",
                "name": item,
                "truth_value": (attention_value / total_attention, 0.8)
            }
            atoms.append(item_atom)
            
            # Attention link
            attention_link = {
                "id": f"attention_link_{primitive.primitive_id}_{item}",
                "type": "EvaluationLink",
                "outgoing": ["attention_predicate", f"attention_item_{item}"],
                "truth_value": (attention_value / total_attention, 0.9)
            }
            links.append(attention_link)
        
        return AtomSpaceFragment(
            atoms=atoms,
            links=links,
            fragment_id=f"attention_{primitive.primitive_id}",
            source_primitive=primitive.primitive_id
        )
    
    def _map_persona_trait_to_atomspace(self, primitive: Ko6mlPrimitive) -> AtomSpaceFragment:
        """Map persona trait to AtomSpace representation"""
        trait_type = primitive.content.get("trait_type", "unknown")
        trait_value = primitive.content.get("value", 0.5)
        persona_id = primitive.content.get("persona_id", "default")
        
        atoms = [{
            "id": f"persona_{persona_id}",
            "type": "ConceptNode",
            "name": f"Persona_{persona_id}",
            "truth_value": (0.9, 0.9)
        }, {
            "id": f"trait_{trait_type}",
            "type": "PredicateNode",
            "name": trait_type,
            "truth_value": (trait_value, 0.8)
        }]
        
        links = [{
            "id": f"trait_eval_{primitive.primitive_id}",
            "type": "EvaluationLink", 
            "outgoing": [f"trait_{trait_type}", f"persona_{persona_id}"],
            "truth_value": (trait_value, 0.8)
        }]
        
        return AtomSpaceFragment(
            atoms=atoms,
            links=links,
            fragment_id=f"trait_{primitive.primitive_id}",
            source_primitive=primitive.primitive_id
        )
    
    def _map_hypergraph_node_to_atomspace(self, primitive: Ko6mlPrimitive) -> AtomSpaceFragment:
        """Map hypergraph node to AtomSpace representation"""
        node_data = primitive.content
        
        atoms = [{
            "id": primitive.primitive_id,
            "type": node_data.get("type", "ConceptNode"),
            "name": node_data.get("name", primitive.primitive_id),
            "truth_value": primitive.truth_value or (0.8, 0.8)
        }]
        
        return AtomSpaceFragment(
            atoms=atoms,
            links=[],
            fragment_id=f"node_{primitive.primitive_id}",
            source_primitive=primitive.primitive_id
        )
    
    def _map_hypergraph_link_to_atomspace(self, primitive: Ko6mlPrimitive) -> AtomSpaceFragment:
        """Map hypergraph link to AtomSpace representation"""
        link_data = primitive.content
        
        links = [{
            "id": primitive.primitive_id,
            "type": link_data.get("type", "InheritanceLink"),
            "outgoing": link_data.get("outgoing", []),
            "truth_value": primitive.truth_value or (0.8, 0.8)
        }]
        
        return AtomSpaceFragment(
            atoms=[],
            links=links,
            fragment_id=f"link_{primitive.primitive_id}",
            source_primitive=primitive.primitive_id
        )
    
    def _map_tensor_fragment_to_atomspace(self, primitive: Ko6mlPrimitive) -> AtomSpaceFragment:
        """Map tensor fragment to AtomSpace representation"""
        tensor_data = primitive.content
        tensor_shape = tensor_data.get("shape", primitive.tensor_signature)
        
        atoms = [{
            "id": f"tensor_{primitive.primitive_id}",
            "type": "ConceptNode",
            "name": f"TensorFragment_{primitive.primitive_id}",
            "truth_value": primitive.truth_value or (0.8, 0.8)
        }]
        
        # Add shape information as properties
        if tensor_shape:
            for i, dim in enumerate(tensor_shape):
                dim_atom = {
                    "id": f"tensor_dim_{primitive.primitive_id}_{i}",
                    "type": "ConceptNode",
                    "name": f"Dimension_{i}_{dim}",
                    "truth_value": (0.9, 0.9)
                }
                atoms.append(dim_atom)
        
        return AtomSpaceFragment(
            atoms=atoms,
            links=[],
            fragment_id=f"tensor_{primitive.primitive_id}",
            source_primitive=primitive.primitive_id
        )
    
    # AtomSpace to Ko6ml translation methods
    def _map_concept_node_to_primitive(self, atom: Dict[str, Any], fragment: AtomSpaceFragment) -> Ko6mlPrimitive:
        """Map ConceptNode to ko6ml primitive"""
        return Ko6mlPrimitive(
            primitive_id=atom["id"],
            primitive_type=Ko6mlPrimitiveType.HYPERGRAPH_NODE,
            content={
                "type": "ConceptNode",
                "name": atom.get("name", atom["id"]),
                "atomspace_type": "concept"
            },
            truth_value=atom.get("truth_value")
        )
    
    def _map_predicate_node_to_primitive(self, atom: Dict[str, Any], fragment: AtomSpaceFragment) -> Ko6mlPrimitive:
        """Map PredicateNode to ko6ml primitive"""
        return Ko6mlPrimitive(
            primitive_id=atom["id"],
            primitive_type=Ko6mlPrimitiveType.HYPERGRAPH_NODE,
            content={
                "type": "PredicateNode", 
                "name": atom.get("name", atom["id"]),
                "atomspace_type": "predicate"
            },
            truth_value=atom.get("truth_value")
        )
    
    def _map_inheritance_link_to_primitive(self, link: Dict[str, Any], fragment: AtomSpaceFragment) -> Ko6mlPrimitive:
        """Map InheritanceLink to ko6ml primitive"""
        return Ko6mlPrimitive(
            primitive_id=link["id"],
            primitive_type=Ko6mlPrimitiveType.HYPERGRAPH_LINK,
            content={
                "type": "InheritanceLink",
                "outgoing": link.get("outgoing", []),
                "atomspace_type": "inheritance"
            },
            truth_value=link.get("truth_value")
        )
    
    def _map_similarity_link_to_primitive(self, link: Dict[str, Any], fragment: AtomSpaceFragment) -> Ko6mlPrimitive:
        """Map SimilarityLink to ko6ml primitive"""
        return Ko6mlPrimitive(
            primitive_id=link["id"],
            primitive_type=Ko6mlPrimitiveType.HYPERGRAPH_LINK,
            content={
                "type": "SimilarityLink",
                "outgoing": link.get("outgoing", []),
                "atomspace_type": "similarity"
            },
            truth_value=link.get("truth_value")
        )
    
    def _map_evaluation_link_to_primitive(self, link: Dict[str, Any], fragment: AtomSpaceFragment) -> Ko6mlPrimitive:
        """Map EvaluationLink to ko6ml primitive"""
        return Ko6mlPrimitive(
            primitive_id=link["id"],
            primitive_type=Ko6mlPrimitiveType.HYPERGRAPH_LINK,
            content={
                "type": "EvaluationLink",
                "outgoing": link.get("outgoing", []),
                "atomspace_type": "evaluation"
            },
            truth_value=link.get("truth_value")
        )
    
    def _map_and_link_to_primitive(self, link: Dict[str, Any], fragment: AtomSpaceFragment) -> Ko6mlPrimitive:
        """Map AndLink to ko6ml primitive"""
        return Ko6mlPrimitive(
            primitive_id=link["id"],
            primitive_type=Ko6mlPrimitiveType.REASONING_PATTERN,
            content={
                "pattern_type": "conjunction",
                "operands": link.get("outgoing", []),
                "atomspace_type": "logical_and"
            },
            truth_value=link.get("truth_value")
        )
    
    def _map_or_link_to_primitive(self, link: Dict[str, Any], fragment: AtomSpaceFragment) -> Ko6mlPrimitive:
        """Map OrLink to ko6ml primitive"""
        return Ko6mlPrimitive(
            primitive_id=link["id"],
            primitive_type=Ko6mlPrimitiveType.REASONING_PATTERN,
            content={
                "pattern_type": "disjunction", 
                "operands": link.get("outgoing", []),
                "atomspace_type": "logical_or"
            },
            truth_value=link.get("truth_value")
        )
    
    def get_tensor_shape_documentation(self) -> Dict[str, Any]:
        """Get documentation of tensor shapes with prime factorization"""
        return {
            "tensor_shapes": self.tensor_shapes,
            "prime_factorization_rationale": {
                "persona": "3×7×13×5×2 = 2730 elements. Prime factors allow evolutionary flexibility",
                "memory": "101×8×5×7×3 = 84,840 elements. Large prime (101) for memory diversity",
                "attention": "17×17×11×7×2 = 44,506 elements. Square matrix (17×17) for attention spreading",
                "reasoning": "23×5×7×3×2 = 4,830 elements. Prime 23 for reasoning pattern diversity",
                "agent_state": "13×11×7×5×3 = 15,015 elements. Balanced primes for state representation"
            },
            "semantic_mappings": {
                "persona": ["persona_id", "trait_id", "time", "context", "valence"],
                "memory": ["memory_node", "memory_type", "salience", "temporal", "relational"], 
                "attention": ["source", "target", "strength", "context", "decay"],
                "reasoning": ["pattern_id", "complexity", "temporal", "context", "validity"],
                "agent_state": ["agent_id", "component", "temporal", "context", "activity"]
            }
        }

def create_ko6ml_adapter() -> Ko6mlAtomSpaceAdapter:
    """Factory function to create ko6ml adapter"""
    return Ko6mlAtomSpaceAdapter()

# Example usage and test patterns
def create_test_primitives() -> List[Ko6mlPrimitive]:
    """Create test ko6ml primitives for validation"""
    primitives = [
        Ko6mlPrimitive(
            primitive_id="agent_001",
            primitive_type=Ko6mlPrimitiveType.AGENT_STATE,
            content={
                "agent_id": "cognitive_agent_1",
                "state": {
                    "active": True,
                    "attention_level": 0.8,
                    "processing_load": 0.6
                }
            },
            truth_value=(0.9, 0.8),
            tensor_signature=(13, 11, 7, 5, 3)
        ),
        Ko6mlPrimitive(
            primitive_id="memory_001", 
            primitive_type=Ko6mlPrimitiveType.MEMORY_FRAGMENT,
            content={
                "content": "Learning about tensor operations",
                "type": "episodic",
                "salience": 0.7
            },
            truth_value=(0.7, 0.9),
            tensor_signature=(101, 8, 5, 7, 3)
        ),
        Ko6mlPrimitive(
            primitive_id="reasoning_001",
            primitive_type=Ko6mlPrimitiveType.REASONING_PATTERN,
            content={
                "pattern_type": "modus_ponens",
                "premises": ["If A then B", "A"],
                "conclusion": "B"
            },
            truth_value=(0.8, 0.7),
            tensor_signature=(23, 5, 7, 3, 2)
        ),
        Ko6mlPrimitive(
            primitive_id="attention_001",
            primitive_type=Ko6mlPrimitiveType.ATTENTION_ALLOCATION,
            content={
                "allocation": {
                    "reasoning_task": 40.0,
                    "memory_consolidation": 30.0,
                    "sensory_processing": 20.0,
                    "meta_cognition": 10.0
                },
                "total_attention": 100.0
            },
            truth_value=(0.9, 0.9),
            tensor_signature=(17, 17, 11, 7, 2)
        ),
        Ko6mlPrimitive(
            primitive_id="trait_creativity",
            primitive_type=Ko6mlPrimitiveType.PERSONA_TRAIT,
            content={
                "trait_type": "creativity",
                "value": 0.8,
                "persona_id": "deep_tree_echo"
            },
            truth_value=(0.8, 0.8),
            tensor_signature=(3, 7, 13, 5, 2)
        )
    ]
    
    return primitives

if __name__ == "__main__":
    # Test the adapter
    adapter = create_ko6ml_adapter()
    test_primitives = create_test_primitives()
    
    print("Ko6ml ↔ AtomSpace Translation Adapter Test")
    print("=" * 50)
    
    for primitive in test_primitives:
        print(f"\nTesting primitive: {primitive.primitive_id}")
        print(f"Type: {primitive.primitive_type}")
        print(f"Scheme expression: {primitive.to_scheme_expr()}")
        
        # Test round-trip translation
        result = adapter.validate_round_trip(primitive)
        print(f"Round-trip success: {result['success']}")
        if result['success']:
            print(f"Similarity: {result['similarity']:.3f}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Print tensor documentation
    print("\n" + "=" * 50)
    print("Tensor Shape Documentation:")
    tensor_docs = adapter.get_tensor_shape_documentation()
    for shape_name, shape in tensor_docs["tensor_shapes"].items():
        print(f"{shape_name}: {shape} = {tensor_docs['prime_factorization_rationale'][shape_name]}")