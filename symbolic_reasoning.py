"""
Symbolic Reasoning and Pattern Matching for Distributed Cognitive Grammar

This module implements PLN-inspired (Probabilistic Logic Networks) symbolic reasoning
and pattern matching capabilities for the distributed cognitive grammar system.

Based on the OpenCog PLN framework adapted for the Echo9ML distributed architecture.
"""

import re
import json
import time
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, namedtuple
from pathlib import Path

logger = logging.getLogger(__name__)

class LogicalOperator(Enum):
    """Logical operators for symbolic reasoning"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    EQUIVALENT = "equivalent"
    SIMILARITY = "similarity"
    INHERITANCE = "inheritance"
    EVALUATION = "evaluation"

class TruthValue(namedtuple("TruthValue", ["strength", "confidence"])):
    """Truth value representation with strength and confidence"""
    
    def __new__(cls, strength: float = 0.5, confidence: float = 0.5):
        # Ensure values are in [0, 1] range
        strength = max(0.0, min(1.0, strength))
        confidence = max(0.0, min(1.0, confidence))
        return super().__new__(cls, strength, confidence)
    
    def __str__(self):
        return f"<{self.strength:.3f}, {self.confidence:.3f}>"
    
    def to_dict(self):
        return {"strength": self.strength, "confidence": self.confidence}

@dataclass
class Atom:
    """Basic atom in the symbolic reasoning system"""
    name: str
    atom_type: str
    truth_value: TruthValue = field(default_factory=TruthValue)
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    
    def __hash__(self):
        return hash((self.name, self.atom_type))
    
    def __eq__(self, other):
        if not isinstance(other, Atom):
            return False
        return self.name == other.name and self.atom_type == other.atom_type
    
    def to_dict(self):
        return {
            "name": self.name,
            "atom_type": self.atom_type,
            "truth_value": self.truth_value.to_dict(),
            "metadata": self.metadata,
            "creation_time": self.creation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        tv_data = data.get("truth_value", {})
        truth_value = TruthValue(
            strength=tv_data.get("strength", 0.5),
            confidence=tv_data.get("confidence", 0.5)
        )
        
        return cls(
            name=data["name"],
            atom_type=data["atom_type"],
            truth_value=truth_value,
            metadata=data.get("metadata", {}),
            creation_time=data.get("creation_time", time.time())
        )

@dataclass
class Link:
    """Link between atoms in the symbolic reasoning system"""
    link_type: str
    outgoing: List[Atom]
    truth_value: TruthValue = field(default_factory=TruthValue)
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    
    def __hash__(self):
        return hash((self.link_type, tuple(self.outgoing)))
    
    def __eq__(self, other):
        if not isinstance(other, Link):
            return False
        return (self.link_type == other.link_type and 
                self.outgoing == other.outgoing)
    
    def to_dict(self):
        return {
            "link_type": self.link_type,
            "outgoing": [atom.to_dict() for atom in self.outgoing],
            "truth_value": self.truth_value.to_dict(),
            "metadata": self.metadata,
            "creation_time": self.creation_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        tv_data = data.get("truth_value", {})
        truth_value = TruthValue(
            strength=tv_data.get("strength", 0.5),
            confidence=tv_data.get("confidence", 0.5)
        )
        
        outgoing = [Atom.from_dict(atom_data) for atom_data in data.get("outgoing", [])]
        
        return cls(
            link_type=data["link_type"],
            outgoing=outgoing,
            truth_value=truth_value,
            metadata=data.get("metadata", {}),
            creation_time=data.get("creation_time", time.time())
        )

@dataclass
class Pattern:
    """Pattern for matching in the symbolic reasoning system"""
    pattern_type: str
    variables: List[str]
    constraints: List[Dict[str, Any]]
    template: Dict[str, Any]
    
    def matches(self, candidate: Union[Atom, Link]) -> bool:
        """Check if candidate matches this pattern"""
        # Simple pattern matching - can be extended
        if isinstance(candidate, Atom):
            return candidate.atom_type == self.pattern_type
        elif isinstance(candidate, Link):
            return candidate.link_type == self.pattern_type
        return False
    
    def to_dict(self):
        return {
            "pattern_type": self.pattern_type,
            "variables": self.variables,
            "constraints": self.constraints,
            "template": self.template
        }

@dataclass
class Rule:
    """Inference rule in the symbolic reasoning system"""
    name: str
    premise_patterns: List[Pattern]
    conclusion_pattern: Pattern
    strength: float = 1.0
    confidence: float = 1.0
    
    def to_dict(self):
        return {
            "name": self.name,
            "premise_patterns": [p.to_dict() for p in self.premise_patterns],
            "conclusion_pattern": self.conclusion_pattern.to_dict(),
            "strength": self.strength,
            "confidence": self.confidence
        }

class SymbolicAtomSpace:
    """Symbolic reasoning atom space for distributed cognitive grammar"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.atoms: Dict[str, Atom] = {}
        self.links: Dict[str, Link] = {}
        self.patterns: Dict[str, Pattern] = {}
        self.rules: Dict[str, Rule] = {}
        self.attention_values: Dict[str, float] = {}
        
        # Initialize basic patterns and rules
        self._initialize_basic_patterns()
        self._initialize_basic_rules()
        
        logger.info(f"Initialized symbolic atom space for agent {agent_id}")
    
    def _initialize_basic_patterns(self):
        """Initialize basic patterns for cognitive reasoning"""
        # Similarity pattern
        similarity_pattern = Pattern(
            pattern_type="SimilarityLink",
            variables=["$X", "$Y"],
            constraints=[
                {"type": "atom_type", "value": "ConceptNode"},
                {"type": "truth_value", "min_strength": 0.5}
            ],
            template={
                "link_type": "SimilarityLink",
                "outgoing": ["$X", "$Y"]
            }
        )
        self.patterns["similarity"] = similarity_pattern
        
        # Inheritance pattern
        inheritance_pattern = Pattern(
            pattern_type="InheritanceLink",
            variables=["$X", "$Y"],
            constraints=[
                {"type": "atom_type", "value": "ConceptNode"}
            ],
            template={
                "link_type": "InheritanceLink",
                "outgoing": ["$X", "$Y"]
            }
        )
        self.patterns["inheritance"] = inheritance_pattern
        
        # Evaluation pattern
        evaluation_pattern = Pattern(
            pattern_type="EvaluationLink",
            variables=["$P", "$X"],
            constraints=[
                {"type": "atom_type", "value": "PredicateNode"}
            ],
            template={
                "link_type": "EvaluationLink",
                "outgoing": ["$P", "$X"]
            }
        )
        self.patterns["evaluation"] = evaluation_pattern
    
    def _initialize_basic_rules(self):
        """Initialize basic inference rules"""
        # Transitivity rule for inheritance
        transitivity_rule = Rule(
            name="inheritance_transitivity",
            premise_patterns=[
                Pattern(
                    pattern_type="InheritanceLink",
                    variables=["$X", "$Y"],
                    constraints=[],
                    template={"link_type": "InheritanceLink", "outgoing": ["$X", "$Y"]}
                ),
                Pattern(
                    pattern_type="InheritanceLink",
                    variables=["$Y", "$Z"],
                    constraints=[],
                    template={"link_type": "InheritanceLink", "outgoing": ["$Y", "$Z"]}
                )
            ],
            conclusion_pattern=Pattern(
                pattern_type="InheritanceLink",
                variables=["$X", "$Z"],
                constraints=[],
                template={"link_type": "InheritanceLink", "outgoing": ["$X", "$Z"]}
            ),
            strength=0.9,
            confidence=0.8
        )
        self.rules["inheritance_transitivity"] = transitivity_rule
        
        # Similarity symmetry rule
        similarity_symmetry_rule = Rule(
            name="similarity_symmetry",
            premise_patterns=[
                Pattern(
                    pattern_type="SimilarityLink",
                    variables=["$X", "$Y"],
                    constraints=[],
                    template={"link_type": "SimilarityLink", "outgoing": ["$X", "$Y"]}
                )
            ],
            conclusion_pattern=Pattern(
                pattern_type="SimilarityLink",
                variables=["$Y", "$X"],
                constraints=[],
                template={"link_type": "SimilarityLink", "outgoing": ["$Y", "$X"]}
            ),
            strength=1.0,
            confidence=0.9
        )
        self.rules["similarity_symmetry"] = similarity_symmetry_rule
    
    def add_atom(self, atom: Atom):
        """Add atom to the atom space"""
        self.atoms[atom.name] = atom
        self.attention_values[atom.name] = 0.5  # Default attention
        logger.debug(f"Added atom: {atom.name} ({atom.atom_type})")
    
    def add_link(self, link: Link):
        """Add link to the atom space"""
        link_id = f"{link.link_type}_{hash(link)}"
        self.links[link_id] = link
        
        # Add outgoing atoms if not already present
        for atom in link.outgoing:
            if atom.name not in self.atoms:
                self.add_atom(atom)
        
        logger.debug(f"Added link: {link.link_type} with {len(link.outgoing)} atoms")
    
    def get_atom(self, name: str) -> Optional[Atom]:
        """Get atom by name"""
        return self.atoms.get(name)
    
    def get_atoms_by_type(self, atom_type: str) -> List[Atom]:
        """Get all atoms of a specific type"""
        return [atom for atom in self.atoms.values() if atom.atom_type == atom_type]
    
    def get_links_by_type(self, link_type: str) -> List[Link]:
        """Get all links of a specific type"""
        return [link for link in self.links.values() if link.link_type == link_type]
    
    def search_atoms(self, query: str, max_results: int = 10) -> List[Atom]:
        """Search atoms by name pattern"""
        pattern = re.compile(query, re.IGNORECASE)
        results = []
        
        for atom in self.atoms.values():
            if pattern.search(atom.name):
                results.append(atom)
                if len(results) >= max_results:
                    break
        
        return results
    
    def pattern_match(self, pattern: Pattern) -> List[Dict[str, Any]]:
        """Find matches for a pattern in the atom space"""
        matches = []
        
        if pattern.pattern_type.endswith("Link"):
            # Match links
            for link in self.links.values():
                if pattern.matches(link):
                    matches.append({
                        "type": "link",
                        "object": link,
                        "bindings": {}  # TODO: Implement variable binding
                    })
        else:
            # Match atoms
            for atom in self.atoms.values():
                if pattern.matches(atom):
                    matches.append({
                        "type": "atom",
                        "object": atom,
                        "bindings": {}
                    })
        
        return matches
    
    def apply_rule(self, rule: Rule) -> List[Union[Atom, Link]]:
        """Apply inference rule to generate new atoms/links"""
        new_items = []
        
        # Find matches for all premise patterns
        premise_matches = []
        for pattern in rule.premise_patterns:
            matches = self.pattern_match(pattern)
            premise_matches.append(matches)
        
        # Generate combinations of premise matches
        if premise_matches:
            # Simple case: single premise pattern
            if len(premise_matches) == 1:
                for match in premise_matches[0]:
                    # Generate conclusion based on pattern
                    conclusion = self._generate_conclusion(rule.conclusion_pattern, match)
                    if conclusion:
                        new_items.append(conclusion)
        
        return new_items
    
    def _generate_conclusion(self, pattern: Pattern, premise_match: Dict[str, Any]) -> Optional[Union[Atom, Link]]:
        """Generate conclusion from pattern and premise match"""
        # Simple conclusion generation - can be extended
        if pattern.pattern_type.endswith("Link"):
            # Generate a new link
            premise_obj = premise_match["object"]
            if isinstance(premise_obj, Link):
                # Create new link based on pattern
                new_link = Link(
                    link_type=pattern.pattern_type,
                    outgoing=premise_obj.outgoing,
                    truth_value=TruthValue(
                        strength=premise_obj.truth_value.strength * 0.9,
                        confidence=premise_obj.truth_value.confidence * 0.8
                    )
                )
                return new_link
        
        return None
    
    def forward_chain(self, max_iterations: int = 10) -> List[Union[Atom, Link]]:
        """Perform forward chaining inference"""
        new_items = []
        
        for iteration in range(max_iterations):
            iteration_items = []
            
            # Apply all rules
            for rule in self.rules.values():
                rule_items = self.apply_rule(rule)
                iteration_items.extend(rule_items)
            
            # Add new items to atom space
            for item in iteration_items:
                if isinstance(item, Atom):
                    if item.name not in self.atoms:
                        self.add_atom(item)
                        new_items.append(item)
                elif isinstance(item, Link):
                    link_id = f"{item.link_type}_{hash(item)}"
                    if link_id not in self.links:
                        self.add_link(item)
                        new_items.append(item)
            
            # Stop if no new items generated
            if not iteration_items:
                break
        
        logger.info(f"Forward chaining generated {len(new_items)} new items")
        return new_items
    
    def backward_chain(self, goal: Union[Atom, Link]) -> List[Dict[str, Any]]:
        """Perform backward chaining inference"""
        proof_trees = []
        
        # Simple backward chaining - find rules that can prove the goal
        for rule in self.rules.values():
            if self._can_prove_goal(rule.conclusion_pattern, goal):
                # Try to prove premises
                premise_proofs = []
                for pattern in rule.premise_patterns:
                    matches = self.pattern_match(pattern)
                    premise_proofs.extend(matches)
                
                if premise_proofs:
                    proof_trees.append({
                        "rule": rule,
                        "goal": goal,
                        "premises": premise_proofs
                    })
        
        return proof_trees
    
    def _can_prove_goal(self, pattern: Pattern, goal: Union[Atom, Link]) -> bool:
        """Check if pattern can prove goal"""
        return pattern.matches(goal)
    
    def calculate_attention(self, atom_name: str) -> float:
        """Calculate attention value for an atom"""
        if atom_name not in self.atoms:
            return 0.0
        
        atom = self.atoms[atom_name]
        
        # Base attention from truth value
        base_attention = atom.truth_value.strength * atom.truth_value.confidence
        
        # Boost from links
        link_boost = 0.0
        for link in self.links.values():
            if any(a.name == atom_name for a in link.outgoing):
                link_boost += 0.1
        
        # Boost from recent access
        time_boost = max(0.0, 1.0 - (time.time() - atom.creation_time) / 86400)  # 1 day decay
        
        total_attention = base_attention + min(link_boost, 0.5) + time_boost * 0.2
        
        self.attention_values[atom_name] = min(1.0, total_attention)
        return self.attention_values[atom_name]
    
    def get_high_attention_atoms(self, threshold: float = 0.7, max_results: int = 10) -> List[Atom]:
        """Get atoms with high attention values"""
        high_attention = []
        
        for atom_name, atom in self.atoms.items():
            attention = self.calculate_attention(atom_name)
            if attention >= threshold:
                high_attention.append((atom, attention))
        
        # Sort by attention and return top atoms
        high_attention.sort(key=lambda x: x[1], reverse=True)
        return [atom for atom, _ in high_attention[:max_results]]
    
    def export_knowledge_fragment(self, max_atoms: int = 50, max_links: int = 25) -> Dict[str, Any]:
        """Export knowledge fragment for sharing with other agents"""
        # Get high attention atoms
        high_attention_atoms = self.get_high_attention_atoms(threshold=0.5, max_results=max_atoms)
        
        # Get related links
        related_links = []
        atom_names = {atom.name for atom in high_attention_atoms}
        
        for link in self.links.values():
            if any(atom.name in atom_names for atom in link.outgoing):
                related_links.append(link)
                if len(related_links) >= max_links:
                    break
        
        fragment = {
            "agent_id": self.agent_id,
            "atoms": [atom.to_dict() for atom in high_attention_atoms],
            "links": [link.to_dict() for link in related_links],
            "patterns": {name: pattern.to_dict() for name, pattern in self.patterns.items()},
            "rules": {name: rule.to_dict() for name, rule in self.rules.items()},
            "export_time": time.time()
        }
        
        return fragment
    
    def import_knowledge_fragment(self, fragment: Dict[str, Any]) -> bool:
        """Import knowledge fragment from another agent"""
        try:
            source_agent = fragment.get("agent_id", "unknown")
            
            # Import atoms
            for atom_data in fragment.get("atoms", []):
                atom = Atom.from_dict(atom_data)
                # Prefix with source agent to avoid conflicts
                atom.name = f"{source_agent}_{atom.name}"
                self.add_atom(atom)
            
            # Import links
            for link_data in fragment.get("links", []):
                link = Link.from_dict(link_data)
                self.add_link(link)
            
            # Import patterns
            for pattern_name, pattern_data in fragment.get("patterns", {}).items():
                pattern = Pattern(**pattern_data)
                self.patterns[f"{source_agent}_{pattern_name}"] = pattern
            
            # Import rules
            for rule_name, rule_data in fragment.get("rules", {}).items():
                # Reconstruct rule (simplified)
                rule = Rule(
                    name=rule_data["name"],
                    premise_patterns=[],  # Simplified
                    conclusion_pattern=Pattern(**rule_data["conclusion_pattern"]),
                    strength=rule_data["strength"],
                    confidence=rule_data["confidence"]
                )
                self.rules[f"{source_agent}_{rule_name}"] = rule
            
            logger.info(f"Imported knowledge fragment from {source_agent}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing knowledge fragment: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the atom space"""
        return {
            "atom_count": len(self.atoms),
            "link_count": len(self.links),
            "pattern_count": len(self.patterns),
            "rule_count": len(self.rules),
            "avg_attention": sum(self.attention_values.values()) / len(self.attention_values) if self.attention_values else 0,
            "high_attention_atoms": len(self.get_high_attention_atoms()),
            "agent_id": self.agent_id
        }

# Example usage and testing
if __name__ == "__main__":
    # Create symbolic atom space
    atom_space = SymbolicAtomSpace("test_agent")
    
    # Add some atoms
    cat = Atom("cat", "ConceptNode", TruthValue(0.9, 0.8))
    animal = Atom("animal", "ConceptNode", TruthValue(0.95, 0.9))
    mammal = Atom("mammal", "ConceptNode", TruthValue(0.85, 0.85))
    
    atom_space.add_atom(cat)
    atom_space.add_atom(animal)
    atom_space.add_atom(mammal)
    
    # Add some links
    cat_animal = Link("InheritanceLink", [cat, animal], TruthValue(0.9, 0.8))
    mammal_animal = Link("InheritanceLink", [mammal, animal], TruthValue(0.95, 0.9))
    cat_mammal = Link("InheritanceLink", [cat, mammal], TruthValue(0.8, 0.7))
    
    atom_space.add_link(cat_animal)
    atom_space.add_link(mammal_animal)
    atom_space.add_link(cat_mammal)
    
    # Perform inference
    new_items = atom_space.forward_chain(max_iterations=5)
    print(f"Generated {len(new_items)} new items through inference")
    
    # Pattern matching
    inheritance_pattern = atom_space.patterns["inheritance"]
    matches = atom_space.pattern_match(inheritance_pattern)
    print(f"Found {len(matches)} inheritance matches")
    
    # Statistics
    stats = atom_space.get_statistics()
    print("Atom space statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Export knowledge fragment
    fragment = atom_space.export_knowledge_fragment()
    print(f"Exported knowledge fragment with {len(fragment['atoms'])} atoms and {len(fragment['links'])} links")