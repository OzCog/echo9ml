"""
Test suite for the distributed agentic cognitive grammar implementation.

This test demonstrates the integration of:
- Distributed cognitive agents
- Hypergraph knowledge representation
- GGML tensor operations
- Symbolic reasoning
- Attention allocation
- Inter-agent communication
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any

# Simple mock for testing when dependencies are missing
import sys

# Import our modules
from distributed_cognitive_grammar import (
    DistributedCognitiveNetwork, Echo9MLNode, AgentType,
    HypergraphFragment, CognitiveMessage, MessageType
)
from ggml_tensor_kernel import GGMLTensorKernel, TensorOperationType
from symbolic_reasoning import SymbolicAtomSpace, Atom, Link, TruthValue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedEcho9MLNode(Echo9MLNode):
    """Enhanced Echo9ML node with tensor kernel and symbolic reasoning"""
    
    def __init__(self, agent_id: str, broker=None):
        super().__init__(agent_id, broker)
        
        # Initialize tensor kernel
        self.tensor_kernel = GGMLTensorKernel(agent_id)
        
        # Initialize symbolic atom space
        self.atom_space = SymbolicAtomSpace(agent_id)
        
        # Create initial tensors
        self._initialize_cognitive_tensors()
        
        # Create initial knowledge
        self._initialize_knowledge()
        
        logger.info(f"Enhanced Echo9ML node {agent_id} initialized")
    
    def _initialize_cognitive_tensors(self):
        """Initialize cognitive tensors for the agent"""
        # Create persona tensor
        persona_tensor = self.tensor_kernel.create_tensor(
            "persona_state", "persona", "cognitive_traits", semantic_weight=0.9
        )
        
        # Create attention tensor
        attention_tensor = self.tensor_kernel.create_tensor(
            "attention_state", "attention", "attention_allocation", semantic_weight=0.8
        )
        
        # Create memory tensor
        memory_tensor = self.tensor_kernel.create_tensor(
            "memory_state", "memory", "memory_consolidation", semantic_weight=0.7
        )
        
        logger.info(f"Created {len(self.tensor_kernel.tensors)} cognitive tensors")
    
    def _initialize_knowledge(self):
        """Initialize symbolic knowledge for the agent"""
        # Create basic concepts
        self_concept = Atom(f"{self.agent_id}_self", "ConceptNode", TruthValue(0.9, 0.9))
        cognitive_concept = Atom("cognitive_process", "ConceptNode", TruthValue(0.8, 0.8))
        learning_concept = Atom("learning", "ConceptNode", TruthValue(0.85, 0.85))
        
        self.atom_space.add_atom(self_concept)
        self.atom_space.add_atom(cognitive_concept)
        self.atom_space.add_atom(learning_concept)
        
        # Create relationships
        self_cognitive_link = Link(
            "InheritanceLink", [self_concept, cognitive_concept], TruthValue(0.9, 0.8)
        )
        cognitive_learning_link = Link(
            "SimilarityLink", [cognitive_concept, learning_concept], TruthValue(0.7, 0.7)
        )
        
        self.atom_space.add_link(self_cognitive_link)
        self.atom_space.add_link(cognitive_learning_link)
        
        logger.info(f"Initialized {len(self.atom_space.atoms)} atoms and {len(self.atom_space.links)} links")
    
    async def _process_cognitive_state(self):
        """Enhanced cognitive state processing"""
        # Process tensor operations
        await self._process_tensor_operations()
        
        # Process symbolic reasoning
        await self._process_symbolic_reasoning()
        
        # Share knowledge periodically
        await self._share_knowledge()
    
    async def _process_tensor_operations(self):
        """Process tensor operations for cognitive evolution"""
        try:
            # Evolve persona tensor
            success = self.tensor_kernel.execute_operation(
                TensorOperationType.PERSONA_EVOLVE,
                ["persona_state"],
                "persona_evolved",
                learning_rate=0.05
            )
            
            if success:
                # Spread attention
                self.tensor_kernel.execute_operation(
                    TensorOperationType.ATTENTION_SPREAD,
                    ["attention_state"],
                    "attention_spread",
                    decay_factor=0.8
                )
                
                # Consolidate memory
                self.tensor_kernel.execute_operation(
                    TensorOperationType.MEMORY_CONSOLIDATE,
                    ["memory_state"],
                    "memory_consolidated",
                    consolidation_threshold=0.6
                )
                
                logger.debug(f"Processed tensor operations for {self.agent_id}")
        
        except Exception as e:
            logger.error(f"Error in tensor operations: {e}")
    
    async def _process_symbolic_reasoning(self):
        """Process symbolic reasoning"""
        try:
            # Perform forward chaining
            new_items = self.atom_space.forward_chain(max_iterations=3)
            
            if new_items:
                logger.info(f"Generated {len(new_items)} new knowledge items")
            
            # Update attention values
            high_attention_atoms = self.atom_space.get_high_attention_atoms(threshold=0.6)
            logger.debug(f"Found {len(high_attention_atoms)} high attention atoms")
        
        except Exception as e:
            logger.error(f"Error in symbolic reasoning: {e}")
    
    async def _share_knowledge(self):
        """Share knowledge with other agents"""
        try:
            # Share symbolic knowledge
            knowledge_fragment = self.atom_space.export_knowledge_fragment(max_atoms=10, max_links=5)
            
            # Create hypergraph fragment
            hypergraph_fragment = HypergraphFragment(
                id=f"{self.agent_id}_knowledge_{int(time.time())}",
                nodes=[
                    {
                        "id": atom_data["name"],
                        "content": atom_data["name"],
                        "salience": atom_data["truth_value"]["strength"]
                    }
                    for atom_data in knowledge_fragment["atoms"]
                ],
                edges=[
                    {
                        "from": link_data["outgoing"][0]["name"],
                        "to": link_data["outgoing"][1]["name"],
                        "type": link_data["link_type"],
                        "weight": link_data["truth_value"]["strength"]
                    }
                    for link_data in knowledge_fragment["links"]
                    if len(link_data["outgoing"]) >= 2
                ],
                source_agent=self.agent_id,
                semantic_weight=0.8
            )
            
            await self.broadcast_hypergraph_fragment(hypergraph_fragment)
            
            # Share tensor catalog
            tensor_catalog = self.tensor_kernel.export_tensor_catalog()
            
            tensor_message = CognitiveMessage(
                message_id=f"{self.agent_id}_tensor_{int(time.time())}",
                message_type=MessageType.TENSOR_UPDATE,
                sender_id=self.agent_id,
                payload={"tensor_catalog": tensor_catalog}
            )
            
            await self._send_message(tensor_message)
            
        except Exception as e:
            logger.error(f"Error sharing knowledge: {e}")
    
    async def _handle_hypergraph_fragment(self, message: CognitiveMessage):
        """Enhanced hypergraph fragment handling"""
        await super()._handle_hypergraph_fragment(message)
        
        # Also integrate into symbolic reasoning
        fragment_data = message.payload.get("fragment", {})
        
        # Convert nodes to atoms
        for node_data in fragment_data.get("nodes", []):
            if "id" in node_data:
                atom = Atom(
                    name=node_data["id"],
                    atom_type="ConceptNode",
                    truth_value=TruthValue(
                        strength=node_data.get("salience", 0.5),
                        confidence=0.7
                    )
                )
                self.atom_space.add_atom(atom)
        
        # Convert edges to links
        for edge_data in fragment_data.get("edges", []):
            if "from" in edge_data and "to" in edge_data:
                from_atom = self.atom_space.get_atom(edge_data["from"])
                to_atom = self.atom_space.get_atom(edge_data["to"])
                
                if from_atom and to_atom:
                    link = Link(
                        link_type=edge_data.get("type", "SimilarityLink"),
                        outgoing=[from_atom, to_atom],
                        truth_value=TruthValue(
                            strength=edge_data.get("weight", 0.5),
                            confidence=0.7
                        )
                    )
                    self.atom_space.add_link(link)
    
    async def _handle_tensor_update(self, message: CognitiveMessage):
        """Enhanced tensor update handling"""
        await super()._handle_tensor_update(message)
        
        # Import tensor catalog
        tensor_catalog = message.payload.get("tensor_catalog", {})
        if tensor_catalog:
            success = self.tensor_kernel.import_tensor_catalog(tensor_catalog)
            if success:
                logger.info(f"Imported tensor catalog from {message.sender_id}")
    
    def get_cognitive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cognitive statistics"""
        tensor_stats = {
            "tensor_count": len(self.tensor_kernel.tensors),
            "tensor_shapes": len(self.tensor_kernel.tensor_shapes),
            "tensor_operations": len(self.tensor_kernel.custom_operations)
        }
        
        atom_space_stats = self.atom_space.get_statistics()
        
        return {
            "agent_id": self.agent_id,
            "tensor_kernel": tensor_stats,
            "symbolic_reasoning": atom_space_stats,
            "peers": len(self.peers),
            "running": self.running
        }

async def test_distributed_cognitive_grammar():
    """Test the distributed cognitive grammar system"""
    logger.info("Starting distributed cognitive grammar test...")
    
    # Create network
    network = DistributedCognitiveNetwork()
    
    # Create enhanced agents
    agent1 = EnhancedEcho9MLNode("echo_agent_1", network.broker)
    agent2 = EnhancedEcho9MLNode("echo_agent_2", network.broker)
    agent3 = EnhancedEcho9MLNode("echo_agent_3", network.broker)
    
    # Add agents to network
    network.add_agent(agent1)
    network.add_agent(agent2)
    network.add_agent(agent3)
    
    # Add some initial knowledge to agent1
    creativity_concept = Atom("creativity", "ConceptNode", TruthValue(0.9, 0.8))
    innovation_concept = Atom("innovation", "ConceptNode", TruthValue(0.85, 0.85))
    agent1.atom_space.add_atom(creativity_concept)
    agent1.atom_space.add_atom(innovation_concept)
    
    creativity_innovation_link = Link(
        "SimilarityLink", [creativity_concept, innovation_concept], TruthValue(0.8, 0.7)
    )
    agent1.atom_space.add_link(creativity_innovation_link)
    
    # Add different knowledge to agent2
    reasoning_concept = Atom("reasoning", "ConceptNode", TruthValue(0.9, 0.9))
    logic_concept = Atom("logic", "ConceptNode", TruthValue(0.85, 0.8))
    agent2.atom_space.add_atom(reasoning_concept)
    agent2.atom_space.add_atom(logic_concept)
    
    reasoning_logic_link = Link(
        "InheritanceLink", [reasoning_concept, logic_concept], TruthValue(0.9, 0.8)
    )
    agent2.atom_space.add_link(reasoning_logic_link)
    
    logger.info("Initialized agents with different knowledge bases")
    
    # Run network for a short time
    async def run_test():
        try:
            # Start network
            network_task = asyncio.create_task(network.start_network())
            
            # Let it run for a while
            await asyncio.sleep(10)
            
            # Stop network
            await network.stop_network()
            network_task.cancel()
            
        except Exception as e:
            logger.error(f"Error in test: {e}")
    
    await run_test()
    
    # Check results
    logger.info("Test results:")
    
    for agent in [agent1, agent2, agent3]:
        stats = agent.get_cognitive_statistics()
        logger.info(f"Agent {agent.agent_id}:")
        logger.info(f"  Atoms: {stats['symbolic_reasoning']['atom_count']}")
        logger.info(f"  Links: {stats['symbolic_reasoning']['link_count']}")
        logger.info(f"  Tensors: {stats['tensor_kernel']['tensor_count']}")
        logger.info(f"  Peers: {stats['peers']}")
    
    # Test tensor operations
    logger.info("\nTesting tensor operations...")
    test_agent = agent1
    
    # Execute various tensor operations
    operations = [
        (TensorOperationType.PERSONA_EVOLVE, ["persona_state"], "persona_test", {"learning_rate": 0.1}),
        (TensorOperationType.ATTENTION_SPREAD, ["attention_state"], "attention_test", {"decay_factor": 0.7}),
        (TensorOperationType.MEMORY_CONSOLIDATE, ["memory_state"], "memory_test", {"consolidation_threshold": 0.5}),
    ]
    
    for op_type, inputs, output, kwargs in operations:
        success = test_agent.tensor_kernel.execute_operation(op_type, inputs, output, **kwargs)
        logger.info(f"  {op_type.value}: {'Success' if success else 'Failed'}")
    
    # Test symbolic reasoning
    logger.info("\nTesting symbolic reasoning...")
    new_items = test_agent.atom_space.forward_chain(max_iterations=5)
    logger.info(f"  Generated {len(new_items)} new items through inference")
    
    # Test pattern matching
    patterns = ["creativity", "reasoning", "logic"]
    for pattern in patterns:
        matches = test_agent.atom_space.search_atoms(pattern)
        logger.info(f"  Pattern '{pattern}': {len(matches)} matches")
    
    # Test knowledge export/import
    logger.info("\nTesting knowledge sharing...")
    fragment = test_agent.atom_space.export_knowledge_fragment()
    logger.info(f"  Exported fragment: {len(fragment['atoms'])} atoms, {len(fragment['links'])} links")
    
    # Import to another agent
    import_success = agent2.atom_space.import_knowledge_fragment(fragment)
    logger.info(f"  Import success: {import_success}")
    
    if import_success:
        final_stats = agent2.atom_space.get_statistics()
        logger.info(f"  Agent 2 final atom count: {final_stats['atom_count']}")
    
    logger.info("Distributed cognitive grammar test completed successfully!")

def test_tensor_kernel():
    """Test the GGML tensor kernel"""
    logger.info("Testing GGML tensor kernel...")
    
    kernel = GGMLTensorKernel("test_agent")
    
    # Create tensors
    persona_tensor = kernel.create_tensor("persona_test", "persona", "cognitive_traits")
    attention_tensor = kernel.create_tensor("attention_test", "attention", "attention_allocation")
    
    # Test operations
    operations = [
        (TensorOperationType.PERSONA_EVOLVE, ["persona_test"], "persona_evolved", {"learning_rate": 0.1}),
        (TensorOperationType.ATTENTION_SPREAD, ["attention_test"], "attention_spread", {"decay_factor": 0.8}),
    ]
    
    for op_type, inputs, output, kwargs in operations:
        success = kernel.execute_operation(op_type, inputs, output, **kwargs)
        logger.info(f"  {op_type.value}: {'Success' if success else 'Failed'}")
    
    # Test catalog export/import
    catalog = kernel.export_tensor_catalog()
    logger.info(f"  Exported catalog with {len(catalog['tensors'])} tensors")
    
    # Create new kernel and import
    kernel2 = GGMLTensorKernel("test_agent_2")
    import_success = kernel2.import_tensor_catalog(catalog)
    logger.info(f"  Import success: {import_success}")
    
    if import_success:
        logger.info(f"  Imported {len(kernel2.tensors)} tensors")

def test_symbolic_reasoning():
    """Test the symbolic reasoning system"""
    logger.info("Testing symbolic reasoning...")
    
    atom_space = SymbolicAtomSpace("test_agent")
    
    # Add test knowledge
    concepts = [
        ("cat", "ConceptNode", TruthValue(0.9, 0.8)),
        ("animal", "ConceptNode", TruthValue(0.95, 0.9)),
        ("mammal", "ConceptNode", TruthValue(0.85, 0.85)),
        ("dog", "ConceptNode", TruthValue(0.9, 0.8)),
    ]
    
    for name, atom_type, truth_value in concepts:
        atom = Atom(name, atom_type, truth_value)
        atom_space.add_atom(atom)
    
    # Add relationships
    relationships = [
        ("InheritanceLink", ["cat", "mammal"], TruthValue(0.9, 0.8)),
        ("InheritanceLink", ["dog", "mammal"], TruthValue(0.9, 0.8)),
        ("InheritanceLink", ["mammal", "animal"], TruthValue(0.95, 0.9)),
        ("SimilarityLink", ["cat", "dog"], TruthValue(0.7, 0.6)),
    ]
    
    for link_type, atom_names, truth_value in relationships:
        atoms = [atom_space.get_atom(name) for name in atom_names]
        if all(atoms):
            link = Link(link_type, atoms, truth_value)
            atom_space.add_link(link)
    
    # Test inference
    new_items = atom_space.forward_chain(max_iterations=5)
    logger.info(f"  Generated {len(new_items)} new items through inference")
    
    # Test pattern matching
    patterns = ["cat", "animal", "mammal"]
    for pattern in patterns:
        matches = atom_space.search_atoms(pattern)
        logger.info(f"  Pattern '{pattern}': {len(matches)} matches")
    
    # Test knowledge export/import
    fragment = atom_space.export_knowledge_fragment()
    logger.info(f"  Exported fragment: {len(fragment['atoms'])} atoms, {len(fragment['links'])} links")
    
    # Test statistics
    stats = atom_space.get_statistics()
    logger.info(f"  Statistics: {stats}")

# Main test execution
if __name__ == "__main__":
    async def run_all_tests():
        logger.info("Running comprehensive distributed cognitive grammar tests...")
        
        # Test individual components
        test_tensor_kernel()
        test_symbolic_reasoning()
        
        # Test integrated system
        await test_distributed_cognitive_grammar()
        
        logger.info("All tests completed!")
    
    asyncio.run(run_all_tests())