#!/usr/bin/env python3
"""
Distributed Cognitive Grammar Demo

This script demonstrates the distributed network of agentic cognitive grammar
for OpenCoq/echo9ml. It creates a simple network of cognitive agents that
share knowledge and perform collaborative reasoning.

Usage:
    python demo_distributed_cognitive_grammar.py
"""

import asyncio
import logging
import json
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock numpy for demo
import sys
sys.path.insert(0, '/tmp')
import numpy_mock as np

# Import our modules
from distributed_cognitive_grammar import (
    DistributedCognitiveNetwork, Echo9MLNode, CognitiveMessage, MessageType
)
from ggml_tensor_kernel import GGMLTensorKernel, TensorOperationType
from symbolic_reasoning import SymbolicAtomSpace, Atom, Link, TruthValue

class DemoAgent(Echo9MLNode):
    """Demo agent with enhanced cognitive capabilities"""
    
    def __init__(self, agent_id: str, broker=None, specialization: str = "general"):
        super().__init__(agent_id, broker)
        self.specialization = specialization
        self.tensor_kernel = GGMLTensorKernel(agent_id)
        self.atom_space = SymbolicAtomSpace(agent_id)
        
        # Initialize based on specialization
        self._initialize_specialization()
        
        logger.info(f"Created demo agent {agent_id} with specialization: {specialization}")
    
    def _initialize_specialization(self):
        """Initialize agent based on specialization"""
        if self.specialization == "creative":
            self._initialize_creative_knowledge()
        elif self.specialization == "logical":
            self._initialize_logical_knowledge()
        elif self.specialization == "memory":
            self._initialize_memory_knowledge()
        else:
            self._initialize_general_knowledge()
    
    def _initialize_creative_knowledge(self):
        """Initialize creative knowledge"""
        concepts = [
            ("creativity", "ConceptNode", TruthValue(0.95, 0.9)),
            ("imagination", "ConceptNode", TruthValue(0.9, 0.85)),
            ("innovation", "ConceptNode", TruthValue(0.88, 0.82)),
            ("art", "ConceptNode", TruthValue(0.92, 0.88))
        ]
        
        for name, atom_type, truth_value in concepts:
            atom = Atom(name, atom_type, truth_value)
            self.atom_space.add_atom(atom)
        
        # Add relationships
        creativity = self.atom_space.get_atom("creativity")
        imagination = self.atom_space.get_atom("imagination")
        innovation = self.atom_space.get_atom("innovation")
        art = self.atom_space.get_atom("art")
        
        if all([creativity, imagination, innovation, art]):
            links = [
                Link("SimilarityLink", [creativity, imagination], TruthValue(0.9, 0.8)),
                Link("SimilarityLink", [creativity, innovation], TruthValue(0.85, 0.8)),
                Link("SimilarityLink", [imagination, art], TruthValue(0.8, 0.75)),
                Link("InheritanceLink", [art, creativity], TruthValue(0.8, 0.7))
            ]
            
            for link in links:
                self.atom_space.add_link(link)
    
    def _initialize_logical_knowledge(self):
        """Initialize logical knowledge"""
        concepts = [
            ("logic", "ConceptNode", TruthValue(0.95, 0.9)),
            ("reasoning", "ConceptNode", TruthValue(0.92, 0.88)),
            ("analysis", "ConceptNode", TruthValue(0.88, 0.85)),
            ("mathematics", "ConceptNode", TruthValue(0.9, 0.87))
        ]
        
        for name, atom_type, truth_value in concepts:
            atom = Atom(name, atom_type, truth_value)
            self.atom_space.add_atom(atom)
        
        # Add relationships
        logic = self.atom_space.get_atom("logic")
        reasoning = self.atom_space.get_atom("reasoning")
        analysis = self.atom_space.get_atom("analysis")
        mathematics = self.atom_space.get_atom("mathematics")
        
        if all([logic, reasoning, analysis, mathematics]):
            links = [
                Link("InheritanceLink", [reasoning, logic], TruthValue(0.9, 0.85)),
                Link("InheritanceLink", [analysis, reasoning], TruthValue(0.8, 0.75)),
                Link("SimilarityLink", [logic, mathematics], TruthValue(0.85, 0.8)),
                Link("SimilarityLink", [reasoning, analysis], TruthValue(0.9, 0.85))
            ]
            
            for link in links:
                self.atom_space.add_link(link)
    
    def _initialize_memory_knowledge(self):
        """Initialize memory knowledge"""
        concepts = [
            ("memory", "ConceptNode", TruthValue(0.95, 0.9)),
            ("learning", "ConceptNode", TruthValue(0.92, 0.88)),
            ("experience", "ConceptNode", TruthValue(0.88, 0.85)),
            ("knowledge", "ConceptNode", TruthValue(0.9, 0.87))
        ]
        
        for name, atom_type, truth_value in concepts:
            atom = Atom(name, atom_type, truth_value)
            self.atom_space.add_atom(atom)
        
        # Add relationships
        memory = self.atom_space.get_atom("memory")
        learning = self.atom_space.get_atom("learning")
        experience = self.atom_space.get_atom("experience")
        knowledge = self.atom_space.get_atom("knowledge")
        
        if all([memory, learning, experience, knowledge]):
            links = [
                Link("InheritanceLink", [learning, memory], TruthValue(0.9, 0.85)),
                Link("InheritanceLink", [experience, memory], TruthValue(0.85, 0.8)),
                Link("SimilarityLink", [learning, knowledge], TruthValue(0.88, 0.82)),
                Link("SimilarityLink", [experience, knowledge], TruthValue(0.8, 0.75))
            ]
            
            for link in links:
                self.atom_space.add_link(link)
    
    def _initialize_general_knowledge(self):
        """Initialize general knowledge"""
        concepts = [
            ("intelligence", "ConceptNode", TruthValue(0.9, 0.85)),
            ("cognition", "ConceptNode", TruthValue(0.88, 0.82)),
            ("understanding", "ConceptNode", TruthValue(0.85, 0.8)),
            ("awareness", "ConceptNode", TruthValue(0.82, 0.78))
        ]
        
        for name, atom_type, truth_value in concepts:
            atom = Atom(name, atom_type, truth_value)
            self.atom_space.add_atom(atom)
    
    async def _process_cognitive_state(self):
        """Enhanced cognitive processing with specialization"""
        # Perform tensor evolution
        if "persona_state" in self.tensor_kernel.tensors:
            self.tensor_kernel.execute_operation(
                TensorOperationType.PERSONA_EVOLVE,
                ["persona_state"],
                "persona_evolved",
                learning_rate=0.02
            )
        
        # Perform symbolic reasoning
        new_items = self.atom_space.forward_chain(max_iterations=2)
        if new_items:
            logger.info(f"Agent {self.agent_id} generated {len(new_items)} new knowledge items")
        
        # Share knowledge based on specialization
        if len(self.atom_space.atoms) > 0:
            await self._share_specialized_knowledge()
    
    async def _share_specialized_knowledge(self):
        """Share specialized knowledge with other agents"""
        # Get high-attention atoms related to specialization
        high_attention = self.atom_space.get_high_attention_atoms(threshold=0.7, max_results=5)
        
        if high_attention:
            # Create knowledge fragment
            from distributed_cognitive_grammar import HypergraphFragment
            
            fragment = HypergraphFragment(
                id=f"{self.agent_id}_knowledge_{int(asyncio.get_event_loop().time())}",
                nodes=[
                    {
                        "id": atom.name,
                        "content": f"{self.specialization}_{atom.name}",
                        "salience": atom.truth_value.strength,
                        "specialization": self.specialization
                    }
                    for atom in high_attention
                ],
                edges=[],
                source_agent=self.agent_id,
                semantic_weight=0.8,
                metadata={"specialization": self.specialization}
            )
            
            await self.broadcast_hypergraph_fragment(fragment)
    
    def get_demo_statistics(self) -> Dict:
        """Get demo statistics"""
        return {
            "agent_id": self.agent_id,
            "specialization": self.specialization,
            "atoms": len(self.atom_space.atoms),
            "links": len(self.atom_space.links),
            "tensors": len(self.tensor_kernel.tensors),
            "peers": len(self.peers),
            "high_attention_atoms": len(self.atom_space.get_high_attention_atoms())
        }

async def run_demo():
    """Run the distributed cognitive grammar demo"""
    logger.info("🚀 Starting Distributed Cognitive Grammar Demo")
    logger.info("=" * 60)
    
    # Create network
    network = DistributedCognitiveNetwork()
    
    # Create specialized agents
    agents = [
        DemoAgent("creative_agent", network.broker, "creative"),
        DemoAgent("logical_agent", network.broker, "logical"),
        DemoAgent("memory_agent", network.broker, "memory"),
        DemoAgent("general_agent", network.broker, "general")
    ]
    
    # Add agents to network
    for agent in agents:
        network.add_agent(agent)
        
        # Initialize tensors
        agent.tensor_kernel.create_tensor("persona_state", "persona", "cognitive_traits")
        agent.tensor_kernel.create_tensor("attention_state", "attention", "attention_allocation")
    
    logger.info(f"Created {len(agents)} specialized agents")
    
    # Show initial state
    logger.info("\n📊 Initial Agent States:")
    for agent in agents:
        stats = agent.get_demo_statistics()
        logger.info(f"  {stats['agent_id']}: {stats['atoms']} atoms, {stats['links']} links, {stats['tensors']} tensors")
    
    # Run network for demonstration
    logger.info("\n🔄 Starting network processing...")
    
    async def demo_task():
        try:
            # Start the network
            network_task = asyncio.create_task(network.start_network())
            
            # Let agents process and share knowledge
            await asyncio.sleep(5)
            
            # Stop network
            await network.stop_network()
            network_task.cancel()
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
    
    await demo_task()
    
    # Show final state
    logger.info("\n📈 Final Agent States:")
    for agent in agents:
        stats = agent.get_demo_statistics()
        logger.info(f"  {stats['agent_id']}: {stats['atoms']} atoms, {stats['links']} links, {stats['peers']} peers")
    
    # Demonstrate knowledge sharing
    logger.info("\n🔍 Knowledge Sharing Analysis:")
    
    for agent in agents:
        # Show concepts learned from other agents
        foreign_concepts = [
            atom.name for atom in agent.atom_space.atoms.values()
            if atom.name.startswith(tuple(a.agent_id for a in agents if a != agent))
        ]
        
        if foreign_concepts:
            logger.info(f"  {agent.agent_id} learned: {', '.join(foreign_concepts[:3])}...")
        else:
            logger.info(f"  {agent.agent_id} maintained original knowledge")
    
    # Demonstrate tensor operations
    logger.info("\n🧮 Tensor Operations Demo:")
    
    demo_agent = agents[0]
    
    # Execute various tensor operations
    operations = [
        (TensorOperationType.PERSONA_EVOLVE, ["persona_state"], "persona_demo", {"learning_rate": 0.1}),
        (TensorOperationType.ATTENTION_SPREAD, ["attention_state"], "attention_demo", {"decay_factor": 0.7}),
    ]
    
    for op_type, inputs, output, kwargs in operations:
        success = demo_agent.tensor_kernel.execute_operation(op_type, inputs, output, **kwargs)
        logger.info(f"  {op_type.value}: {'✅ Success' if success else '❌ Failed'}")
    
    # Demonstrate symbolic reasoning
    logger.info("\n🤔 Symbolic Reasoning Demo:")
    
    reasoning_agent = agents[1]  # Use logical agent
    
    # Perform reasoning
    new_items = reasoning_agent.atom_space.forward_chain(max_iterations=3)
    logger.info(f"  Generated {len(new_items)} new knowledge items through inference")
    
    # Pattern matching
    patterns = ["logic", "creativity", "memory", "intelligence"]
    
    for pattern in patterns:
        matches = reasoning_agent.atom_space.search_atoms(pattern)
        logger.info(f"  Pattern '{pattern}': {len(matches)} matches")
    
    # Network statistics
    logger.info("\n🌐 Network Statistics:")
    logger.info(f"  Active agents: {len([a for a in agents if a.running])}")
    logger.info(f"  Total atoms: {sum(len(a.atom_space.atoms) for a in agents)}")
    logger.info(f"  Total links: {sum(len(a.atom_space.links) for a in agents)}")
    logger.info(f"  Total tensors: {sum(len(a.tensor_kernel.tensors) for a in agents)}")
    
    logger.info("\n✅ Demo completed successfully!")
    logger.info("=" * 60)

def main():
    """Main demo function"""
    print("🧠 Distributed Network of Agentic Cognitive Grammar Demo")
    print("OpenCoq/echo9ml Implementation")
    print()
    
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()