"""
MOSES-Inspired Evolutionary Search for Cognitive Grammar Fragments

This module implements evolutionary optimization for discovering and optimizing
cognitive grammar patterns in the distributed system. Based on the MOSES
(Meta-Optimizing Semantic Evolutionary Search) approach.

Key Features:
- Genetic algorithm-like optimization of cognitive patterns
- Fitness evaluation based on semantic coherence and attention allocation
- Population-based search with selection, mutation, and crossover
- Integration with hypergraph fragments and tensor operations
"""

import random
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import copy

logger = logging.getLogger(__name__)

class MutationType(Enum):
    """Types of mutations for cognitive patterns"""
    WEIGHT_ADJUSTMENT = "weight_adjustment"
    STRUCTURE_MODIFICATION = "structure_modification"
    ATTENTION_REALLOCATION = "attention_reallocation"
    SEMANTIC_DRIFT = "semantic_drift"
    TENSOR_RESHAPE = "tensor_reshape"

class SelectionMethod(Enum):
    """Selection methods for evolutionary search"""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    ELITIST = "elitist"

@dataclass
class CognitivePattern:
    """Represents a cognitive pattern for evolutionary optimization"""
    pattern_id: str
    pattern_type: str  # "hypergraph", "tensor", "symbolic", "hybrid"
    genes: Dict[str, Any]  # Pattern parameters/weights
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = str(uuid.uuid4())
    
    def copy(self) -> 'CognitivePattern':
        """Create a deep copy of the pattern"""
        new_pattern = CognitivePattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type=self.pattern_type,
            genes=copy.deepcopy(self.genes),
            fitness=self.fitness,
            generation=self.generation + 1,
            parent_ids=[self.pattern_id],
            mutation_history=self.mutation_history.copy(),
            creation_time=time.time()
        )
        return new_pattern
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for serialization"""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "genes": self.genes,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "mutation_history": self.mutation_history,
            "creation_time": self.creation_time
        }

@dataclass
class EvolutionaryParameters:
    """Parameters for evolutionary search"""
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_rate: float = 0.1
    tournament_size: int = 3
    max_generations: int = 100
    fitness_threshold: float = 0.9
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    diversity_pressure: float = 0.1

class FitnessEvaluator:
    """Evaluates fitness of cognitive patterns"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.evaluation_history: Dict[str, float] = {}
    
    def evaluate_pattern(self, pattern: CognitivePattern, 
                        context: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate fitness of a cognitive pattern"""
        try:
            # Multi-criteria fitness evaluation
            fitness_components = {
                "semantic_coherence": self._evaluate_semantic_coherence(pattern),
                "attention_efficiency": self._evaluate_attention_efficiency(pattern),
                "structural_complexity": self._evaluate_structural_complexity(pattern),
                "contextual_relevance": self._evaluate_contextual_relevance(pattern, context),
                "novelty": self._evaluate_novelty(pattern)
            }
            
            # Weighted combination of fitness components
            weights = {
                "semantic_coherence": 0.3,
                "attention_efficiency": 0.25,
                "structural_complexity": 0.2,
                "contextual_relevance": 0.15,
                "novelty": 0.1
            }
            
            fitness = sum(
                fitness_components[component] * weights[component]
                for component in fitness_components
            )
            
            # Cache evaluation
            self.evaluation_history[pattern.pattern_id] = fitness
            
            return max(0.0, min(1.0, fitness))
        
        except Exception as e:
            logger.error(f"Error evaluating pattern {pattern.pattern_id}: {e}")
            return 0.0
    
    def _evaluate_semantic_coherence(self, pattern: CognitivePattern) -> float:
        """Evaluate semantic coherence of the pattern"""
        genes = pattern.genes
        
        if pattern.pattern_type == "hypergraph":
            # For hypergraph patterns, check node-edge consistency
            nodes = genes.get("nodes", [])
            edges = genes.get("edges", [])
            
            if not nodes:
                return 0.0
            
            # Calculate connectivity ratio
            max_edges = len(nodes) * (len(nodes) - 1) / 2
            connectivity = len(edges) / max_edges if max_edges > 0 else 0
            
            # Check semantic weight consistency
            weights = [node.get("semantic_weight", 0.5) for node in nodes]
            weight_variance = sum((w - 0.5) ** 2 for w in weights) / len(weights)
            coherence = 1.0 - weight_variance
            
            return (connectivity + coherence) / 2
        
        elif pattern.pattern_type == "tensor":
            # For tensor patterns, check dimensionality consistency
            shape = genes.get("shape", [])
            if not shape:
                return 0.0
            
            # Prefer prime factor decomposition
            prime_factors = self._get_prime_factors(shape)
            prime_ratio = len(prime_factors) / len(shape) if shape else 0
            
            # Check semantic mapping completeness
            semantic_mapping = genes.get("semantic_mapping", {})
            mapping_coverage = len(semantic_mapping) / len(shape) if shape else 0
            
            return (prime_ratio + mapping_coverage) / 2
        
        else:
            # Default coherence evaluation
            return random.uniform(0.3, 0.7)
    
    def _evaluate_attention_efficiency(self, pattern: CognitivePattern) -> float:
        """Evaluate attention allocation efficiency"""
        genes = pattern.genes
        
        # Check attention allocation patterns
        attention_weights = genes.get("attention_weights", [])
        if not attention_weights:
            return 0.5
        
        # Prefer balanced attention distribution
        total_attention = sum(attention_weights)
        if total_attention == 0:
            return 0.0
        
        normalized_weights = [w / total_attention for w in attention_weights]
        
        # Calculate entropy (higher entropy = more balanced distribution)
        import math
        entropy = -sum(w * math.log2(w + 1e-10) for w in normalized_weights if w > 0)
        max_entropy = math.log2(len(attention_weights)) if len(attention_weights) > 1 else 1
        
        return entropy / max_entropy if max_entropy > 0 else 0.5
    
    def _evaluate_structural_complexity(self, pattern: CognitivePattern) -> float:
        """Evaluate structural complexity (prefer moderate complexity)"""
        genes = pattern.genes
        
        if pattern.pattern_type == "hypergraph":
            nodes = genes.get("nodes", [])
            edges = genes.get("edges", [])
            
            # Moderate complexity is preferred
            node_count = len(nodes)
            edge_count = len(edges)
            
            # Optimal complexity range
            optimal_node_range = (5, 20)
            optimal_edge_range = (3, 30)
            
            node_fitness = self._gaussian_fitness(node_count, optimal_node_range)
            edge_fitness = self._gaussian_fitness(edge_count, optimal_edge_range)
            
            return (node_fitness + edge_fitness) / 2
        
        elif pattern.pattern_type == "tensor":
            shape = genes.get("shape", [])
            if not shape:
                return 0.0
            
            # Prefer moderate dimensionality
            dimension_count = len(shape)
            dimension_size = sum(shape)
            
            optimal_dim_range = (3, 7)
            optimal_size_range = (100, 10000)
            
            dim_fitness = self._gaussian_fitness(dimension_count, optimal_dim_range)
            size_fitness = self._gaussian_fitness(dimension_size, optimal_size_range)
            
            return (dim_fitness + size_fitness) / 2
        
        else:
            return 0.5
    
    def _evaluate_contextual_relevance(self, pattern: CognitivePattern, 
                                     context: Optional[Dict[str, Any]]) -> float:
        """Evaluate relevance to current context"""
        if not context:
            return 0.5
        
        genes = pattern.genes
        
        # Check context alignment
        context_keywords = context.get("keywords", [])
        pattern_keywords = genes.get("keywords", [])
        
        if not context_keywords or not pattern_keywords:
            return 0.5
        
        # Calculate keyword overlap
        overlap = set(context_keywords) & set(pattern_keywords)
        relevance = len(overlap) / max(len(context_keywords), len(pattern_keywords))
        
        return relevance
    
    def _evaluate_novelty(self, pattern: CognitivePattern) -> float:
        """Evaluate novelty compared to previous patterns"""
        if not self.evaluation_history:
            return 1.0  # First pattern is maximally novel
        
        # Simple novelty measure based on fitness distance
        similar_patterns = [
            fitness for fitness in self.evaluation_history.values()
            if abs(fitness - pattern.fitness) < 0.1
        ]
        
        novelty = 1.0 - len(similar_patterns) / len(self.evaluation_history)
        return max(0.0, novelty)
    
    def _gaussian_fitness(self, value: float, optimal_range: Tuple[float, float]) -> float:
        """Calculate fitness using Gaussian distribution around optimal range"""
        min_val, max_val = optimal_range
        optimal_val = (min_val + max_val) / 2
        sigma = (max_val - min_val) / 4  # 95% of values within range
        
        # Gaussian fitness function
        import math
        fitness = math.exp(-0.5 * ((value - optimal_val) / sigma) ** 2)
        return fitness
    
    def _get_prime_factors(self, shape: List[int]) -> List[int]:
        """Get prime factors from tensor shape dimensions"""
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n ** 0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        primes = [dim for dim in shape if is_prime(dim)]
        return primes

class MOSESEvolutionarySearch:
    """Main evolutionary search engine for cognitive patterns"""
    
    def __init__(self, agent_id: str, parameters: Optional[EvolutionaryParameters] = None):
        self.agent_id = agent_id
        self.parameters = parameters or EvolutionaryParameters()
        self.fitness_evaluator = FitnessEvaluator(agent_id)
        self.population: List[CognitivePattern] = []
        self.generation = 0
        self.best_patterns: List[CognitivePattern] = []
        self.evolution_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized MOSES evolutionary search for agent {agent_id}")
    
    def initialize_population(self, seed_patterns: Optional[List[CognitivePattern]] = None):
        """Initialize the population with random or seed patterns"""
        self.population = []
        
        if seed_patterns:
            self.population.extend(seed_patterns[:self.parameters.population_size])
        
        # Fill remaining population with random patterns
        while len(self.population) < self.parameters.population_size:
            pattern = self._create_random_pattern()
            self.population.append(pattern)
        
        # Evaluate initial population
        for pattern in self.population:
            pattern.fitness = self.fitness_evaluator.evaluate_pattern(pattern)
        
        logger.info(f"Initialized population with {len(self.population)} patterns")
    
    def evolve(self, generations: Optional[int] = None, 
               context: Optional[Dict[str, Any]] = None) -> List[CognitivePattern]:
        """Run evolutionary search for specified generations"""
        max_generations = generations or self.parameters.max_generations
        
        for gen in range(max_generations):
            self.generation = gen
            
            # Evaluate population
            for pattern in self.population:
                pattern.fitness = self.fitness_evaluator.evaluate_pattern(pattern, context)
            
            # Sort by fitness
            self.population.sort(key=lambda p: p.fitness, reverse=True)
            
            # Track best patterns
            if self.population:
                best_pattern = self.population[0]
                if not self.best_patterns or best_pattern.fitness > self.best_patterns[-1].fitness:
                    self.best_patterns.append(best_pattern.copy())
            
            # Record evolution statistics
            stats = self._calculate_generation_stats()
            self.evolution_history.append(stats)
            
            # Check termination criteria
            if self.population[0].fitness >= self.parameters.fitness_threshold:
                logger.info(f"Fitness threshold reached at generation {gen}")
                break
            
            # Create next generation
            new_population = self._create_next_generation()
            self.population = new_population
            
            logger.debug(f"Generation {gen}: best_fitness={self.population[0].fitness:.3f}")
        
        logger.info(f"Evolution completed after {self.generation + 1} generations")
        return self.best_patterns
    
    def _create_random_pattern(self) -> CognitivePattern:
        """Create a random cognitive pattern"""
        pattern_types = ["hypergraph", "tensor", "symbolic", "hybrid"]
        pattern_type = random.choice(pattern_types)
        
        if pattern_type == "hypergraph":
            genes = self._create_random_hypergraph_genes()
        elif pattern_type == "tensor":
            genes = self._create_random_tensor_genes()
        elif pattern_type == "symbolic":
            genes = self._create_random_symbolic_genes()
        else:  # hybrid
            genes = self._create_random_hybrid_genes()
        
        return CognitivePattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type=pattern_type,
            genes=genes,
            generation=self.generation
        )
    
    def _create_random_hypergraph_genes(self) -> Dict[str, Any]:
        """Create random hypergraph pattern genes"""
        node_count = random.randint(3, 15)
        edge_count = random.randint(1, node_count * 2)
        
        nodes = []
        for i in range(node_count):
            nodes.append({
                "id": f"node_{i}",
                "semantic_weight": random.uniform(0.1, 1.0),
                "keywords": [f"concept_{random.randint(1, 100)}"]
            })
        
        edges = []
        for i in range(edge_count):
            from_node = random.randint(0, node_count - 1)
            to_node = random.randint(0, node_count - 1)
            if from_node != to_node:
                edges.append({
                    "from": from_node,
                    "to": to_node,
                    "weight": random.uniform(0.1, 1.0),
                    "type": random.choice(["similarity", "inheritance", "causal"])
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "attention_weights": [random.uniform(0.1, 1.0) for _ in range(node_count)]
        }
    
    def _create_random_tensor_genes(self) -> Dict[str, Any]:
        """Create random tensor pattern genes"""
        # Use prime numbers for shape dimensions
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        dimension_count = random.randint(3, 6)
        shape = [random.choice(primes) for _ in range(dimension_count)]
        
        semantic_dimensions = ["persona", "trait", "time", "context", "valence", "attention"]
        semantic_mapping = {}
        for i, dim in enumerate(shape):
            if i < len(semantic_dimensions):
                semantic_mapping[semantic_dimensions[i]] = i
        
        return {
            "shape": shape,
            "semantic_mapping": semantic_mapping,
            "attention_weights": [random.uniform(0.1, 1.0) for _ in range(len(shape))],
            "keywords": [f"tensor_concept_{random.randint(1, 50)}"]
        }
    
    def _create_random_symbolic_genes(self) -> Dict[str, Any]:
        """Create random symbolic pattern genes"""
        rule_count = random.randint(2, 8)
        rules = []
        
        for i in range(rule_count):
            rules.append({
                "premise": f"concept_{random.randint(1, 20)}",
                "conclusion": f"concept_{random.randint(1, 20)}",
                "strength": random.uniform(0.5, 1.0),
                "confidence": random.uniform(0.5, 1.0)
            })
        
        return {
            "rules": rules,
            "attention_weights": [random.uniform(0.1, 1.0) for _ in range(rule_count)],
            "keywords": [f"symbolic_concept_{random.randint(1, 30)}"]
        }
    
    def _create_random_hybrid_genes(self) -> Dict[str, Any]:
        """Create random hybrid pattern genes"""
        hypergraph_genes = self._create_random_hypergraph_genes()
        tensor_genes = self._create_random_tensor_genes()
        
        return {
            "hypergraph": hypergraph_genes,
            "tensor": tensor_genes,
            "integration_weights": [random.uniform(0.1, 1.0) for _ in range(5)]
        }
    
    def _create_next_generation(self) -> List[CognitivePattern]:
        """Create the next generation through selection, crossover, and mutation"""
        new_population = []
        
        # Elitism: keep best patterns
        elite_count = int(self.parameters.population_size * self.parameters.elitism_rate)
        elites = self.population[:elite_count]
        new_population.extend([pattern.copy() for pattern in elites])
        
        # Fill remaining population through crossover and mutation
        while len(new_population) < self.parameters.population_size:
            if random.random() < self.parameters.crossover_rate:
                # Crossover
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child = self._crossover(parent1, parent2)
            else:
                # Reproduction
                parent = self._select_parent()
                child = parent.copy()
            
            # Mutation
            if random.random() < self.parameters.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population[:self.parameters.population_size]
    
    def _select_parent(self) -> CognitivePattern:
        """Select parent using specified selection method"""
        if self.parameters.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection()
        elif self.parameters.selection_method == SelectionMethod.ROULETTE_WHEEL:
            return self._roulette_wheel_selection()
        elif self.parameters.selection_method == SelectionMethod.RANK_BASED:
            return self._rank_based_selection()
        else:  # ELITIST
            return self._elitist_selection()
    
    def _tournament_selection(self) -> CognitivePattern:
        """Tournament selection"""
        tournament = random.sample(self.population, 
                                 min(self.parameters.tournament_size, len(self.population)))
        return max(tournament, key=lambda p: p.fitness)
    
    def _roulette_wheel_selection(self) -> CognitivePattern:
        """Roulette wheel selection"""
        total_fitness = sum(p.fitness for p in self.population)
        if total_fitness == 0:
            return random.choice(self.population)
        
        selection_point = random.uniform(0, total_fitness)
        current_sum = 0
        
        for pattern in self.population:
            current_sum += pattern.fitness
            if current_sum >= selection_point:
                return pattern
        
        return self.population[-1]  # Fallback
    
    def _rank_based_selection(self) -> CognitivePattern:
        """Rank-based selection"""
        # Population is already sorted by fitness
        ranks = list(range(len(self.population), 0, -1))
        total_rank = sum(ranks)
        
        selection_point = random.uniform(0, total_rank)
        current_sum = 0
        
        for i, rank in enumerate(ranks):
            current_sum += rank
            if current_sum >= selection_point:
                return self.population[i]
        
        return self.population[-1]  # Fallback
    
    def _elitist_selection(self) -> CognitivePattern:
        """Elitist selection (always select from top performers)"""
        elite_size = max(1, int(len(self.population) * 0.2))
        return random.choice(self.population[:elite_size])
    
    def _crossover(self, parent1: CognitivePattern, parent2: CognitivePattern) -> CognitivePattern:
        """Create offspring through crossover"""
        if parent1.pattern_type != parent2.pattern_type:
            # Different types: create hybrid
            child = CognitivePattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type="hybrid",
                genes={
                    "type1": parent1.genes,
                    "type2": parent2.genes,
                    "mixing_weights": [random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)]
                },
                generation=self.generation + 1,
                parent_ids=[parent1.pattern_id, parent2.pattern_id]
            )
        else:
            # Same type: blend genes
            child_genes = self._blend_genes(parent1.genes, parent2.genes)
            child = CognitivePattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type=parent1.pattern_type,
                genes=child_genes,
                generation=self.generation + 1,
                parent_ids=[parent1.pattern_id, parent2.pattern_id]
            )
        
        return child
    
    def _blend_genes(self, genes1: Dict[str, Any], genes2: Dict[str, Any]) -> Dict[str, Any]:
        """Blend genes from two parents"""
        blended_genes = {}
        
        all_keys = set(genes1.keys()) | set(genes2.keys())
        
        for key in all_keys:
            if key in genes1 and key in genes2:
                val1, val2 = genes1[key], genes2[key]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numerical blending
                    alpha = random.uniform(0.3, 0.7)
                    blended_genes[key] = alpha * val1 + (1 - alpha) * val2
                elif isinstance(val1, list) and isinstance(val2, list):
                    # List blending
                    blended_genes[key] = self._blend_lists(val1, val2)
                else:
                    # Random selection
                    blended_genes[key] = random.choice([val1, val2])
            else:
                # Take from available parent
                blended_genes[key] = genes1.get(key, genes2.get(key))
        
        return blended_genes
    
    def _blend_lists(self, list1: List[Any], list2: List[Any]) -> List[Any]:
        """Blend two lists"""
        max_len = max(len(list1), len(list2))
        blended_list = []
        
        for i in range(max_len):
            if i < len(list1) and i < len(list2):
                # Both have elements: blend or choose
                if isinstance(list1[i], (int, float)) and isinstance(list2[i], (int, float)):
                    alpha = random.uniform(0.3, 0.7)
                    blended_list.append(alpha * list1[i] + (1 - alpha) * list2[i])
                else:
                    blended_list.append(random.choice([list1[i], list2[i]]))
            elif i < len(list1):
                blended_list.append(list1[i])
            else:
                blended_list.append(list2[i])
        
        return blended_list
    
    def _mutate(self, pattern: CognitivePattern) -> CognitivePattern:
        """Apply mutation to a pattern"""
        mutation_type = random.choice(list(MutationType))
        pattern.mutation_history.append(mutation_type.value)
        
        if mutation_type == MutationType.WEIGHT_ADJUSTMENT:
            self._mutate_weights(pattern)
        elif mutation_type == MutationType.STRUCTURE_MODIFICATION:
            self._mutate_structure(pattern)
        elif mutation_type == MutationType.ATTENTION_REALLOCATION:
            self._mutate_attention(pattern)
        elif mutation_type == MutationType.SEMANTIC_DRIFT:
            self._mutate_semantics(pattern)
        elif mutation_type == MutationType.TENSOR_RESHAPE:
            self._mutate_tensor_shape(pattern)
        
        return pattern
    
    def _mutate_weights(self, pattern: CognitivePattern):
        """Mutate numerical weights in the pattern"""
        genes = pattern.genes
        
        for key, value in genes.items():
            if isinstance(value, (int, float)):
                mutation_strength = random.uniform(0.05, 0.2)
                direction = random.choice([-1, 1])
                genes[key] = max(0.0, min(1.0, value + direction * mutation_strength))
            elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                for i in range(len(value)):
                    if random.random() < 0.3:  # 30% chance to mutate each element
                        mutation_strength = random.uniform(0.05, 0.15)
                        direction = random.choice([-1, 1])
                        value[i] = max(0.0, min(1.0, value[i] + direction * mutation_strength))
    
    def _mutate_structure(self, pattern: CognitivePattern):
        """Mutate structural elements of the pattern"""
        genes = pattern.genes
        
        if pattern.pattern_type == "hypergraph":
            nodes = genes.get("nodes", [])
            edges = genes.get("edges", [])
            
            if random.random() < 0.5 and len(nodes) > 2:
                # Remove a node
                nodes.pop(random.randint(0, len(nodes) - 1))
            else:
                # Add a node
                nodes.append({
                    "id": f"node_{len(nodes)}",
                    "semantic_weight": random.uniform(0.1, 1.0),
                    "keywords": [f"concept_{random.randint(1, 100)}"]
                })
    
    def _mutate_attention(self, pattern: CognitivePattern):
        """Mutate attention allocation in the pattern"""
        genes = pattern.genes
        attention_weights = genes.get("attention_weights", [])
        
        if attention_weights:
            # Redistribute attention randomly
            for i in range(len(attention_weights)):
                if random.random() < 0.3:
                    attention_weights[i] = random.uniform(0.1, 1.0)
    
    def _mutate_semantics(self, pattern: CognitivePattern):
        """Mutate semantic aspects of the pattern"""
        genes = pattern.genes
        
        # Update keywords
        for key in ["keywords"]:
            if key in genes and isinstance(genes[key], list):
                if random.random() < 0.5:
                    # Replace a keyword
                    if genes[key]:
                        idx = random.randint(0, len(genes[key]) - 1)
                        genes[key][idx] = f"concept_{random.randint(1, 100)}"
                else:
                    # Add a keyword
                    genes[key].append(f"concept_{random.randint(1, 100)}")
    
    def _mutate_tensor_shape(self, pattern: CognitivePattern):
        """Mutate tensor shape (if applicable)"""
        if pattern.pattern_type in ["tensor", "hybrid"]:
            genes = pattern.genes
            
            if "shape" in genes:
                shape = genes["shape"]
                if shape:
                    # Mutate one dimension
                    idx = random.randint(0, len(shape) - 1)
                    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
                    shape[idx] = random.choice(primes)
    
    def _calculate_generation_stats(self) -> Dict[str, Any]:
        """Calculate statistics for current generation"""
        if not self.population:
            return {}
        
        fitnesses = [p.fitness for p in self.population]
        
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": max(fitnesses),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "worst_fitness": min(fitnesses),
            "fitness_std": self._calculate_std(fitnesses),
            "diversity": self._calculate_diversity(),
            "timestamp": time.time()
        }
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 0.0
        
        # Simple diversity measure based on fitness variance
        fitnesses = [p.fitness for p in self.population]
        return self._calculate_std(fitnesses)
    
    def get_best_patterns(self, top_k: int = 5) -> List[CognitivePattern]:
        """Get top-k best patterns from evolution"""
        sorted_patterns = sorted(self.best_patterns, key=lambda p: p.fitness, reverse=True)
        return sorted_patterns[:top_k]
    
    def export_evolution_results(self) -> Dict[str, Any]:
        """Export complete evolution results"""
        return {
            "agent_id": self.agent_id,
            "parameters": {
                "population_size": self.parameters.population_size,
                "mutation_rate": self.parameters.mutation_rate,
                "crossover_rate": self.parameters.crossover_rate,
                "max_generations": self.parameters.max_generations
            },
            "final_generation": self.generation,
            "best_patterns": [p.to_dict() for p in self.get_best_patterns()],
            "evolution_history": self.evolution_history,
            "total_evaluations": len(self.fitness_evaluator.evaluation_history),
            "export_time": time.time()
        }

# Example usage and integration
if __name__ == "__main__":
    # Create evolutionary search instance
    moses_search = MOSESEvolutionarySearch("test_agent")
    
    # Initialize population
    moses_search.initialize_population()
    
    # Run evolution
    context = {
        "keywords": ["creativity", "reasoning", "attention"],
        "goal": "optimize_cognitive_patterns"
    }
    
    best_patterns = moses_search.evolve(generations=10, context=context)
    
    # Print results
    print(f"Evolution completed with {len(best_patterns)} best patterns:")
    for i, pattern in enumerate(best_patterns[:3]):
        print(f"  Pattern {i+1}: {pattern.pattern_type}, fitness={pattern.fitness:.3f}")
    
    # Export results
    results = moses_search.export_evolution_results()
    print(f"Total evaluations: {results['total_evaluations']}")
    print(f"Final generation: {results['final_generation']}")