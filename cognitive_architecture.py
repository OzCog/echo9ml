import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
import logging
from pathlib import Path
import json
import datetime
from collections import deque
import time

# Import the echoself introspection module
try:
    from echoself_introspection import EchoselfIntrospection
except ImportError:
    EchoselfIntrospection = None

class MemoryType(Enum):
    DECLARATIVE = "declarative"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"
    INTENTIONAL = "intentional"
    EMOTIONAL = "emotional"

@dataclass
class Memory:
    content: str
    memory_type: MemoryType
    timestamp: float
    associations: Set[str] = field(default_factory=set)
    emotional_valence: float = 0.0
    importance: float = 0.0
    context: Dict = field(default_factory=dict)

@dataclass
class Goal:
    description: str
    priority: float
    deadline: Optional[float]
    subgoals: List['Goal'] = field(default_factory=list)
    status: str = "pending"
    progress: float = 0.0
    context: Dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

class PersonalityTrait:
    def __init__(self, name: str, base_value: float):
        self.name = name
        self.base_value = base_value
        self.current_value = base_value
        self.history = deque(maxlen=1000)
        
    def update(self, value: float, context: Dict):
        self.current_value = 0.7 * self.current_value + 0.3 * value
        self.history.append((datetime.datetime.now(), value, context))

class CognitiveArchitecture:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memories: Dict[str, Memory] = {}
        self.goals: List[Goal] = []
        self.active_goals: List[Goal] = []
        self.personality_traits = {
            "curiosity": PersonalityTrait("curiosity", 0.8),
            "adaptability": PersonalityTrait("adaptability", 0.9),
            "persistence": PersonalityTrait("persistence", 0.7),
            "creativity": PersonalityTrait("creativity", 0.8),
            "analytical": PersonalityTrait("analytical", 0.85),
            "social": PersonalityTrait("social", 0.6)
        }
        
        # Initialize memory paths
        self.memory_path = Path.home() / '.deep_tree_echo' / 'memories'
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize cognitive paths
        self.echo_dir = Path.home() / '.deep_tree_echo'
        self.cognitive_dir = self.echo_dir / 'cognitive'
        self.cognitive_dir.mkdir(parents=True, exist_ok=True)
        self.activity_file = self.cognitive_dir / 'activity.json'
        self.activities = []
        self._load_activities()
        
        # Initialize echoself introspection system
        self.echoself_introspection = None
        if EchoselfIntrospection:
            try:
                # Initialize introspection for the current working directory
                self.echoself_introspection = EchoselfIntrospection(".")
                self.logger.info("Echoself introspection system initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize echoself introspection: {e}")
        
        # Load existing memories and goals
        self._load_state()
        
    def _load_state(self):
        """Load memories and goals from disk"""
        try:
            memory_file = self.memory_path / 'memories.json'
            if memory_file.exists():
                with open(memory_file) as f:
                    data = json.load(f)
                    for mem_data in data.get('memories', []):
                        self.memories[mem_data['id']] = Memory(**mem_data)
                    for goal_data in data.get('goals', []):
                        self.goals.append(Goal(**goal_data))
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            
    def _load_activities(self):
        """Load existing activities"""
        if self.activity_file.exists():
            try:
                with open(self.activity_file) as f:
                    self.activities = json.load(f)
            except:
                self.activities = []
                
    def _save_activities(self):
        """Save activities to file"""
        with open(self.activity_file, 'w') as f:
            json.dump(self.activities[-1000:], f)  # Keep last 1000 activities
            
    def _log_activity(self, description: str, context: Dict = None):
        """Log a cognitive activity"""
        try:
            activity_file = Path('activity_logs/cognitive/activity.json')
            
            # Read existing activities
            current = []
            if activity_file.exists():
                with open(activity_file) as f:
                    current = json.load(f)
            
            # Add new activity
            activity = {
                'time': time.time(),
                'description': description,
                'context': context or {}
            }
            current.append(activity)
            
            # Keep last 1000 activities
            if len(current) > 1000:
                current = current[-1000:]
            
            # Write back
            with open(activity_file, 'w') as f:
                json.dump(current, f)
                
        except Exception as e:
            self.logger.error(f"Error logging activity: {e}")
            
    def save_state(self):
        """Save current state to disk"""
        self._log_activity("Saving cognitive state")
        try:
            data = {
                'memories': [self._memory_to_dict(m) for m in self.memories.values()],
                'goals': [self._goal_to_dict(g) for g in self.goals]
            }
            with open(self.memory_path / 'memories.json', 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            
    def generate_goals(self, context: Dict) -> List[Goal]:
        """Generate new goals based on current state and context"""
        self._log_activity(
            "Generating new goals",
            {'context': context}
        )
        goals = []
        
        # Factor in personality traits
        curiosity = self.personality_traits["curiosity"].current_value
        creativity = self.personality_traits["creativity"].current_value
        analytical = self.personality_traits["analytical"].current_value
        
        # Learning goals based on curiosity
        if curiosity > 0.6:
            knowledge_gaps = self._identify_knowledge_gaps()
            for gap in knowledge_gaps:
                goals.append(Goal(
                    description=f"Learn about: {gap}",
                    priority=curiosity * 0.8,
                    deadline=None,
                    context={"type": "learning", "area": gap}
                ))
                
        # System improvement goals based on analytical trait
        if analytical > 0.7:
            improvement_areas = self._analyze_system_performance()
            for area in improvement_areas:
                goals.append(Goal(
                    description=f"Improve system {area}",
                    priority=analytical * 0.9,
                    deadline=None,
                    context={"type": "improvement", "area": area}
                ))
                
        # Creative exploration goals
        if creativity > 0.6:
            exploration_ideas = self._generate_creative_ideas()
            for idea in exploration_ideas:
                goals.append(Goal(
                    description=f"Explore: {idea}",
                    priority=creativity * 0.7,
                    deadline=None,
                    context={"type": "exploration", "idea": idea}
                ))
                
        return goals
    
    def update_personality(self, experiences: List[Dict]):
        """Update personality traits based on experiences"""
        for exp in experiences:
            # Update curiosity based on learning experiences
            if exp.get('type') == 'learning':
                success = exp.get('success', 0.5)
                self.personality_traits["curiosity"].update(
                    success * 1.2,
                    {"experience": exp}
                )
                
            # Update adaptability based on change handling
            elif exp.get('type') == 'adaptation':
                effectiveness = exp.get('effectiveness', 0.5)
                self.personality_traits["adaptability"].update(
                    effectiveness,
                    {"experience": exp}
                )
                
            # Update persistence based on challenge handling
            elif exp.get('type') == 'challenge':
                resolution = exp.get('resolution', 0.5)
                self.personality_traits["persistence"].update(
                    resolution,
                    {"experience": exp}
                )
                
    def learn_from_experience(self, experience: Dict):
        """Learn from new experiences"""
        self._log_activity(
            "Learning from experience",
            {'experience': experience}
        )
        # Create memory
        memory = Memory(
            content=experience.get('description', ''),
            memory_type=MemoryType(experience.get('type', 'episodic')),
            timestamp=datetime.datetime.now().timestamp(),
            emotional_valence=experience.get('emotional_impact', 0.0),
            importance=experience.get('importance', 0.5),
            context=experience
        )
        
        # Store memory
        self.memories[str(len(self.memories))] = memory
        
        # Update personality based on experience
        self.update_personality([experience])
        
        # Generate new goals if needed
        if experience.get('importance', 0) > 0.7:
            new_goals = self.generate_goals({"trigger": experience})
            self.goals.extend(new_goals)
            
    def _identify_knowledge_gaps(self) -> List[str]:
        """Identify areas where knowledge is lacking"""
        # Analyze memories and identify areas with low coverage
        knowledge_areas = {}
        for memory in self.memories.values():
            if memory.memory_type == MemoryType.DECLARATIVE:
                area = memory.context.get('area', 'general')
                knowledge_areas[area] = knowledge_areas.get(area, 0) + 1
                
        # Find areas with low coverage
        gaps = []
        for area, count in knowledge_areas.items():
            if count < 5:  # Arbitrary threshold
                gaps.append(area)
                
        return gaps
    
    def _analyze_system_performance(self) -> List[str]:
        """Analyze system performance and identify areas for improvement"""
        # Example areas to monitor
        areas = ['memory_usage', 'response_time', 'learning_rate', 'goal_completion']
        improvements = []
        
        # Add areas that need improvement based on metrics
        for area in areas:
            if self._get_performance_metric(area) < 0.7:
                improvements.append(area)
                
        return improvements
    
    def _generate_creative_ideas(self) -> List[str]:
        """Generate new ideas for exploration"""
        # Combine existing knowledge in novel ways
        ideas = []
        memory_pairs = list(zip(
            self.memories.values(),
            self.memories.values()
        ))
        
        for mem1, mem2 in memory_pairs[:5]:  # Limit to prevent explosion
            if mem1.memory_type != mem2.memory_type:
                idea = f"Explore connection between {mem1.content} and {mem2.content}"
                ideas.append(idea)
                
        return ideas
    
    def _get_performance_metric(self, metric: str) -> float:
        """Get performance metric value"""
        # Placeholder for actual metrics
        return np.random.random()
    
    def _memory_to_dict(self, memory: Memory) -> Dict:
        """Convert memory to dictionary for storage"""
        return {
            'content': memory.content,
            'memory_type': memory.memory_type.value,
            'timestamp': memory.timestamp,
            'associations': list(memory.associations),
            'emotional_valence': memory.emotional_valence,
            'importance': memory.importance,
            'context': memory.context
        }
        
    def _goal_to_dict(self, goal: Goal) -> Dict:
        """Convert goal to dictionary for storage"""
        return {
            'description': goal.description,
            'priority': goal.priority,
            'deadline': goal.deadline,
            'status': goal.status,
            'progress': goal.progress,
            'context': goal.context,
            'dependencies': goal.dependencies,
            'subgoals': [self._goal_to_dict(g) for g in goal.subgoals]
        }

    def process_experience(self, experience: str, context: Dict = None) -> None:
        """Process a new experience"""
        self._log_activity(f"Processing experience: {experience}", context)
        # Rest of the method...

    def generate_goal(self, description: str, priority: float = 0.5,
                   deadline: Optional[float] = None) -> Goal:
        """Generate a new goal"""
        self._log_activity(f"Generated goal: {description}", 
                         {'priority': priority, 'deadline': deadline})
        # Rest of the method...

    def update_goal(self, goal: Goal, progress: float) -> None:
        """Update goal progress"""
        self._log_activity(f"Updated goal: {goal.description}", 
                         {'progress': progress, 'status': goal.status})
        # Rest of the method...

    def generate_and_update_goals(self, experiences: List[Dict]):
        """Generate and update goals based on experiences"""
        self._log_activity("Generating and updating goals", {'experiences': experiences})
        for exp in experiences:
            new_goals = self.generate_goals({"trigger": exp})
            self.goals.extend(new_goals)
            for goal in self.goals:
                if goal.status == "pending":
                    goal.progress += exp.get('progress', 0.1)
                    if goal.progress >= 1.0:
                        goal.status = "completed"
                        self._log_activity(f"Goal completed: {goal.description}", {'goal': goal})
                    else:
                        self._log_activity(f"Goal updated: {goal.description}", {'goal': goal})
        self.save_state()

    def enhanced_memory_management(self, memory: Memory):
        """Enhance memory management with better logging and error handling"""
        try:
            self.memories[str(len(self.memories))] = memory
            self._log_activity("Memory added", {'memory': memory})
        except Exception as e:
            self.logger.error(f"Error adding memory: {str(e)}")
            self._log_activity("Error adding memory", {'error': str(e)})

    def enhanced_goal_management(self, goal: Goal):
        """Enhance goal management with better logging and error handling"""
        try:
            self.goals.append(goal)
            self._log_activity("Goal added", {'goal': goal})
        except Exception as e:
            self.logger.error(f"Error adding goal: {str(e)}")
            self._log_activity("Error adding goal", {'error': str(e)})

    def enhanced_personality_management(self, trait: PersonalityTrait, value: float, context: Dict):
        """Enhance personality management with better logging and error handling"""
        try:
            trait.update(value, context)
            self._log_activity("Personality trait updated", {'trait': trait, 'value': value, 'context': context})
        except Exception as e:
            self.logger.error(f"Error updating personality trait: {str(e)}")
            self._log_activity("Error updating personality trait", {'error': str(e)})

    def perform_recursive_introspection(self, current_cognitive_load: float = None, 
                                      recent_activity_level: float = None) -> Optional[str]:
        """
        Perform recursive self-model introspection using the echoself system
        
        Args:
            current_cognitive_load: Current cognitive load (0.0-1.0), defaults to calculated value
            recent_activity_level: Recent activity level (0.0-1.0), defaults to calculated value
            
        Returns:
            Introspection prompt or None if system unavailable
        """
        if not self.echoself_introspection:
            self.logger.warning("Echoself introspection system not available")
            return None
        
        try:
            # Calculate cognitive load if not provided
            if current_cognitive_load is None:
                current_cognitive_load = self._calculate_current_cognitive_load()
            
            # Calculate recent activity if not provided  
            if recent_activity_level is None:
                recent_activity_level = self._calculate_recent_activity()
            
            # Generate introspection prompt
            prompt = self.echoself_introspection.inject_repo_input_into_prompt(
                current_cognitive_load, recent_activity_level
            )
            
            # Log the introspection activity
            self._log_activity(
                "Performed recursive introspection",
                {
                    "cognitive_load": current_cognitive_load,
                    "activity_level": recent_activity_level,
                    "prompt_length": len(prompt)
                }
            )
            
            # Store introspection as a memory
            introspection_memory = Memory(
                content=f"Recursive introspection performed with load {current_cognitive_load:.3f}",
                memory_type=MemoryType.INTENTIONAL,
                timestamp=datetime.datetime.now().timestamp(),
                importance=0.8,
                context={
                    "type": "introspection",
                    "cognitive_load": current_cognitive_load,
                    "activity_level": recent_activity_level
                }
            )
            self.memories[f"introspection_{int(time.time())}"] = introspection_memory
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error during recursive introspection: {e}")
            self._log_activity("Error during introspection", {'error': str(e)})
            return None
    
    def get_introspection_metrics(self) -> Dict[str, any]:
        """Get metrics from the introspection system"""
        if not self.echoself_introspection:
            return {"error": "Introspection system not available"}
        
        try:
            return self.echoself_introspection.get_attention_metrics()
        except Exception as e:
            self.logger.error(f"Error getting introspection metrics: {e}")
            return {"error": str(e)}
    
    def export_introspection_data(self, output_path: str) -> bool:
        """Export introspection hypergraph data"""
        if not self.echoself_introspection:
            self.logger.warning("Introspection system not available for export")
            return False
        
        try:
            self.echoself_introspection.export_hypergraph(output_path)
            self._log_activity("Exported introspection data", {"output_path": output_path})
            return True
        except Exception as e:
            self.logger.error(f"Error exporting introspection data: {e}")
            return False
    
    def adaptive_goal_generation_with_introspection(self) -> List[Goal]:
        """
        Generate goals using introspection-informed analysis
        """
        goals = []
        
        # Perform introspection to understand current state
        introspection_prompt = self.perform_recursive_introspection()
        
        if introspection_prompt:
            # Generate introspection-informed goals
            curiosity = self.personality_traits["curiosity"].current_value
            analytical = self.personality_traits["analytical"].current_value
            
            # Self-improvement goals based on introspection
            if analytical > 0.7:
                goals.append(Goal(
                    description="Analyze recursive self-model integration opportunities",
                    priority=analytical * 0.9,
                    deadline=None,
                    context={
                        "type": "introspection_analysis",
                        "introspection_available": True
                    }
                ))
            
            # Cognitive architecture optimization goals
            if curiosity > 0.6:
                goals.append(Goal(
                    description="Explore hypergraph-encoded cognitive patterns",
                    priority=curiosity * 0.8,
                    deadline=None,
                    context={
                        "type": "cognitive_exploration",
                        "method": "hypergraph_analysis"
                    }
                ))
        
        # Add traditional goal generation
        traditional_goals = self.generate_goals({"trigger": "introspection_enhanced"})
        goals.extend(traditional_goals)
        
        return goals
    
    def _calculate_current_cognitive_load(self) -> float:
        """Calculate current cognitive load based on system state"""
        # Base load factors
        memory_load = min(len(self.memories) / 1000, 1.0)  # Normalize to 1000 memories
        goal_load = min(len(self.goals) / 50, 1.0)  # Normalize to 50 goals
        activity_load = min(len(self.activities) / 100, 1.0)  # Recent activities
        
        # Weighted combination
        total_load = (memory_load * 0.4 + goal_load * 0.3 + activity_load * 0.3)
        
        return min(max(total_load, 0.1), 0.9)  # Clamp between 0.1 and 0.9
    
    def _calculate_recent_activity(self) -> float:
        """Calculate recent activity level"""
        if not self.activities:
            return 0.1
        
        # Count activities in the last hour
        current_time = time.time()
        recent_activities = [
            a for a in self.activities 
            if isinstance(a, dict) and 
               current_time - a.get('time', 0) < 3600  # Last hour
        ]
        
        # Normalize to reasonable scale
        activity_level = min(len(recent_activities) / 20, 1.0)
        return max(activity_level, 0.1)
