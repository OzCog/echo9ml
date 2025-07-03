"""
Echoself Introspection Module

Hypergraph-encoded recursive self-model introspection
Inspired by DeepTreeEcho/Eva Self Model and echoself.md

This module implements the cognitive introspection pipeline with:
- Recursive repository traversal with attention filtering
- Semantic salience assessment
- Adaptive attention allocation
- Hypergraph encoding for neural-symbolic integration
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class HypergraphNode:
    """Hypergraph node representation (translated from Scheme make-node)"""
    id: str
    node_type: str
    content: str
    links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    salience: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return {
            'id': self.id,
            'type': self.node_type,
            'content': self.content,
            'links': self.links,
            'metadata': self.metadata,
            'salience': self.salience,
            'timestamp': self.timestamp
        }

class SemanticSalienceAssessor:
    """Semantic salience assessment based on heuristics"""
    
    def __init__(self):
        # Salience weights for different path patterns (order matters - check most specific first)
        self.salience_patterns = [
            ('btree-psi.scm', 0.98),
            ('eva-model', 0.95),
            ('echoself.md', 0.95),
            ('eva-behavior', 0.92),
            ('readme', 0.9),  # Case insensitive
            ('architecture.md', 0.9),
            ('deep_tree_echo', 0.85),
            ('components.md', 0.85),
            ('src/', 0.85),
            ('cognitive_', 0.8),
            ('memory_', 0.8),
            ('btree.scm', 0.7),
            ('.md', 0.7),
            ('.py', 0.6),
            ('test_', 0.5),
            ('__pycache__', 0.1),
            ('.git', 0.1),
            ('node_modules', 0.1),
        ]
        
    def assess_semantic_salience(self, path: str) -> float:
        """
        Assign salience scores based on heuristics
        Translated from Scheme semantic-salience function
        """
        path_str = str(path).lower()
        
        # Check patterns in order of specificity
        for pattern, salience in self.salience_patterns:
            if pattern.lower() in path_str:
                return salience
                
        # Default salience for unmatched files
        return 0.5

class AdaptiveAttentionAllocator:
    """Adaptive attention allocation mechanism"""
    
    def __init__(self):
        self.base_threshold = 0.5
        
    def adaptive_attention(self, current_load: float, recent_activity: float) -> float:
        """
        Dynamically adjust attention threshold based on cognitive load and recent activity
        Translated from Scheme adaptive-attention function
        
        High load or low activity leads to higher threshold (less data processed)
        """
        threshold = self.base_threshold + (current_load * 0.3) + (0.2 - recent_activity)
        # Ensure threshold stays within reasonable bounds
        return max(0.0, min(1.0, threshold))

class RepositoryIntrospector:
    """Recursive repository introspection with attention filtering"""
    
    def __init__(self, max_file_size: int = 50000):
        self.max_file_size = max_file_size
        self.salience_assessor = SemanticSalienceAssessor()
        self.attention_allocator = AdaptiveAttentionAllocator()
        
    def is_valid_file(self, path: Path) -> bool:
        """Check if file should be processed"""
        if not path.exists() or not path.is_file():
            return False
            
        # Skip binary files and other non-text files
        binary_extensions = {'.pyc', '.so', '.dll', '.exe', '.bin', '.jpg', '.png', '.gif', '.pdf'}
        if path.suffix.lower() in binary_extensions:
            return False
            
        try:
            file_size = path.stat().st_size
            return file_size > 0 and file_size <= self.max_file_size
        except (OSError, IOError):
            return False
    
    def safe_read_file(self, path: Path) -> str:
        """
        Safely read file content with size constraints
        Translated from Scheme safe-read-file function
        """
        try:
            if not path.exists() or not path.is_file():
                return "[File not accessible]"
                
            file_size = path.stat().st_size
            
            # Check file size first
            if file_size > self.max_file_size:
                return f"[File too large: {file_size} bytes, summarized or omitted]"
            
            # Check if it's a binary file
            binary_extensions = {'.pyc', '.so', '.dll', '.exe', '.bin', '.jpg', '.png', '.gif', '.pdf'}
            if path.suffix.lower() in binary_extensions:
                return "[File not accessible or binary]"
                
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
                
        except (IOError, OSError, UnicodeDecodeError) as e:
            logger.warning(f"Error reading file {path}: {e}")
            return f"[Error reading file: {e}]"
    
    def repo_file_list(self, root: Path, attention_threshold: float) -> List[Path]:
        """
        Recursive repository traversal with attention filtering
        Translated from Scheme repo-file-list function
        """
        if not root.exists():
            return []
            
        if root.is_file():
            salience = self.salience_assessor.assess_semantic_salience(str(root))
            if salience > attention_threshold:
                return [root]
            else:
                return []
        
        # Directory traversal
        files = []
        try:
            for item in root.iterdir():
                if item.name.startswith('.') and item.name not in {'.gitignore', '.env.example'}:
                    continue  # Skip hidden files except important ones
                    
                files.extend(self.repo_file_list(item, attention_threshold))
                
        except (OSError, PermissionError) as e:
            logger.warning(f"Error accessing directory {root}: {e}")
            
        return files

    def assemble_hypergraph_input(self, root: Path, attention_threshold: float) -> List[HypergraphNode]:
        """
        Assemble hypergraph-encoded input from repository files
        Translated from Scheme assemble-hypergraph-input function
        """
        files = self.repo_file_list(root, attention_threshold)
        nodes = []
        
        for path in files:
            content = self.safe_read_file(path)
            salience = self.salience_assessor.assess_semantic_salience(str(path))
            
            node = HypergraphNode(
                id=str(path),
                node_type='file',
                content=content,
                salience=salience,
                metadata={
                    'file_size': len(content),
                    'file_extension': path.suffix,
                    'relative_path': str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
                }
            )
            nodes.append(node)
            
        return nodes

class HypergraphStringSerializer:
    """Hypergraph to string serialization for prompt integration"""
    
    @staticmethod
    def hypergraph_to_string(nodes: List[HypergraphNode]) -> str:
        """
        Convert hypergraph nodes to string representation
        Translated from Scheme hypergraph->string function
        """
        result = []
        for node in nodes:
            # Format: (file "path" "content")
            escaped_content = node.content.replace('"', '\\"').replace('\n', '\\n')[:1000]  # Limit content length
            result.append(f'(file "{node.id}" "{escaped_content}")')
        
        return '\n'.join(result)

class EchoselfIntrospector:
    """Main introspection class integrating all components"""
    
    def __init__(self, repository_root: Optional[Path] = None):
        self.repository_root = repository_root or Path.cwd()
        self.introspector = RepositoryIntrospector()
        self.serializer = HypergraphStringSerializer()
        self.attention_allocator = AdaptiveAttentionAllocator()
        
    def prompt_template(self, input_content: str) -> str:
        """
        Generate prompt template with repository input
        Translated from Scheme prompt-template function
        """
        return f"DeepTreeEcho Prompt:\n{input_content}"
    
    def inject_repo_input_into_prompt(self, current_load: float = 0.6, 
                                    recent_activity: float = 0.4) -> str:
        """
        Complete introspection pipeline: inject repository input into prompt
        Translated from Scheme inject-repo-input-into-prompt function
        """
        attention_threshold = self.attention_allocator.adaptive_attention(
            current_load, recent_activity
        )
        
        logger.info(f"Using attention threshold: {attention_threshold}")
        
        nodes = self.introspector.assemble_hypergraph_input(
            self.repository_root, attention_threshold
        )
        
        logger.info(f"Assembled {len(nodes)} hypergraph nodes")
        
        hypergraph_string = self.serializer.hypergraph_to_string(nodes)
        
        return self.prompt_template(hypergraph_string)
    
    def get_cognitive_snapshot(self, current_load: float = 0.6, 
                             recent_activity: float = 0.4) -> Dict[str, Any]:
        """
        Get comprehensive cognitive snapshot for neural-symbolic integration
        """
        attention_threshold = self.attention_allocator.adaptive_attention(
            current_load, recent_activity
        )
        
        nodes = self.introspector.assemble_hypergraph_input(
            self.repository_root, attention_threshold
        )
        
        # Aggregate statistics
        total_files = len(nodes)
        avg_salience = sum(node.salience for node in nodes) / total_files if total_files > 0 else 0
        high_salience_files = [node for node in nodes if node.salience > 0.8]
        
        return {
            'timestamp': time.time(),
            'attention_threshold': attention_threshold,
            'cognitive_load': current_load,
            'recent_activity': recent_activity,
            'total_files_processed': total_files,
            'average_salience': avg_salience,
            'high_salience_count': len(high_salience_files),
            'nodes': [node.to_dict() for node in nodes],
            'repository_root': str(self.repository_root)
        }

# Example usage and integration point
def main():
    """Example usage of the introspection system"""
    introspector = EchoselfIntrospector()
    
    # Example of adaptive attention usage
    prompt = introspector.inject_repo_input_into_prompt(
        current_load=0.6, 
        recent_activity=0.4
    )
    
    print("Generated prompt snippet:")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    # Get cognitive snapshot
    snapshot = introspector.get_cognitive_snapshot()
    print(f"\nCognitive snapshot: {snapshot['total_files_processed']} files, "
          f"avg salience: {snapshot['average_salience']:.3f}")

if __name__ == "__main__":
    main()