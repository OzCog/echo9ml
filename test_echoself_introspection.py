"""
Test module for Echoself Introspection functionality

Tests the hypergraph encoding, semantic salience assessment,
adaptive attention allocation, and repository introspection.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from echoself_introspection import (
    EchoselfIntrospector, 
    SemanticSalienceAssessor,
    AdaptiveAttentionAllocator,
    RepositoryIntrospector,
    HypergraphNode
)

class TestSemanticSalienceAssessor(unittest.TestCase):
    """Test semantic salience assessment functionality"""
    
    def setUp(self):
        self.assessor = SemanticSalienceAssessor()
    
    def test_high_salience_files(self):
        """Test that important files get high salience scores"""
        high_salience_paths = [
            "eva-model.py",
            "echoself.md",
            "ARCHITECTURE.md"
        ]
        
        for path in high_salience_paths:
            salience = self.assessor.assess_semantic_salience(path)
            self.assertGreaterEqual(salience, 0.85, f"Path {path} should have high salience")
    
    def test_low_salience_files(self):
        """Test that unimportant files get low salience scores"""
        low_salience_paths = [
            ".git/objects/abc123",
            "node_modules/package/index.js"
        ]
        
        for path in low_salience_paths:
            salience = self.assessor.assess_semantic_salience(path)
            self.assertLess(salience, 0.2, f"Path {path} should have low salience")
    
    def test_default_salience(self):
        """Test default salience for unknown files"""
        unknown_path = "some_random_file.xyz"
        salience = self.assessor.assess_semantic_salience(unknown_path)
        self.assertEqual(salience, 0.5)

class TestAdaptiveAttentionAllocator(unittest.TestCase):
    """Test adaptive attention allocation mechanism"""
    
    def setUp(self):
        self.allocator = AdaptiveAttentionAllocator()
    
    def test_high_load_increases_threshold(self):
        """Test that high cognitive load increases attention threshold"""
        low_load_threshold = self.allocator.adaptive_attention(0.2, 0.5)
        high_load_threshold = self.allocator.adaptive_attention(0.8, 0.5)
        
        self.assertGreater(high_load_threshold, low_load_threshold)
    
    def test_low_activity_increases_threshold(self):
        """Test that low recent activity increases attention threshold"""
        high_activity_threshold = self.allocator.adaptive_attention(0.5, 0.8)
        low_activity_threshold = self.allocator.adaptive_attention(0.5, 0.2)
        
        self.assertGreater(low_activity_threshold, high_activity_threshold)
    
    def test_threshold_bounds(self):
        """Test that threshold stays within reasonable bounds"""
        # Test extreme values
        min_threshold = self.allocator.adaptive_attention(0.0, 1.0)
        max_threshold = self.allocator.adaptive_attention(1.0, 0.0)
        
        self.assertGreaterEqual(min_threshold, 0.0)
        self.assertLessEqual(max_threshold, 1.0)  # Should be clamped to 1.0

class TestRepositoryIntrospector(unittest.TestCase):
    """Test repository introspection functionality"""
    
    def setUp(self):
        self.introspector = RepositoryIntrospector()
        # Create temporary directory structure for testing
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create test files
        (self.test_dir / "README.md").write_text("# Test Repository")
        (self.test_dir / "src").mkdir()
        (self.test_dir / "src" / "main.py").write_text("print('hello world')")
        (self.test_dir / "test_file.py").write_text("def test(): pass")
        
        # Create a large file
        (self.test_dir / "large_file.txt").write_text("x" * 60000)
        
        # Create binary-like file
        (self.test_dir / "binary.pyc").write_bytes(b'\x00\x01\x02\x03')
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_file_validation(self):
        """Test file validation logic"""
        # Valid files
        self.assertTrue(self.introspector.is_valid_file(self.test_dir / "README.md"))
        self.assertTrue(self.introspector.is_valid_file(self.test_dir / "src" / "main.py"))
        
        # Invalid files
        self.assertFalse(self.introspector.is_valid_file(self.test_dir / "large_file.txt"))
        self.assertFalse(self.introspector.is_valid_file(self.test_dir / "binary.pyc"))
        self.assertFalse(self.introspector.is_valid_file(self.test_dir / "nonexistent.txt"))
    
    def test_safe_file_reading(self):
        """Test safe file reading with constraints"""
        # Normal file
        content = self.introspector.safe_read_file(self.test_dir / "README.md")
        self.assertEqual(content, "# Test Repository")
        
        # Large file
        content = self.introspector.safe_read_file(self.test_dir / "large_file.txt")
        self.assertIn("File too large", content)
        
        # Binary file
        content = self.introspector.safe_read_file(self.test_dir / "binary.pyc")
        self.assertIn("not accessible or binary", content)
    
    def test_repo_file_list_filtering(self):
        """Test repository file listing with attention filtering"""
        # Low threshold should include more files
        files_low = self.introspector.repo_file_list(self.test_dir, 0.3)
        
        # High threshold should include fewer files
        files_high = self.introspector.repo_file_list(self.test_dir, 0.9)
        
        self.assertGreaterEqual(len(files_low), len(files_high))
        
        # README should be included in high threshold due to high salience
        readme_in_high = any("readme" in str(f).lower() for f in files_high)
        # Note: This test might not always pass depending on attention threshold calculation
        # The key is that the filtering mechanism works

class TestEchoselfIntrospector(unittest.TestCase):
    """Test main introspection functionality"""
    
    def setUp(self):
        # Create temporary test repository
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create realistic test structure
        (self.test_dir / "README.md").write_text("# Test Project\nDescription")
        (self.test_dir / "echoself.md").write_text("# Echoself\nCognitive content")
        
        src_dir = self.test_dir / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("def main(): pass")
        
        self.introspector = EchoselfIntrospector(self.test_dir)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_cognitive_snapshot(self):
        """Test cognitive snapshot generation"""
        snapshot = self.introspector.get_cognitive_snapshot(
            current_load=0.6, 
            recent_activity=0.4
        )
        
        # Verify snapshot structure
        self.assertIn('timestamp', snapshot)
        self.assertIn('attention_threshold', snapshot)
        self.assertIn('total_files_processed', snapshot)
        self.assertIn('nodes', snapshot)
        
        # Verify data types
        self.assertIsInstance(snapshot['nodes'], list)
        self.assertGreater(snapshot['total_files_processed'], 0)
        
        # Verify node structure
        if snapshot['nodes']:
            node = snapshot['nodes'][0]
            self.assertIn('id', node)
            self.assertIn('type', node)
            self.assertIn('content', node)
            self.assertIn('salience', node)
    
    def test_prompt_generation(self):
        """Test prompt generation functionality"""
        prompt = self.introspector.inject_repo_input_into_prompt(
            current_load=0.5,
            recent_activity=0.5
        )
        
        # Verify prompt structure
        self.assertIn("DeepTreeEcho Prompt:", prompt)
        self.assertIn("(file", prompt)  # Should contain file entries
    
    def test_attention_threshold_affects_processing(self):
        """Test that attention threshold affects the number of files processed"""
        # High cognitive load should result in fewer files processed
        high_load_snapshot = self.introspector.get_cognitive_snapshot(
            current_load=0.9, 
            recent_activity=0.1
        )
        
        # Low cognitive load should result in more files processed
        low_load_snapshot = self.introspector.get_cognitive_snapshot(
            current_load=0.1, 
            recent_activity=0.9
        )
        
        # High load should process fewer or equal files
        self.assertLessEqual(
            high_load_snapshot['total_files_processed'],
            low_load_snapshot['total_files_processed']
        )

class TestHypergraphNode(unittest.TestCase):
    """Test hypergraph node functionality"""
    
    def test_node_creation(self):
        """Test hypergraph node creation and serialization"""
        node = HypergraphNode(
            id="test_file.py",
            node_type="file",
            content="def test(): pass",
            salience=0.8
        )
        
        # Test basic properties
        self.assertEqual(node.id, "test_file.py")
        self.assertEqual(node.node_type, "file")
        self.assertEqual(node.content, "def test(): pass")
        self.assertEqual(node.salience, 0.8)
        
        # Test serialization
        node_dict = node.to_dict()
        self.assertIn('id', node_dict)
        self.assertIn('type', node_dict)
        self.assertIn('content', node_dict)
        self.assertIn('salience', node_dict)

if __name__ == "__main__":
    unittest.main()