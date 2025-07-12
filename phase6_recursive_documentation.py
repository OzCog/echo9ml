#!/usr/bin/env python3
"""
Phase 6: Recursive Documentation and Flowchart Auto-Generation
Auto-generate architectural flowcharts for every module

This module implements recursive documentation generation as specified in Phase 6
of the Distributed Agentic Cognitive Grammar Network implementation.
"""

import ast
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import importlib
import inspect
import re
from datetime import datetime


class RecursiveDocumentationGenerator:
    """
    Auto-generates comprehensive documentation and flowcharts for all modules
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.documentation = {}
        self.flowcharts = {}
        self.module_dependencies = {}
        self.architectural_patterns = {}
        
    def discover_all_modules(self) -> List[str]:
        """Discover all Python modules in the project"""
        modules = []
        for py_file in self.project_root.glob("*.py"):
            if not py_file.name.startswith("test_") and py_file.name != "setup.py":
                modules.append(py_file.stem)
        return sorted(modules)
    
    def analyze_module_structure(self, module_name: str) -> Dict[str, Any]:
        """Analyze the structure of a module for documentation"""
        module_path = self.project_root / f"{module_name}.py"
        if not module_path.exists():
            return {"error": f"Module {module_name} not found"}
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            analysis = {
                "module": module_name,
                "docstring": ast.get_docstring(tree),
                "imports": [],
                "classes": [],
                "functions": [],
                "constants": [],
                "dependencies": set(),
                "cognitive_patterns": [],
                "architectural_role": self._determine_architectural_role(module_name, source)
            }
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis["imports"].append(alias.name)
                        analysis["dependencies"].add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis["imports"].append(node.module)
                        analysis["dependencies"].add(node.module.split('.')[0])
            
            # Extract classes
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node),
                        "methods": [],
                        "bases": [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                        "line": node.lineno,
                        "cognitive_features": self._extract_cognitive_features(node.name, ast.get_docstring(node) or "")
                    }
                    
                    # Extract methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = {
                                "name": item.name,
                                "docstring": ast.get_docstring(item),
                                "args": [arg.arg for arg in item.args.args],
                                "line": item.lineno,
                                "returns": self._extract_return_type(item),
                                "complexity": self._calculate_complexity(item)
                            }
                            class_info["methods"].append(method_info)
                    
                    analysis["classes"].append(class_info)
                
                elif isinstance(node, ast.FunctionDef):
                    function_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node),
                        "args": [arg.arg for arg in node.args.args],
                        "line": node.lineno,
                        "returns": self._extract_return_type(node),
                        "complexity": self._calculate_complexity(node),
                        "cognitive_features": self._extract_cognitive_features(node.name, ast.get_docstring(node) or "")
                    }
                    analysis["functions"].append(function_info)
                
                elif isinstance(node, ast.Assign):
                    # Extract constants
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            analysis["constants"].append({
                                "name": target.id,
                                "line": node.lineno,
                                "value": self._safe_eval(node.value)
                            })
            
            # Identify cognitive patterns
            analysis["cognitive_patterns"] = self._identify_cognitive_patterns(analysis)
            
            return analysis
        
        except Exception as e:
            return {"error": str(e), "module": module_name}
    
    def _determine_architectural_role(self, module_name: str, source: str) -> str:
        """Determine the architectural role of a module"""
        roles = {
            "Core Engine": ["echo9ml", "deep_tree_echo"],
            "Cognitive Architecture": ["cognitive_architecture", "cognitive_integration"],
            "Neural Processing": ["neural_", "ggml_", "tensor_"],
            "Symbolic Processing": ["symbolic_", "hypergraph_"],
            "Memory Systems": ["memory_", "episodic_", "declarative_"],
            "Attention Systems": ["attention_", "ecan_", "allocation"],
            "Interface Layer": ["web_", "gui_", "browser_", "selenium_"],
            "Testing Framework": ["test_", "validate_", "verify_"],
            "Documentation": ["doc", "generate", "analysis"],
            "Utility": ["util", "helper", "tool"]
        }
        
        for role, patterns in roles.items():
            if any(pattern in module_name.lower() for pattern in patterns):
                return role
        
        # Analyze source content for role hints
        if any(keyword in source.lower() for keyword in ["class.*cognitive", "neural", "tensor"]):
            return "Cognitive Architecture"
        elif any(keyword in source.lower() for keyword in ["def.*test", "unittest", "pytest"]):
            return "Testing Framework"
        elif any(keyword in source.lower() for keyword in ["flask", "web", "http", "gui"]):
            return "Interface Layer"
        
        return "Utility"
    
    def _extract_cognitive_features(self, name: str, docstring: str) -> List[str]:
        """Extract cognitive features from names and docstrings"""
        features = []
        text = (name + " " + (docstring or "")).lower()
        
        cognitive_keywords = {
            "attention": ["attention", "focus", "salience"],
            "memory": ["memory", "remember", "recall", "episodic", "declarative"],
            "learning": ["learn", "adapt", "evolve", "train"],
            "reasoning": ["reason", "logic", "symbolic", "inference"],
            "perception": ["perceive", "sense", "input", "stimulus"],
            "action": ["action", "motor", "output", "behavior"],
            "emotion": ["emotion", "mood", "affect", "sentiment"],
            "metacognition": ["meta", "self", "recursive", "introspection"],
            "neural": ["neural", "network", "tensor", "activation"],
            "symbolic": ["symbolic", "rule", "graph", "hypergraph"]
        }
        
        for feature, keywords in cognitive_keywords.items():
            if any(keyword in text for keyword in keywords):
                features.append(feature)
        
        return features
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _extract_return_type(self, node: ast.FunctionDef) -> str:
        """Extract return type annotation if present"""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Constant):
                return str(node.returns.value)
            else:
                return str(node.returns)
        return "Any"
    
    def _safe_eval(self, node: ast.AST) -> str:
        """Safely evaluate AST node for constants"""
        try:
            if isinstance(node, ast.Constant):
                return str(node.value)
            elif isinstance(node, ast.Name):
                return node.id
            else:
                return "..."
        except:
            return "..."
    
    def _identify_cognitive_patterns(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify cognitive architectural patterns in the module"""
        patterns = []
        
        # Check for common cognitive patterns
        class_names = [cls["name"] for cls in analysis["classes"]]
        function_names = [func["name"] for func in analysis["functions"]]
        all_names = class_names + function_names
        
        # Pattern: Observer/Publisher-Subscriber
        if any(name.lower().endswith(("observer", "listener", "subscriber")) for name in all_names):
            patterns.append("Observer Pattern")
        
        # Pattern: Strategy
        if any("strategy" in name.lower() or "algorithm" in name.lower() for name in all_names):
            patterns.append("Strategy Pattern")
        
        # Pattern: Factory
        if any("factory" in name.lower() or "create" in name.lower() for name in all_names):
            patterns.append("Factory Pattern")
        
        # Pattern: Recursive Processing
        if any("recursive" in name.lower() or "recurse" in name.lower() for name in all_names):
            patterns.append("Recursive Pattern")
        
        # Pattern: Neural Network
        if any(keyword in name.lower() for name in all_names for keyword in ["neural", "layer", "network", "tensor"]):
            patterns.append("Neural Network Pattern")
        
        # Pattern: State Machine
        if any(keyword in name.lower() for name in all_names for keyword in ["state", "transition", "machine"]):
            patterns.append("State Machine Pattern")
        
        # Pattern: Memory Management
        if any(keyword in name.lower() for name in all_names for keyword in ["memory", "cache", "store", "retrieve"]):
            patterns.append("Memory Management Pattern")
        
        return patterns
    
    def generate_mermaid_flowchart(self, module_analysis: Dict[str, Any]) -> str:
        """Generate Mermaid flowchart for a module"""
        if "error" in module_analysis:
            return f"```mermaid\ngraph TD\n    ERROR[Error: {module_analysis['error']}]\n```"
        
        module_name = module_analysis["module"]
        mermaid = f"```mermaid\ngraph TD\n"
        mermaid += f"    {module_name}[{module_name}]\n"
        
        # Add classes
        for cls in module_analysis["classes"]:
            class_node = f"{module_name}_{cls['name']}"
            mermaid += f"    {class_node}[{cls['name']}]\n"
            mermaid += f"    {module_name} --> {class_node}\n"
            
            # Add methods
            for method in cls["methods"][:5]:  # Limit to first 5 methods
                method_node = f"{class_node}_{method['name']}"
                mermaid += f"    {method_node}[{method['name']}()]\n"
                mermaid += f"    {class_node} --> {method_node}\n"
        
        # Add standalone functions
        for func in module_analysis["functions"][:5]:  # Limit to first 5 functions
            func_node = f"{module_name}_{func['name']}"
            mermaid += f"    {func_node}[{func['name']}()]\n"
            mermaid += f"    {module_name} --> {func_node}\n"
        
        # Add styling based on architectural role
        role = module_analysis.get("architectural_role", "Utility")
        if role == "Core Engine":
            mermaid += f"    style {module_name} fill:#ff9999\n"
        elif role == "Cognitive Architecture":
            mermaid += f"    style {module_name} fill:#99ccff\n"
        elif role == "Neural Processing":
            mermaid += f"    style {module_name} fill:#99ff99\n"
        elif role == "Interface Layer":
            mermaid += f"    style {module_name} fill:#ffcc99\n"
        
        mermaid += "```"
        return mermaid
    
    def generate_dependency_graph(self, modules: List[str]) -> str:
        """Generate dependency graph for all modules"""
        mermaid = "```mermaid\ngraph TD\n"
        
        dependencies = {}
        for module in modules:
            analysis = self.analyze_module_structure(module)
            if "error" not in analysis:
                deps = analysis.get("dependencies", set())
                local_deps = [dep for dep in deps if dep in modules]
                dependencies[module] = local_deps
        
        # Add nodes
        for module in modules:
            role = self.analyze_module_structure(module).get("architectural_role", "Utility")
            mermaid += f"    {module}[{module}]\n"
        
        # Add edges
        for module, deps in dependencies.items():
            for dep in deps:
                if dep in modules:
                    mermaid += f"    {dep} --> {module}\n"
        
        # Add styling
        for module in modules:
            analysis = self.analyze_module_structure(module)
            if "error" not in analysis:
                role = analysis.get("architectural_role", "Utility")
                if role == "Core Engine":
                    mermaid += f"    style {module} fill:#ff9999\n"
                elif role == "Cognitive Architecture":
                    mermaid += f"    style {module} fill:#99ccff\n"
                elif role == "Neural Processing":
                    mermaid += f"    style {module} fill:#99ff99\n"
                elif role == "Interface Layer":
                    mermaid += f"    style {module} fill:#ffcc99\n"
        
        mermaid += "```"
        return mermaid
    
    def generate_comprehensive_documentation(self) -> Dict[str, Any]:
        """Generate comprehensive documentation for all modules"""
        modules = self.discover_all_modules()
        documentation = {
            "generated_at": datetime.now().isoformat(),
            "total_modules": len(modules),
            "modules": {},
            "architectural_overview": {},
            "dependency_graph": self.generate_dependency_graph(modules),
            "cognitive_patterns_summary": {},
            "statistics": {}
        }
        
        # Analyze each module
        for module in modules:
            analysis = self.analyze_module_structure(module)
            documentation["modules"][module] = analysis
            if "error" not in analysis:
                # Generate flowchart
                analysis["flowchart"] = self.generate_mermaid_flowchart(analysis)
        
        # Generate architectural overview
        roles = {}
        total_classes = 0
        total_functions = 0
        total_lines = 0
        
        for module, analysis in documentation["modules"].items():
            if "error" not in analysis:
                role = analysis.get("architectural_role", "Utility")
                if role not in roles:
                    roles[role] = []
                roles[role].append(module)
                
                total_classes += len(analysis.get("classes", []))
                total_functions += len(analysis.get("functions", []))
                
        documentation["architectural_overview"] = roles
        
        # Generate statistics
        documentation["statistics"] = {
            "total_classes": total_classes,
            "total_functions": total_functions,
            "architectural_roles": len(roles),
            "modules_by_role": {role: len(modules) for role, modules in roles.items()}
        }
        
        # Summarize cognitive patterns
        all_patterns = {}
        for module, analysis in documentation["modules"].items():
            if "error" not in analysis:
                for pattern in analysis.get("cognitive_patterns", []):
                    if pattern not in all_patterns:
                        all_patterns[pattern] = []
                    all_patterns[pattern].append(module)
        
        documentation["cognitive_patterns_summary"] = all_patterns
        
        return documentation
    
    def save_documentation(self, documentation: Dict[str, Any], output_dir: str = "docs_generated"):
        """Save generated documentation to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save main documentation as JSON
        with open(output_path / "phase6_documentation.json", 'w') as f:
            json.dump(documentation, f, indent=2, default=str)
        
        # Generate markdown documentation
        md_content = self._generate_markdown_documentation(documentation)
        with open(output_path / "PHASE6_GENERATED_DOCUMENTATION.md", 'w') as f:
            f.write(md_content)
        
        # Save individual module flowcharts
        flowcharts_dir = output_path / "flowcharts"
        flowcharts_dir.mkdir(exist_ok=True)
        
        for module, analysis in documentation["modules"].items():
            if "error" not in analysis and "flowchart" in analysis:
                with open(flowcharts_dir / f"{module}_flowchart.md", 'w') as f:
                    f.write(f"# {module} Module Flowchart\n\n")
                    f.write(analysis["flowchart"])
        
        print(f"📚 Documentation saved to {output_path}")
        return output_path
    
    def _generate_markdown_documentation(self, documentation: Dict[str, Any]) -> str:
        """Generate markdown documentation from analysis"""
        md = "# Phase 6: Auto-Generated Comprehensive Documentation\n\n"
        md += f"Generated at: {documentation['generated_at']}\n\n"
        
        # Statistics
        stats = documentation["statistics"]
        md += "## 📊 Project Statistics\n\n"
        md += f"- **Total Modules**: {documentation['total_modules']}\n"
        md += f"- **Total Classes**: {stats['total_classes']}\n"
        md += f"- **Total Functions**: {stats['total_functions']}\n"
        md += f"- **Architectural Roles**: {stats['architectural_roles']}\n\n"
        
        # Architectural Overview
        md += "## 🏗️ Architectural Overview\n\n"
        for role, modules in documentation["architectural_overview"].items():
            md += f"### {role}\n"
            for module in modules:
                md += f"- `{module}`\n"
            md += "\n"
        
        # Dependency Graph
        md += "## 🔗 System Dependency Graph\n\n"
        md += documentation["dependency_graph"] + "\n\n"
        
        # Cognitive Patterns
        md += "## 🧠 Cognitive Patterns Summary\n\n"
        for pattern, modules in documentation["cognitive_patterns_summary"].items():
            md += f"### {pattern}\n"
            md += f"Found in: {', '.join(modules)}\n\n"
        
        # Module Details
        md += "## 📦 Module Details\n\n"
        for module, analysis in documentation["modules"].items():
            if "error" not in analysis:
                md += f"### {module}\n\n"
                if analysis.get("docstring"):
                    md += f"**Description**: {analysis['docstring']}\n\n"
                
                md += f"**Architectural Role**: {analysis.get('architectural_role', 'Unknown')}\n\n"
                
                if analysis.get("cognitive_features"):
                    md += f"**Cognitive Features**: {', '.join(analysis['cognitive_features'])}\n\n"
                
                if analysis.get("classes"):
                    md += "**Classes**:\n"
                    for cls in analysis["classes"]:
                        md += f"- `{cls['name']}` ({len(cls['methods'])} methods)\n"
                    md += "\n"
                
                if analysis.get("functions"):
                    md += "**Functions**:\n"
                    for func in analysis["functions"]:
                        md += f"- `{func['name']}()` (complexity: {func['complexity']})\n"
                    md += "\n"
                
                if analysis.get("flowchart"):
                    md += "**Module Flowchart**:\n\n"
                    md += analysis["flowchart"] + "\n\n"
        
        return md


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 6 Recursive Documentation Generator")
    parser.add_argument("--generate", action="store_true", help="Generate comprehensive documentation")
    parser.add_argument("--module", type=str, help="Analyze specific module")
    parser.add_argument("--output", type=str, default="docs_generated", help="Output directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    generator = RecursiveDocumentationGenerator()
    
    if args.module:
        analysis = generator.analyze_module_structure(args.module)
        if args.json:
            print(json.dumps(analysis, indent=2, default=str))
        else:
            print(f"Analysis for {args.module}:")
            if "error" not in analysis:
                print(f"Role: {analysis.get('architectural_role')}")
                print(f"Classes: {len(analysis.get('classes', []))}")
                print(f"Functions: {len(analysis.get('functions', []))}")
                print(f"Cognitive Features: {analysis.get('cognitive_features', [])}")
                print("\nFlowchart:")
                print(generator.generate_mermaid_flowchart(analysis))
    
    elif args.generate:
        print("📚 Generating comprehensive documentation...")
        documentation = generator.generate_comprehensive_documentation()
        output_path = generator.save_documentation(documentation, args.output)
        
        if args.json:
            print(json.dumps(documentation, indent=2, default=str))
        else:
            print(f"✅ Documentation generated successfully!")
            print(f"📊 Total modules analyzed: {documentation['total_modules']}")
            print(f"🏗️  Architectural roles: {len(documentation['architectural_overview'])}")
            print(f"📁 Output saved to: {output_path}")
    
    else:
        print("Use --generate to create comprehensive documentation or --module <name> to analyze a specific module")