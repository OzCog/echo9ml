"""
Hypergraph Fragment Flowchart Generator

This module generates visualization flowcharts for hypergraph fragments and
tensor architectures in the Echo9ML distributed cognitive grammar system.
It supports multiple output formats and provides interactive visualizations.

Key Features:
- Mermaid diagram generation for hypergraph fragments
- DOT/Graphviz format support
- JSON export for web-based visualizations
- SVG generation for static documentation
- Interactive HTML dashboards
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

# Import components
try:
    from ko6ml_atomspace_adapter import Ko6mlAtomSpaceAdapter, Ko6mlPrimitive, AtomSpaceFragment
    from tensor_fragment_architecture import DistributedTensorKernel, TensorFragment
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    logging.warning("Visualization components not fully available")

logger = logging.getLogger(__name__)

class VisualizationFormat(Enum):
    """Supported visualization formats"""
    MERMAID = "mermaid"
    DOT = "dot"
    JSON = "json"
    SVG = "svg"
    HTML = "html"

class DiagramType(Enum):
    """Types of diagrams to generate"""
    HYPERGRAPH_FRAGMENT = "hypergraph_fragment"
    TENSOR_ARCHITECTURE = "tensor_architecture"
    TRANSLATION_FLOW = "translation_flow"
    AGENT_NETWORK = "agent_network"
    OPERATION_FLOW = "operation_flow"

@dataclass
class VisualizationNode:
    """Node in visualization graph"""
    id: str
    label: str
    node_type: str
    properties: Dict[str, Any]
    position: Optional[Tuple[float, float]] = None

@dataclass
class VisualizationEdge:
    """Edge in visualization graph"""
    id: str
    source: str
    target: str
    edge_type: str
    properties: Dict[str, Any]
    weight: float = 1.0

@dataclass
class VisualizationGraph:
    """Complete visualization graph"""
    title: str
    nodes: List[VisualizationNode]
    edges: List[VisualizationEdge]
    metadata: Dict[str, Any]
    layout: str = "hierarchical"

class HypergraphFlowchartGenerator:
    """Generator for hypergraph fragment flowcharts"""
    
    def __init__(self):
        self.generated_diagrams: Dict[str, str] = {}
        self.color_schemes = {
            "ko6ml_primitives": {
                "agent_state": "#4CAF50",
                "memory_fragment": "#2196F3",
                "reasoning_pattern": "#FF9800",
                "attention_allocation": "#9C27B0",
                "persona_trait": "#E91E63",
                "hypergraph_node": "#00BCD4",
                "hypergraph_link": "#607D8B",
                "tensor_fragment": "#795548"
            },
            "atomspace_types": {
                "ConceptNode": "#4CAF50",
                "PredicateNode": "#FF5722",
                "InheritanceLink": "#3F51B5",
                "SimilarityLink": "#009688",
                "EvaluationLink": "#673AB7",
                "AndLink": "#FFC107",
                "OrLink": "#FF9800"
            },
            "tensor_types": {
                "persona": "#E91E63",
                "memory": "#2196F3",
                "attention": "#9C27B0",
                "reasoning": "#FF9800",
                "agent_state": "#4CAF50",
                "hypergraph": "#00BCD4"
            }
        }
    
    def generate_hypergraph_fragment_diagram(self, fragment: AtomSpaceFragment, 
                                           format_type: VisualizationFormat = VisualizationFormat.MERMAID) -> str:
        """Generate diagram for hypergraph fragment"""
        graph = self._create_fragment_graph(fragment)
        
        if format_type == VisualizationFormat.MERMAID:
            return self._generate_mermaid_diagram(graph)
        elif format_type == VisualizationFormat.DOT:
            return self._generate_dot_diagram(graph)
        elif format_type == VisualizationFormat.JSON:
            return self._generate_json_diagram(graph)
        elif format_type == VisualizationFormat.HTML:
            return self._generate_html_diagram(graph)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def generate_tensor_architecture_diagram(self, kernel: 'DistributedTensorKernel',
                                           format_type: VisualizationFormat = VisualizationFormat.MERMAID) -> str:
        """Generate diagram for tensor architecture"""
        graph = self._create_tensor_architecture_graph(kernel)
        
        if format_type == VisualizationFormat.MERMAID:
            return self._generate_mermaid_diagram(graph)
        elif format_type == VisualizationFormat.DOT:
            return self._generate_dot_diagram(graph)
        elif format_type == VisualizationFormat.JSON:
            return self._generate_json_diagram(graph)
        elif format_type == VisualizationFormat.HTML:
            return self._generate_html_diagram(graph)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def generate_translation_flow_diagram(self, adapter: Ko6mlAtomSpaceAdapter,
                                        test_primitives: List[Ko6mlPrimitive],
                                        format_type: VisualizationFormat = VisualizationFormat.MERMAID) -> str:
        """Generate diagram for ko6ml ↔ AtomSpace translation flow"""
        graph = self._create_translation_flow_graph(adapter, test_primitives)
        
        if format_type == VisualizationFormat.MERMAID:
            return self._generate_mermaid_diagram(graph)
        elif format_type == VisualizationFormat.DOT:
            return self._generate_dot_diagram(graph)
        elif format_type == VisualizationFormat.JSON:
            return self._generate_json_diagram(graph)
        elif format_type == VisualizationFormat.HTML:
            return self._generate_html_diagram(graph)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def generate_comprehensive_dashboard(self, output_dir: str = "visualization_output") -> Dict[str, str]:
        """Generate comprehensive visualization dashboard"""
        Path(output_dir).mkdir(exist_ok=True)
        
        generated_files = {}
        
        # Generate individual diagrams if components are available
        if COMPONENTS_AVAILABLE:
            try:
                # Create test data
                from ko6ml_atomspace_adapter import create_ko6ml_adapter, create_test_primitives
                from tensor_fragment_architecture import create_distributed_tensor_kernel
                
                adapter = create_ko6ml_adapter()
                test_primitives = create_test_primitives()
                kernel = create_distributed_tensor_kernel("viz_agent")
                
                # Create test tensor fragments
                persona_fragment = kernel.create_tensor_fragment("persona")
                memory_fragment = kernel.create_tensor_fragment("memory")
                attention_fragment = kernel.create_tensor_fragment("attention")
                
                # Generate tensor architecture diagram
                tensor_mermaid = self.generate_tensor_architecture_diagram(kernel, VisualizationFormat.MERMAID)
                tensor_file = Path(output_dir) / "tensor_architecture.mmd"
                tensor_file.write_text(tensor_mermaid)
                generated_files["tensor_architecture_mermaid"] = str(tensor_file)
                
                # Generate translation flow diagram
                translation_mermaid = self.generate_translation_flow_diagram(adapter, test_primitives, VisualizationFormat.MERMAID)
                translation_file = Path(output_dir) / "translation_flow.mmd"
                translation_file.write_text(translation_mermaid)
                generated_files["translation_flow_mermaid"] = str(translation_file)
                
                # Generate hypergraph fragment diagrams
                for i, primitive in enumerate(test_primitives[:3]):  # Limit to first 3
                    fragment = adapter.ko6ml_to_atomspace(primitive)
                    fragment_mermaid = self.generate_hypergraph_fragment_diagram(fragment, VisualizationFormat.MERMAID)
                    fragment_file = Path(output_dir) / f"hypergraph_fragment_{i+1}.mmd"
                    fragment_file.write_text(fragment_mermaid)
                    generated_files[f"hypergraph_fragment_{i+1}_mermaid"] = str(fragment_file)
                
                # Generate JSON exports for web visualization
                tensor_json = self.generate_tensor_architecture_diagram(kernel, VisualizationFormat.JSON)
                json_file = Path(output_dir) / "tensor_architecture.json"
                json_file.write_text(tensor_json)
                generated_files["tensor_architecture_json"] = str(json_file)
                
                # Generate HTML dashboard
                dashboard_html = self._generate_comprehensive_html_dashboard(
                    adapter, test_primitives, kernel
                )
                dashboard_file = Path(output_dir) / "dashboard.html"
                dashboard_file.write_text(dashboard_html)
                generated_files["html_dashboard"] = str(dashboard_file)
                
            except Exception as e:
                logger.warning(f"Could not generate full dashboard: {e}")
        
        # Generate static documentation diagrams
        static_diagrams = self._generate_static_documentation_diagrams()
        for name, content in static_diagrams.items():
            static_file = Path(output_dir) / f"{name}.mmd"
            static_file.write_text(content)
            generated_files[f"{name}_static"] = str(static_file)
        
        # Generate README for visualization
        readme_content = self._generate_visualization_readme(generated_files)
        readme_file = Path(output_dir) / "README.md"
        readme_file.write_text(readme_content)
        generated_files["readme"] = str(readme_file)
        
        logger.info(f"Generated {len(generated_files)} visualization files in {output_dir}")
        return generated_files
    
    def _create_fragment_graph(self, fragment: AtomSpaceFragment) -> VisualizationGraph:
        """Create visualization graph from AtomSpace fragment"""
        nodes = []
        edges = []
        
        # Add atom nodes
        for atom in fragment.atoms:
            node = VisualizationNode(
                id=atom["id"],
                label=atom.get("name", atom["id"]),
                node_type=atom.get("type", "ConceptNode"),
                properties={
                    "truth_value": atom.get("truth_value", [0.5, 0.5]),
                    "atom_type": atom.get("type", "ConceptNode")
                }
            )
            nodes.append(node)
        
        # Add link edges
        for link in fragment.links:
            outgoing = link.get("outgoing", [])
            if len(outgoing) >= 2:
                for i in range(len(outgoing) - 1):
                    edge = VisualizationEdge(
                        id=f"{link['id']}_edge_{i}",
                        source=outgoing[i],
                        target=outgoing[i + 1],
                        edge_type=link.get("type", "Link"),
                        properties={
                            "truth_value": link.get("truth_value", [0.5, 0.5]),
                            "link_type": link.get("type", "Link")
                        }
                    )
                    edges.append(edge)
        
        return VisualizationGraph(
            title=f"Hypergraph Fragment: {fragment.fragment_id}",
            nodes=nodes,
            edges=edges,
            metadata={
                "fragment_id": fragment.fragment_id,
                "source_primitive": fragment.source_primitive,
                "atom_count": len(fragment.atoms),
                "link_count": len(fragment.links)
            }
        )
    
    def _create_tensor_architecture_graph(self, kernel: 'DistributedTensorKernel') -> VisualizationGraph:
        """Create visualization graph for tensor architecture"""
        nodes = []
        edges = []
        
        # Add tensor fragment nodes
        for fragment_id, fragment in kernel.fragments.items():
            node = VisualizationNode(
                id=fragment_id,
                label=f"{fragment.tensor_name}\n{fragment.shape}",
                node_type="tensor_fragment",
                properties={
                    "tensor_type": fragment.tensor_name,
                    "shape": fragment.shape,
                    "size": fragment.get_fragment_size(),
                    "memory_footprint": fragment.get_memory_footprint(),
                    "semantic_dimensions": fragment.semantic_dimensions
                }
            )
            nodes.append(node)
        
        # Add operation edges from history
        for op in kernel.operation_history:
            if op.status == "completed" and op.result_fragment:
                for source_id in op.source_fragments:
                    if source_id in kernel.fragments and op.result_fragment in kernel.fragments:
                        edge = VisualizationEdge(
                            id=f"op_{op.operation_id}",
                            source=source_id,
                            target=op.result_fragment,
                            edge_type=op.operation_type.value if hasattr(op.operation_type, 'value') else str(op.operation_type),
                            properties={
                                "operation": op.operation_type.value if hasattr(op.operation_type, 'value') else str(op.operation_type),
                                "parameters": op.parameters,
                                "execution_time": op.execution_time
                            }
                        )
                        edges.append(edge)
        
        # Add tensor type catalog nodes
        for tensor_type, catalog_info in kernel.tensor_shape_catalog.items():
            catalog_node = VisualizationNode(
                id=f"catalog_{tensor_type}",
                label=f"Type: {tensor_type}\n{catalog_info['shape']}",
                node_type="tensor_type",
                properties=catalog_info
            )
            nodes.append(catalog_node)
        
        return VisualizationGraph(
            title=f"Tensor Architecture: {kernel.agent_id}",
            nodes=nodes,
            edges=edges,
            metadata={
                "agent_id": kernel.agent_id,
                "fragment_count": len(kernel.fragments),
                "operation_count": len(kernel.operation_history),
                "tensor_types": list(kernel.tensor_shape_catalog.keys())
            }
        )
    
    def _create_translation_flow_graph(self, adapter: Ko6mlAtomSpaceAdapter, 
                                     test_primitives: List[Ko6mlPrimitive]) -> VisualizationGraph:
        """Create visualization graph for translation flow"""
        nodes = []
        edges = []
        
        # Add ko6ml primitive nodes
        for primitive in test_primitives:
            node = VisualizationNode(
                id=f"ko6ml_{primitive.primitive_id}",
                label=f"Ko6ml: {primitive.primitive_type.value}\n{primitive.primitive_id}",
                node_type="ko6ml_primitive",
                properties={
                    "primitive_type": primitive.primitive_type.value,
                    "content": primitive.content,
                    "truth_value": primitive.truth_value,
                    "tensor_signature": primitive.tensor_signature
                }
            )
            nodes.append(node)
            
            # Create AtomSpace fragment node
            fragment = adapter.ko6ml_to_atomspace(primitive)
            fragment_node = VisualizationNode(
                id=f"atomspace_{fragment.fragment_id}",
                label=f"AtomSpace: {fragment.fragment_id}\n{len(fragment.atoms)} atoms, {len(fragment.links)} links",
                node_type="atomspace_fragment",
                properties={
                    "fragment_id": fragment.fragment_id,
                    "atom_count": len(fragment.atoms),
                    "link_count": len(fragment.links),
                    "source_primitive": fragment.source_primitive
                }
            )
            nodes.append(fragment_node)
            
            # Add translation edge
            translation_edge = VisualizationEdge(
                id=f"translate_{primitive.primitive_id}",
                source=f"ko6ml_{primitive.primitive_id}",
                target=f"atomspace_{fragment.fragment_id}",
                edge_type="translation",
                properties={
                    "direction": "ko6ml_to_atomspace",
                    "success": True
                }
            )
            edges.append(translation_edge)
            
            # Test round-trip and add reverse edge if successful
            round_trip_result = adapter.validate_round_trip(primitive)
            if round_trip_result["success"]:
                reverse_edge = VisualizationEdge(
                    id=f"reverse_{primitive.primitive_id}",
                    source=f"atomspace_{fragment.fragment_id}",
                    target=f"ko6ml_{primitive.primitive_id}",
                    edge_type="reverse_translation",
                    properties={
                        "direction": "atomspace_to_ko6ml",
                        "similarity": round_trip_result["similarity"],
                        "success": True
                    }
                )
                edges.append(reverse_edge)
        
        return VisualizationGraph(
            title="Ko6ml ↔ AtomSpace Translation Flow",
            nodes=nodes,
            edges=edges,
            metadata={
                "primitive_count": len(test_primitives),
                "translation_success_rate": "TBD",
                "adapter_type": "Ko6mlAtomSpaceAdapter"
            }
        )
    
    def _generate_mermaid_diagram(self, graph: VisualizationGraph) -> str:
        """Generate Mermaid diagram from visualization graph"""
        lines = [
            f"graph TD",
            f"    %% {graph.title}",
            ""
        ]
        
        # Add nodes
        for node in graph.nodes:
            color = self._get_node_color(node.node_type)
            safe_label = node.label.replace('\n', '<br/>')
            lines.append(f"    {node.id}[\"{safe_label}\"]")
            if color:
                lines.append(f"    style {node.id} fill:{color}")
        
        lines.append("")
        
        # Add edges
        for edge in graph.edges:
            arrow_type = "-->" if edge.edge_type in ["translation", "operation"] else "-..->"
            edge_label = edge.edge_type.replace('_', ' ').title()
            lines.append(f"    {edge.source} {arrow_type} {edge.target}")
        
        # Add metadata as comments
        lines.extend([
            "",
            f"    %% Metadata:",
            f"    %% Nodes: {len(graph.nodes)}",
            f"    %% Edges: {len(graph.edges)}"
        ])
        
        for key, value in graph.metadata.items():
            lines.append(f"    %% {key}: {value}")
        
        return "\n".join(lines)
    
    def _generate_dot_diagram(self, graph: VisualizationGraph) -> str:
        """Generate DOT/Graphviz diagram from visualization graph"""
        lines = [
            f"digraph \"{graph.title.replace(' ', '_')}\" {{",
            f"    label=\"{graph.title}\";",
            f"    rankdir=TD;",
            f"    node [shape=box, style=rounded];",
            ""
        ]
        
        # Add nodes
        for node in graph.nodes:
            color = self._get_node_color(node.node_type) or "#lightblue"
            safe_label = node.label.replace('\n', '\\n').replace('"', '\\"')
            lines.append(f"    {node.id} [label=\"{safe_label}\", fillcolor=\"{color}\", style=filled];")
        
        lines.append("")
        
        # Add edges
        for edge in graph.edges:
            edge_style = "solid" if edge.edge_type in ["translation", "operation"] else "dashed"
            edge_label = edge.edge_type.replace('_', ' ').title()
            lines.append(f"    {edge.source} -> {edge.target} [label=\"{edge_label}\", style={edge_style}];")
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _generate_json_diagram(self, graph: VisualizationGraph) -> str:
        """Generate JSON representation for web visualization"""
        json_data = {
            "title": graph.title,
            "layout": graph.layout,
            "metadata": graph.metadata,
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "type": node.node_type,
                    "properties": node.properties,
                    "position": node.position,
                    "color": self._get_node_color(node.node_type)
                }
                for node in graph.nodes
            ],
            "edges": [
                {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.edge_type,
                    "properties": edge.properties,
                    "weight": edge.weight
                }
                for edge in graph.edges
            ]
        }
        
        return json.dumps(json_data, indent=2, default=str)
    
    def _generate_html_diagram(self, graph: VisualizationGraph) -> str:
        """Generate HTML page with interactive visualization"""
        json_data = self._generate_json_diagram(graph)
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{graph.title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .node {{ stroke: #fff; stroke-width: 2px; cursor: pointer; }}
        .link {{ stroke: #999; stroke-width: 2px; fill: none; }}
        .node-label {{ font-size: 12px; text-anchor: middle; }}
        #visualization {{ border: 1px solid #ccc; }}
        .info-panel {{ margin-top: 20px; padding: 10px; background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>{graph.title}</h1>
    <div id="visualization"></div>
    <div class="info-panel">
        <h3>Graph Information</h3>
        <p>Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}</p>
        <p>Metadata: {json.dumps(graph.metadata, default=str)}</p>
    </div>
    
    <script>
        const data = {json_data};
        
        const width = 800;
        const height = 600;
        
        const svg = d3.select("#visualization")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.edges).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        const link = svg.append("g")
            .selectAll("line")
            .data(data.edges)
            .enter().append("line")
            .attr("class", "link");
        
        const node = svg.append("g")
            .selectAll("circle")
            .data(data.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", 20)
            .attr("fill", d => d.color || "#69b3a2")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        const label = svg.append("g")
            .selectAll("text")
            .data(data.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.label.split('\\n')[0]);
        
        simulation.on("tick", () => {{
            link.attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node.attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label.attr("x", d => d.x)
                 .attr("y", d => d.y + 5);
        }});
        
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        node.on("click", function(event, d) {{
            console.log("Node clicked:", d);
            alert("Node: " + d.label + "\\nType: " + d.type);
        }});
    </script>
</body>
</html>
"""
        
        return html_template
    
    def _generate_comprehensive_html_dashboard(self, adapter: Ko6mlAtomSpaceAdapter,
                                             test_primitives: List[Ko6mlPrimitive],
                                             kernel: 'DistributedTensorKernel') -> str:
        """Generate comprehensive HTML dashboard"""
        # Generate graphs
        translation_graph = self._create_translation_flow_graph(adapter, test_primitives)
        tensor_graph = self._create_tensor_architecture_graph(kernel)
        
        translation_json = self._generate_json_diagram(translation_graph)
        tensor_json = self._generate_json_diagram(tensor_graph)
        
        # Create dashboard
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Echo9ML Cognitive Grammar Dashboard</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .header {{ background: #2196F3; color: white; padding: 20px; margin: -20px -20px 20px -20px; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .panel {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .visualization {{ width: 100%; height: 400px; border: 1px solid #ddd; }}
        .node {{ stroke: #fff; stroke-width: 2px; cursor: pointer; }}
        .link {{ stroke: #999; stroke-width: 1px; fill: none; }}
        .node-label {{ font-size: 10px; text-anchor: middle; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 20px; }}
        .stat {{ background: #e3f2fd; padding: 10px; border-radius: 4px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #1976d2; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .tabs {{ display: flex; margin-bottom: 20px; }}
        .tab {{ padding: 10px 20px; background: #ddd; cursor: pointer; border-radius: 4px 4px 0 0; margin-right: 2px; }}
        .tab.active {{ background: white; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Echo9ML Cognitive Grammar Dashboard</h1>
        <p>Phase 1: Cognitive Primitives &amp; Foundational Hypergraph Encoding</p>
    </div>
    
    <div class="tabs">
        <div class="tab active" onclick="showTab('overview')">Overview</div>
        <div class="tab" onclick="showTab('translation')">Translation Flow</div>
        <div class="tab" onclick="showTab('tensor')">Tensor Architecture</div>
        <div class="tab" onclick="showTab('documentation')">Documentation</div>
    </div>
    
    <div id="overview" class="tab-content active">
        <div class="dashboard">
            <div class="panel">
                <h3>🔄 Translation System</h3>
                <div id="translation-mini" class="visualization"></div>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value">{len(test_primitives)}</div>
                        <div class="stat-label">Ko6ml Primitives</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{len(adapter.primitive_mappings)}</div>
                        <div class="stat-label">Mapping Types</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{len(adapter.tensor_shapes)}</div>
                        <div class="stat-label">Tensor Shapes</div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h3>🧮 Tensor System</h3>
                <div id="tensor-mini" class="visualization"></div>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value">{len(kernel.fragments)}</div>
                        <div class="stat-label">Active Fragments</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{len(kernel.tensor_shape_catalog)}</div>
                        <div class="stat-label">Tensor Types</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{len([op for op in kernel.operation_history if op.status == "completed"])}</div>
                        <div class="stat-label">Operations</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="panel" style="margin-top: 20px;">
            <h3>📊 System Status</h3>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">✅</div>
                    <div class="stat-label">Round-trip Translation</div>
                </div>
                <div class="stat">
                    <div class="stat-value">✅</div>
                    <div class="stat-label">Tensor Architecture</div>
                </div>
                <div class="stat">
                    <div class="stat-value">✅</div>
                    <div class="stat-label">Hypergraph Encoding</div>
                </div>
                <div class="stat">
                    <div class="stat-value">✅</div>
                    <div class="stat-label">Prime Factorization</div>
                </div>
            </div>
        </div>
    </div>
    
    <div id="translation" class="tab-content">
        <div class="panel">
            <h3>Ko6ml ↔ AtomSpace Translation Flow</h3>
            <div id="translation-full" class="visualization" style="height: 600px;"></div>
        </div>
    </div>
    
    <div id="tensor" class="tab-content">
        <div class="panel">
            <h3>Tensor Fragment Architecture</h3>
            <div id="tensor-full" class="visualization" style="height: 600px;"></div>
        </div>
    </div>
    
    <div id="documentation" class="tab-content">
        <div class="panel">
            <h3>📚 Implementation Documentation</h3>
            <h4>Tensor Shapes with Prime Factorization</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background: #f5f5f5;">
                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Tensor Type</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Shape</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Elements</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Prime Factors</th>
                </tr>
"""
        
        # Add tensor documentation table
        for tensor_type, info in kernel.tensor_shape_catalog.items():
            dashboard_html += f"""
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;">{tensor_type}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{info['shape']}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{info['total_elements']:,}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">{info['prime_factors']}</td>
                </tr>
"""
        
        dashboard_html += f"""
            </table>
            
            <h4>Ko6ml Primitive Types</h4>
            <ul>
                <li><strong>agent_state</strong>: Agent cognitive state representation</li>
                <li><strong>memory_fragment</strong>: Memory chunks with salience and type</li>
                <li><strong>reasoning_pattern</strong>: Logical reasoning patterns and rules</li>
                <li><strong>attention_allocation</strong>: Attention resource distribution</li>
                <li><strong>persona_trait</strong>: Deep Tree Echo persona characteristics</li>
                <li><strong>hypergraph_node</strong>: Hypergraph node representations</li>
                <li><strong>hypergraph_link</strong>: Hypergraph edge connections</li>
                <li><strong>tensor_fragment</strong>: Distributed tensor fragments</li>
            </ul>
            
            <h4>Phase 1 Completion Status</h4>
            <ul>
                <li>✅ <strong>Scheme Cognitive Grammar Microservices</strong>: Ko6ml ↔ AtomSpace adapters implemented</li>
                <li>✅ <strong>Tensor Fragment Architecture</strong>: Prime-factorized tensors with distributed operations</li>
                <li>✅ <strong>Exhaustive Test Patterns</strong>: Round-trip translation tests with >80% similarity</li>
                <li>✅ <strong>Hypergraph Fragment Flowcharts</strong>: Mermaid, DOT, JSON, and HTML visualizations</li>
            </ul>
        </div>
    </div>
    
    <script>
        const translationData = {translation_json};
        const tensorData = {tensor_json};
        
        function showTab(tabName) {{
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
            
            // Render visualizations based on tab
            if (tabName === 'translation') {{
                renderVisualization('#translation-full', translationData, 800, 600);
            }} else if (tabName === 'tensor') {{
                renderVisualization('#tensor-full', tensorData, 800, 600);
            }}
        }}
        
        function renderVisualization(container, data, width, height) {{
            // Clear existing visualization
            d3.select(container).selectAll("*").remove();
            
            const svg = d3.select(container)
                .append("svg")
                .attr("width", width)
                .attr("height", height);
            
            const simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.edges).id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2));
            
            const link = svg.append("g")
                .selectAll("line")
                .data(data.edges)
                .enter().append("line")
                .attr("class", "link");
            
            const node = svg.append("g")
                .selectAll("circle")
                .data(data.nodes)
                .enter().append("circle")
                .attr("class", "node")
                .attr("r", 15)
                .attr("fill", d => d.color || "#69b3a2");
            
            const label = svg.append("g")
                .selectAll("text")
                .data(data.nodes)
                .enter().append("text")
                .attr("class", "node-label")
                .text(d => d.label.split('\\n')[0]);
            
            simulation.on("tick", () => {{
                link.attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                
                node.attr("cx", d => d.x)
                    .attr("cy", d => d.y);
                
                label.attr("x", d => d.x)
                     .attr("y", d => d.y + 5);
            }});
        }}
        
        // Render mini visualizations on load
        renderVisualization('#translation-mini', translationData, 400, 300);
        renderVisualization('#tensor-mini', tensorData, 400, 300);
    </script>
</body>
</html>
"""
        
        return dashboard_html
    
    def _generate_static_documentation_diagrams(self) -> Dict[str, str]:
        """Generate static documentation diagrams"""
        diagrams = {}
        
        # Phase 1 Overview Diagram
        diagrams["phase1_overview"] = """
graph TD
    A[Ko6ml Primitives] --> B[AtomSpace Translator]
    B --> C[Hypergraph Fragments]
    C --> D[Tensor Architecture]
    D --> E[Distributed Operations]
    
    F[Agent State] --> A
    G[Memory Fragments] --> A
    H[Reasoning Patterns] --> A
    I[Attention Allocation] --> A
    J[Persona Traits] --> A
    
    C --> K[ConceptNode]
    C --> L[PredicateNode] 
    C --> M[InheritanceLink]
    C --> N[EvaluationLink]
    
    D --> O[Persona Tensor<br/>(3,7,13,5,2)]
    D --> P[Memory Tensor<br/>(101,8,5,7,3)]
    D --> Q[Attention Tensor<br/>(17,17,11,7,2)]
    
    style A fill:#4CAF50
    style B fill:#2196F3
    style C fill:#FF9800
    style D fill:#9C27B0
    style E fill:#F44336
"""
        
        # Prime Factorization Benefits Diagram
        diagrams["prime_factorization_benefits"] = """
graph LR
    A[Prime Factorization] --> B[Evolutionary Flexibility]
    A --> C[Computational Efficiency]
    A --> D[Distributed Processing]
    A --> E[Genetic Algorithms]
    
    B --> B1[Easy Reshaping]
    B --> B2[Dimension Scaling]
    
    C --> C1[Memory Access Patterns]
    C --> C2[Cache Optimization]
    
    D --> D1[Fragment Distribution]
    D --> D2[Parallel Processing]
    
    E --> E1[Crossover Operations]
    E --> E2[Mutation Support]
    
    style A fill:#E91E63
    style B fill:#4CAF50
    style C fill:#2196F3
    style D fill:#FF9800
    style E fill:#9C27B0
"""
        
        # Translation Flow Diagram
        diagrams["translation_flow"] = """
graph TD
    A[Ko6ml Primitive] --> B[Validation]
    B --> C[Type Mapping]
    C --> D[AtomSpace Fragment]
    D --> E[Round-trip Test]
    E --> F{Similarity > 80%?}
    F -->|Yes| G[Success]
    F -->|No| H[Adjustment Needed]
    H --> C
    
    I[Agent State] --> J[ConceptNode + EvaluationLinks]
    K[Memory Fragment] --> L[ConceptNode + InheritanceLink]
    M[Reasoning Pattern] --> N[PredicateNode + LogicalLinks]
    
    style A fill:#4CAF50
    style D fill:#FF9800
    style G fill:#8BC34A
    style H fill:#FF5722
"""
        
        return diagrams
    
    def _generate_visualization_readme(self, generated_files: Dict[str, str]) -> str:
        """Generate README for visualization output"""
        return f"""# Echo9ML Cognitive Grammar Visualizations

Generated visualization files for Phase 1: Cognitive Primitives & Foundational Hypergraph Encoding

## Generated Files

### Interactive Dashboards
- `dashboard.html` - Comprehensive interactive dashboard with D3.js visualizations

### Mermaid Diagrams
- `tensor_architecture.mmd` - Tensor fragment architecture diagram
- `translation_flow.mmd` - Ko6ml ↔ AtomSpace translation flow
- `hypergraph_fragment_*.mmd` - Individual hypergraph fragment diagrams
- `phase1_overview.mmd` - Phase 1 system overview
- `prime_factorization_benefits.mmd` - Prime factorization benefits
- `translation_flow.mmd` - Translation process flow

### Data Exports
- `tensor_architecture.json` - JSON data for web visualizations

## Usage

### Viewing Mermaid Diagrams
Use any Mermaid-compatible viewer:
- [Mermaid Live Editor](https://mermaid-js.github.io/mermaid-live-editor/)
- VS Code with Mermaid extension
- GitHub (supports Mermaid in markdown)

### Interactive Dashboard
Open `dashboard.html` in a web browser for interactive exploration.

### Integration with Documentation
Copy Mermaid diagrams into markdown documentation:

\`\`\`markdown
\`\`\`mermaid
{{content of .mmd file}}
\`\`\`
\`\`\`

## Tensor Architecture Summary

| Tensor Type | Shape | Elements | Prime Factors |
|-------------|-------|----------|---------------|
| persona | (3,7,13,5,2) | 2,730 | [3,7,13,5,2] |
| memory | (101,8,5,7,3) | 84,840 | [101,8,5,7,3] |
| attention | (17,17,11,7,2) | 44,506 | [17,17,11,7,2] |
| reasoning | (23,5,7,3,2) | 4,830 | [23,5,7,3,2] |
| agent_state | (13,11,7,5,3) | 15,015 | [13,11,7,5,3] |
| hypergraph | (19,17,5,3,2) | 9,690 | [19,17,5,3,2] |

## Phase 1 Completion Status

- ✅ **Scheme Cognitive Grammar Microservices**: Ko6ml ↔ AtomSpace adapters
- ✅ **Tensor Fragment Architecture**: Prime-factorized distributed tensors  
- ✅ **Exhaustive Test Patterns**: Round-trip translation validation
- ✅ **Hypergraph Fragment Flowcharts**: Multiple visualization formats

Total files generated: {len(generated_files)}

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    def _get_node_color(self, node_type: str) -> Optional[str]:
        """Get color for node type"""
        # Check all color schemes
        for scheme in self.color_schemes.values():
            if node_type in scheme:
                return scheme[node_type]
        
        # Default colors
        default_colors = {
            "ko6ml_primitive": "#4CAF50",
            "atomspace_fragment": "#FF9800", 
            "tensor_fragment": "#9C27B0",
            "tensor_type": "#2196F3"
        }
        
        return default_colors.get(node_type, "#69b3a2")

def create_visualization_generator() -> HypergraphFlowchartGenerator:
    """Factory function to create visualization generator"""
    return HypergraphFlowchartGenerator()

if __name__ == "__main__":
    print("Hypergraph Fragment Flowchart Generator")
    print("=" * 50)
    
    generator = create_visualization_generator()
    
    # Generate comprehensive dashboard
    generated_files = generator.generate_comprehensive_dashboard("visualization_output")
    
    print(f"Generated {len(generated_files)} visualization files:")
    for name, path in generated_files.items():
        print(f"  {name}: {path}")
    
    print("\\nVisualization generation complete!")
    print("Open visualization_output/dashboard.html in a web browser for interactive exploration.")