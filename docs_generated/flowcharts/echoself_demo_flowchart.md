# echoself_demo Module Flowchart

```mermaid
graph TD
    echoself_demo[echoself_demo]
    echoself_demo_setup_logging[setup_logging()]
    echoself_demo --> echoself_demo_setup_logging
    echoself_demo_demonstrate_introspection_cycle[demonstrate_introspection_cycle()]
    echoself_demo --> echoself_demo_demonstrate_introspection_cycle
    echoself_demo_demonstrate_adaptive_attention[demonstrate_adaptive_attention()]
    echoself_demo --> echoself_demo_demonstrate_adaptive_attention
    echoself_demo_demonstrate_hypergraph_export[demonstrate_hypergraph_export()]
    echoself_demo --> echoself_demo_demonstrate_hypergraph_export
    echoself_demo_demonstrate_neural_symbolic_synergy[demonstrate_neural_symbolic_synergy()]
    echoself_demo --> echoself_demo_demonstrate_neural_symbolic_synergy
    style echoself_demo fill:#99ccff
```