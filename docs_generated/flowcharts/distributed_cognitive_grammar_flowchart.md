# distributed_cognitive_grammar Module Flowchart

```mermaid
graph TD
    distributed_cognitive_grammar[distributed_cognitive_grammar]
    distributed_cognitive_grammar_AgentType[AgentType]
    distributed_cognitive_grammar --> distributed_cognitive_grammar_AgentType
    distributed_cognitive_grammar_MessageType[MessageType]
    distributed_cognitive_grammar --> distributed_cognitive_grammar_MessageType
    distributed_cognitive_grammar_HypergraphFragment[HypergraphFragment]
    distributed_cognitive_grammar --> distributed_cognitive_grammar_HypergraphFragment
    distributed_cognitive_grammar_TensorShape[TensorShape]
    distributed_cognitive_grammar --> distributed_cognitive_grammar_TensorShape
    distributed_cognitive_grammar_TensorShape___post_init__[__post_init__()]
    distributed_cognitive_grammar_TensorShape --> distributed_cognitive_grammar_TensorShape___post_init__
    distributed_cognitive_grammar_CognitiveMessage[CognitiveMessage]
    distributed_cognitive_grammar --> distributed_cognitive_grammar_CognitiveMessage
    distributed_cognitive_grammar_DistributedCognitiveAgent[DistributedCognitiveAgent]
    distributed_cognitive_grammar --> distributed_cognitive_grammar_DistributedCognitiveAgent
    distributed_cognitive_grammar_DistributedCognitiveAgent___init__[__init__()]
    distributed_cognitive_grammar_DistributedCognitiveAgent --> distributed_cognitive_grammar_DistributedCognitiveAgent___init__
    distributed_cognitive_grammar_DistributedCognitiveAgent__initialize_tensor_shapes[_initialize_tensor_shapes()]
    distributed_cognitive_grammar_DistributedCognitiveAgent --> distributed_cognitive_grammar_DistributedCognitiveAgent__initialize_tensor_shapes
    distributed_cognitive_grammar_Echo9MLNode[Echo9MLNode]
    distributed_cognitive_grammar --> distributed_cognitive_grammar_Echo9MLNode
    distributed_cognitive_grammar_Echo9MLNode___init__[__init__()]
    distributed_cognitive_grammar_Echo9MLNode --> distributed_cognitive_grammar_Echo9MLNode___init__
    distributed_cognitive_grammar_DistributedCognitiveNetwork[DistributedCognitiveNetwork]
    distributed_cognitive_grammar --> distributed_cognitive_grammar_DistributedCognitiveNetwork
    distributed_cognitive_grammar_DistributedCognitiveNetwork___init__[__init__()]
    distributed_cognitive_grammar_DistributedCognitiveNetwork --> distributed_cognitive_grammar_DistributedCognitiveNetwork___init__
    distributed_cognitive_grammar_DistributedCognitiveNetwork_add_agent[add_agent()]
    distributed_cognitive_grammar_DistributedCognitiveNetwork --> distributed_cognitive_grammar_DistributedCognitiveNetwork_add_agent
    style distributed_cognitive_grammar fill:#99ccff
```