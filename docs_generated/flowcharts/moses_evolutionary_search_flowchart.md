# moses_evolutionary_search Module Flowchart

```mermaid
graph TD
    moses_evolutionary_search[moses_evolutionary_search]
    moses_evolutionary_search_MutationType[MutationType]
    moses_evolutionary_search --> moses_evolutionary_search_MutationType
    moses_evolutionary_search_SelectionMethod[SelectionMethod]
    moses_evolutionary_search --> moses_evolutionary_search_SelectionMethod
    moses_evolutionary_search_CognitivePattern[CognitivePattern]
    moses_evolutionary_search --> moses_evolutionary_search_CognitivePattern
    moses_evolutionary_search_CognitivePattern___post_init__[__post_init__()]
    moses_evolutionary_search_CognitivePattern --> moses_evolutionary_search_CognitivePattern___post_init__
    moses_evolutionary_search_CognitivePattern_copy[copy()]
    moses_evolutionary_search_CognitivePattern --> moses_evolutionary_search_CognitivePattern_copy
    moses_evolutionary_search_CognitivePattern_to_dict[to_dict()]
    moses_evolutionary_search_CognitivePattern --> moses_evolutionary_search_CognitivePattern_to_dict
    moses_evolutionary_search_EvolutionaryParameters[EvolutionaryParameters]
    moses_evolutionary_search --> moses_evolutionary_search_EvolutionaryParameters
    moses_evolutionary_search_FitnessEvaluator[FitnessEvaluator]
    moses_evolutionary_search --> moses_evolutionary_search_FitnessEvaluator
    moses_evolutionary_search_FitnessEvaluator___init__[__init__()]
    moses_evolutionary_search_FitnessEvaluator --> moses_evolutionary_search_FitnessEvaluator___init__
    moses_evolutionary_search_FitnessEvaluator_evaluate_pattern[evaluate_pattern()]
    moses_evolutionary_search_FitnessEvaluator --> moses_evolutionary_search_FitnessEvaluator_evaluate_pattern
    moses_evolutionary_search_FitnessEvaluator__evaluate_semantic_coherence[_evaluate_semantic_coherence()]
    moses_evolutionary_search_FitnessEvaluator --> moses_evolutionary_search_FitnessEvaluator__evaluate_semantic_coherence
    moses_evolutionary_search_FitnessEvaluator__evaluate_attention_efficiency[_evaluate_attention_efficiency()]
    moses_evolutionary_search_FitnessEvaluator --> moses_evolutionary_search_FitnessEvaluator__evaluate_attention_efficiency
    moses_evolutionary_search_FitnessEvaluator__evaluate_structural_complexity[_evaluate_structural_complexity()]
    moses_evolutionary_search_FitnessEvaluator --> moses_evolutionary_search_FitnessEvaluator__evaluate_structural_complexity
    moses_evolutionary_search_MOSESEvolutionarySearch[MOSESEvolutionarySearch]
    moses_evolutionary_search --> moses_evolutionary_search_MOSESEvolutionarySearch
    moses_evolutionary_search_MOSESEvolutionarySearch___init__[__init__()]
    moses_evolutionary_search_MOSESEvolutionarySearch --> moses_evolutionary_search_MOSESEvolutionarySearch___init__
    moses_evolutionary_search_MOSESEvolutionarySearch_initialize_population[initialize_population()]
    moses_evolutionary_search_MOSESEvolutionarySearch --> moses_evolutionary_search_MOSESEvolutionarySearch_initialize_population
    moses_evolutionary_search_MOSESEvolutionarySearch_evolve[evolve()]
    moses_evolutionary_search_MOSESEvolutionarySearch --> moses_evolutionary_search_MOSESEvolutionarySearch_evolve
    moses_evolutionary_search_MOSESEvolutionarySearch__create_random_pattern[_create_random_pattern()]
    moses_evolutionary_search_MOSESEvolutionarySearch --> moses_evolutionary_search_MOSESEvolutionarySearch__create_random_pattern
    moses_evolutionary_search_MOSESEvolutionarySearch__create_random_hypergraph_genes[_create_random_hypergraph_genes()]
    moses_evolutionary_search_MOSESEvolutionarySearch --> moses_evolutionary_search_MOSESEvolutionarySearch__create_random_hypergraph_genes
    style moses_evolutionary_search fill:#99ccff
```