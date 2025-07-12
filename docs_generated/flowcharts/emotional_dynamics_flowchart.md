# emotional_dynamics Module Flowchart

```mermaid
graph TD
    emotional_dynamics[emotional_dynamics]
    emotional_dynamics_CoreEmotion[CoreEmotion]
    emotional_dynamics --> emotional_dynamics_CoreEmotion
    emotional_dynamics_EmotionalState[EmotionalState]
    emotional_dynamics --> emotional_dynamics_EmotionalState
    emotional_dynamics_EmotionalState___post_init__[__post_init__()]
    emotional_dynamics_EmotionalState --> emotional_dynamics_EmotionalState___post_init__
    emotional_dynamics_EmotionalDynamics[EmotionalDynamics]
    emotional_dynamics --> emotional_dynamics_EmotionalDynamics
    emotional_dynamics_EmotionalDynamics___init__[__init__()]
    emotional_dynamics_EmotionalDynamics --> emotional_dynamics_EmotionalDynamics___init__
    emotional_dynamics_EmotionalDynamics__setup_julia[_setup_julia()]
    emotional_dynamics_EmotionalDynamics --> emotional_dynamics_EmotionalDynamics__setup_julia
    emotional_dynamics_EmotionalDynamics__generate_compound_emotions[_generate_compound_emotions()]
    emotional_dynamics_EmotionalDynamics --> emotional_dynamics_EmotionalDynamics__generate_compound_emotions
    emotional_dynamics_EmotionalDynamics_simulate_emotional_dynamics[simulate_emotional_dynamics()]
    emotional_dynamics_EmotionalDynamics --> emotional_dynamics_EmotionalDynamics_simulate_emotional_dynamics
    emotional_dynamics_EmotionalDynamics__simulate_python_fallback[_simulate_python_fallback()]
    emotional_dynamics_EmotionalDynamics --> emotional_dynamics_EmotionalDynamics__simulate_python_fallback
    style emotional_dynamics fill:#ffcc99
```