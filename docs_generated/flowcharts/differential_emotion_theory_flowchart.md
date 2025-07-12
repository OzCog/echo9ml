# differential_emotion_theory Module Flowchart

```mermaid
graph TD
    differential_emotion_theory[differential_emotion_theory]
    differential_emotion_theory_DETEmotion[DETEmotion]
    differential_emotion_theory --> differential_emotion_theory_DETEmotion
    differential_emotion_theory_EmotionalScript[EmotionalScript]
    differential_emotion_theory --> differential_emotion_theory_EmotionalScript
    differential_emotion_theory_EmotionalScript_matches_emotions[matches_emotions()]
    differential_emotion_theory_EmotionalScript --> differential_emotion_theory_EmotionalScript_matches_emotions
    differential_emotion_theory_DETState[DETState]
    differential_emotion_theory --> differential_emotion_theory_DETState
    differential_emotion_theory_DETState___post_init__[__post_init__()]
    differential_emotion_theory_DETState --> differential_emotion_theory_DETState___post_init__
    differential_emotion_theory_DifferentialEmotionSystem[DifferentialEmotionSystem]
    differential_emotion_theory --> differential_emotion_theory_DifferentialEmotionSystem
    differential_emotion_theory_DifferentialEmotionSystem___init__[__init__()]
    differential_emotion_theory_DifferentialEmotionSystem --> differential_emotion_theory_DifferentialEmotionSystem___init__
    differential_emotion_theory_DifferentialEmotionSystem__setup_julia_extensions[_setup_julia_extensions()]
    differential_emotion_theory_DifferentialEmotionSystem --> differential_emotion_theory_DifferentialEmotionSystem__setup_julia_extensions
    differential_emotion_theory_DifferentialEmotionSystem__create_script_library[_create_script_library()]
    differential_emotion_theory_DifferentialEmotionSystem --> differential_emotion_theory_DifferentialEmotionSystem__create_script_library
    differential_emotion_theory_DifferentialEmotionSystem_map_core_to_det[map_core_to_det()]
    differential_emotion_theory_DifferentialEmotionSystem --> differential_emotion_theory_DifferentialEmotionSystem_map_core_to_det
    differential_emotion_theory_DifferentialEmotionSystem_map_det_to_core[map_det_to_core()]
    differential_emotion_theory_DifferentialEmotionSystem --> differential_emotion_theory_DifferentialEmotionSystem_map_det_to_core
    style differential_emotion_theory fill:#ffcc99
```