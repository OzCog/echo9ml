# ml_system Module Flowchart

```mermaid
graph TD
    ml_system[ml_system]
    ml_system_FallbackModel[FallbackModel]
    ml_system --> ml_system_FallbackModel
    ml_system_FallbackModel___init__[__init__()]
    ml_system_FallbackModel --> ml_system_FallbackModel___init__
    ml_system_FallbackModel_predict[predict()]
    ml_system_FallbackModel --> ml_system_FallbackModel_predict
    ml_system_FallbackModel_fit[fit()]
    ml_system_FallbackModel --> ml_system_FallbackModel_fit
    ml_system_FallbackModel_save[save()]
    ml_system_FallbackModel --> ml_system_FallbackModel_save
    ml_system_FallbackModel_load[load()]
    ml_system_FallbackModel --> ml_system_FallbackModel_load
    ml_system_MLSystem[MLSystem]
    ml_system --> ml_system_MLSystem
    ml_system_MLSystem___init__[__init__()]
    ml_system_MLSystem --> ml_system_MLSystem___init__
    ml_system_MLSystem__load_models[_load_models()]
    ml_system_MLSystem --> ml_system_MLSystem__load_models
    ml_system_MLSystem__create_fallback_models[_create_fallback_models()]
    ml_system_MLSystem --> ml_system_MLSystem__create_fallback_models
    ml_system_MLSystem__load_activities[_load_activities()]
    ml_system_MLSystem --> ml_system_MLSystem__load_activities
    ml_system_MLSystem__save_activities[_save_activities()]
    ml_system_MLSystem --> ml_system_MLSystem__save_activities
    style ml_system fill:#99ccff
```