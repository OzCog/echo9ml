# echopilot Module Flowchart

```mermaid
graph TD
    echopilot[echopilot]
    echopilot_ESMWorker[ESMWorker]
    echopilot --> echopilot_ESMWorker
    echopilot_ESMWorker___init__[__init__()]
    echopilot_ESMWorker --> echopilot_ESMWorker___init__
    echopilot_ConstraintEmitter[ConstraintEmitter]
    echopilot --> echopilot_ConstraintEmitter
    echopilot_ConstraintEmitter___init__[__init__()]
    echopilot_ConstraintEmitter --> echopilot_ConstraintEmitter___init__
    echopilot_ConstraintEmitter_update[update()]
    echopilot_ConstraintEmitter --> echopilot_ConstraintEmitter_update
    echopilot_ConstraintEmitter_get_constraints[get_constraints()]
    echopilot_ConstraintEmitter --> echopilot_ConstraintEmitter_get_constraints
```