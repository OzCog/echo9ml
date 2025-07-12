# sensory_motor Module Flowchart

```mermaid
graph TD
    sensory_motor[sensory_motor]
    sensory_motor_SensoryMotor[SensoryMotor]
    sensory_motor --> sensory_motor_SensoryMotor
    sensory_motor_SensoryMotor___init__[__init__()]
    sensory_motor_SensoryMotor --> sensory_motor_SensoryMotor___init__
    sensory_motor_SensoryMotor__load_activities[_load_activities()]
    sensory_motor_SensoryMotor --> sensory_motor_SensoryMotor__load_activities
    sensory_motor_SensoryMotor__save_activities[_save_activities()]
    sensory_motor_SensoryMotor --> sensory_motor_SensoryMotor__save_activities
    sensory_motor_SensoryMotor__log_activity[_log_activity()]
    sensory_motor_SensoryMotor --> sensory_motor_SensoryMotor__log_activity
    sensory_motor_SensoryMotor_capture_screen[capture_screen()]
    sensory_motor_SensoryMotor --> sensory_motor_SensoryMotor_capture_screen
    style sensory_motor fill:#ffcc99
```