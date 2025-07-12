# sensory_motor_simple Module Flowchart

```mermaid
graph TD
    sensory_motor_simple[sensory_motor_simple]
    sensory_motor_simple_SensoryMotorSystem[SensoryMotorSystem]
    sensory_motor_simple --> sensory_motor_simple_SensoryMotorSystem
    sensory_motor_simple_SensoryMotorSystem___init__[__init__()]
    sensory_motor_simple_SensoryMotorSystem --> sensory_motor_simple_SensoryMotorSystem___init__
    sensory_motor_simple_SensoryMotorSystem__get_mouse_position[_get_mouse_position()]
    sensory_motor_simple_SensoryMotorSystem --> sensory_motor_simple_SensoryMotorSystem__get_mouse_position
    sensory_motor_simple_SensoryMotorSystem__load_activities[_load_activities()]
    sensory_motor_simple_SensoryMotorSystem --> sensory_motor_simple_SensoryMotorSystem__load_activities
    sensory_motor_simple_SensoryMotorSystem__save_activities[_save_activities()]
    sensory_motor_simple_SensoryMotorSystem --> sensory_motor_simple_SensoryMotorSystem__save_activities
    sensory_motor_simple_SensoryMotorSystem__log_activity[_log_activity()]
    sensory_motor_simple_SensoryMotorSystem --> sensory_motor_simple_SensoryMotorSystem__log_activity
    sensory_motor_simple_create_xauth_file[create_xauth_file()]
    sensory_motor_simple --> sensory_motor_simple_create_xauth_file
    sensory_motor_simple_create_x11_auth_cookie[create_x11_auth_cookie()]
    sensory_motor_simple --> sensory_motor_simple_create_x11_auth_cookie
    sensory_motor_simple_setup_x11_auth[setup_x11_auth()]
    sensory_motor_simple --> sensory_motor_simple_setup_x11_auth
    sensory_motor_simple_ensure_display[ensure_display()]
    sensory_motor_simple --> sensory_motor_simple_ensure_display
    style sensory_motor_simple fill:#ffcc99
```