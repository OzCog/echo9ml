# monitor Module Flowchart

```mermaid
graph TD
    monitor[monitor]
    monitor_TeamMember[TeamMember]
    monitor --> monitor_TeamMember
    monitor_TeamMember___init__[__init__()]
    monitor_TeamMember --> monitor_TeamMember___init__
    monitor_DeepEchoMonitor[DeepEchoMonitor]
    monitor --> monitor_DeepEchoMonitor
    monitor_DeepEchoMonitor___init__[__init__()]
    monitor_DeepEchoMonitor --> monitor_DeepEchoMonitor___init__
    monitor_DeepEchoMonitor__get_system_info[_get_system_info()]
    monitor_DeepEchoMonitor --> monitor_DeepEchoMonitor__get_system_info
    monitor_DeepEchoMonitor_get_process[get_process()]
    monitor_DeepEchoMonitor --> monitor_DeepEchoMonitor_get_process
    monitor_DeepEchoMonitor_get_system_stats[get_system_stats()]
    monitor_DeepEchoMonitor --> monitor_DeepEchoMonitor_get_system_stats
    monitor_DeepEchoMonitor_get_process_stats[get_process_stats()]
    monitor_DeepEchoMonitor --> monitor_DeepEchoMonitor_get_process_stats
```