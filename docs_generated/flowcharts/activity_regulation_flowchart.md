# activity_regulation Module Flowchart

```mermaid
graph TD
    activity_regulation[activity_regulation]
    activity_regulation_ActivityState[ActivityState]
    activity_regulation --> activity_regulation_ActivityState
    activity_regulation_TaskPriority[TaskPriority]
    activity_regulation --> activity_regulation_TaskPriority
    activity_regulation_ScheduledTask[ScheduledTask]
    activity_regulation --> activity_regulation_ScheduledTask
    activity_regulation_ActivityRegulator[ActivityRegulator]
    activity_regulation --> activity_regulation_ActivityRegulator
    activity_regulation_ActivityRegulator___init__[__init__()]
    activity_regulation_ActivityRegulator --> activity_regulation_ActivityRegulator___init__
    activity_regulation_ActivityRegulator__monitor_activities[_monitor_activities()]
    activity_regulation_ActivityRegulator --> activity_regulation_ActivityRegulator__monitor_activities
    activity_regulation_ActivityRegulator__log_activity[_log_activity()]
    activity_regulation_ActivityRegulator --> activity_regulation_ActivityRegulator__log_activity
    activity_regulation_ActivityRegulator_add_task[add_task()]
    activity_regulation_ActivityRegulator --> activity_regulation_ActivityRegulator_add_task
    activity_regulation_ActivityRegulator_remove_task[remove_task()]
    activity_regulation_ActivityRegulator --> activity_regulation_ActivityRegulator_remove_task
```