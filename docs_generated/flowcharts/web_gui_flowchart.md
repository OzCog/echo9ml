# web_gui Module Flowchart

```mermaid
graph TD
    web_gui[web_gui]
    web_gui_start_heartbeat_thread[start_heartbeat_thread()]
    web_gui --> web_gui_start_heartbeat_thread
    web_gui_parse_arguments[parse_arguments()]
    web_gui --> web_gui_parse_arguments
    web_gui_get_system_metrics[get_system_metrics()]
    web_gui --> web_gui_get_system_metrics
    web_gui_get_memory_stats[get_memory_stats()]
    web_gui --> web_gui_get_memory_stats
    web_gui_get_recent_logs[get_recent_logs()]
    web_gui --> web_gui_get_recent_logs
    style web_gui fill:#ffcc99
```