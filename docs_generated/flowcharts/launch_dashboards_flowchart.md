# launch_dashboards Module Flowchart

```mermaid
graph TD
    launch_dashboards[launch_dashboards]
    launch_dashboards_signal_handler[signal_handler()]
    launch_dashboards --> launch_dashboards_signal_handler
    launch_dashboards_cleanup[cleanup()]
    launch_dashboards --> launch_dashboards_cleanup
    launch_dashboards_launch_gui_dashboard[launch_gui_dashboard()]
    launch_dashboards --> launch_dashboards_launch_gui_dashboard
    launch_dashboards_launch_web_dashboard[launch_web_dashboard()]
    launch_dashboards --> launch_dashboards_launch_web_dashboard
    launch_dashboards_monitor_output[monitor_output()]
    launch_dashboards --> launch_dashboards_monitor_output
    style launch_dashboards fill:#ffcc99
```