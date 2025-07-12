# gui_dashboard Module Flowchart

```mermaid
graph TD
    gui_dashboard[gui_dashboard]
    gui_dashboard_GUIDashboard[GUIDashboard]
    gui_dashboard --> gui_dashboard_GUIDashboard
    gui_dashboard_GUIDashboard___init__[__init__()]
    gui_dashboard_GUIDashboard --> gui_dashboard_GUIDashboard___init__
    gui_dashboard_GUIDashboard_log_message[log_message()]
    gui_dashboard_GUIDashboard --> gui_dashboard_GUIDashboard_log_message
    gui_dashboard_GUIDashboard_create_widgets[create_widgets()]
    gui_dashboard_GUIDashboard --> gui_dashboard_GUIDashboard_create_widgets
    gui_dashboard_GUIDashboard_create_dashboard_tab[create_dashboard_tab()]
    gui_dashboard_GUIDashboard --> gui_dashboard_GUIDashboard_create_dashboard_tab
    gui_dashboard_GUIDashboard_create_system_tab[create_system_tab()]
    gui_dashboard_GUIDashboard --> gui_dashboard_GUIDashboard_create_system_tab
    style gui_dashboard fill:#ffcc99
```