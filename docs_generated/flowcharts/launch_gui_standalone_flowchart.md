# launch_gui_standalone Module Flowchart

```mermaid
graph TD
    launch_gui_standalone[launch_gui_standalone]
    launch_gui_standalone_signal_handler[signal_handler()]
    launch_gui_standalone --> launch_gui_standalone_signal_handler
    launch_gui_standalone_main[main()]
    launch_gui_standalone --> launch_gui_standalone_main
    style launch_gui_standalone fill:#ffcc99
```