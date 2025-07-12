# terminal_controller Module Flowchart

```mermaid
graph TD
    terminal_controller[terminal_controller]
    terminal_controller_TerminalController[TerminalController]
    terminal_controller --> terminal_controller_TerminalController
    terminal_controller_TerminalController___init__[__init__()]
    terminal_controller_TerminalController --> terminal_controller_TerminalController___init__
    terminal_controller_TerminalController_start[start()]
    terminal_controller_TerminalController --> terminal_controller_TerminalController_start
    terminal_controller_TerminalController_stop[stop()]
    terminal_controller_TerminalController --> terminal_controller_TerminalController_stop
    terminal_controller_TerminalController_execute_command[execute_command()]
    terminal_controller_TerminalController --> terminal_controller_TerminalController_execute_command
    terminal_controller_TerminalController__process_commands[_process_commands()]
    terminal_controller_TerminalController --> terminal_controller_TerminalController__process_commands
```