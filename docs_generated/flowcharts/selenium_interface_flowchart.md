# selenium_interface Module Flowchart

```mermaid
graph TD
    selenium_interface[selenium_interface]
    selenium_interface_SeleniumInterface[SeleniumInterface]
    selenium_interface --> selenium_interface_SeleniumInterface
    selenium_interface_SeleniumInterface___init__[__init__()]
    selenium_interface_SeleniumInterface --> selenium_interface_SeleniumInterface___init__
    selenium_interface_SeleniumInterface_find_existing_browser[find_existing_browser()]
    selenium_interface_SeleniumInterface --> selenium_interface_SeleniumInterface_find_existing_browser
    selenium_interface_SeleniumInterface_init[init()]
    selenium_interface_SeleniumInterface --> selenium_interface_SeleniumInterface_init
    selenium_interface_SeleniumInterface__setup_event_listeners[_setup_event_listeners()]
    selenium_interface_SeleniumInterface --> selenium_interface_SeleniumInterface__setup_event_listeners
    selenium_interface_SeleniumInterface__handle_console_message[_handle_console_message()]
    selenium_interface_SeleniumInterface --> selenium_interface_SeleniumInterface__handle_console_message
    selenium_interface_main[main()]
    selenium_interface --> selenium_interface_main
    style selenium_interface fill:#ffcc99
```