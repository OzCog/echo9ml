# fix_locale_gui Module Flowchart

```mermaid
graph TD
    fix_locale_gui[fix_locale_gui]
    fix_locale_gui_signal_handler[signal_handler()]
    fix_locale_gui --> fix_locale_gui_signal_handler
    fix_locale_gui_patch_ttkbootstrap_locale[patch_ttkbootstrap_locale()]
    fix_locale_gui --> fix_locale_gui_patch_ttkbootstrap_locale
    fix_locale_gui_get_ip_and_hostname[get_ip_and_hostname()]
    fix_locale_gui --> fix_locale_gui_get_ip_and_hostname
    fix_locale_gui_main[main()]
    fix_locale_gui --> fix_locale_gui_main
    style fix_locale_gui fill:#ffcc99
```