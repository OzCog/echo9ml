# network_config Module Flowchart

```mermaid
graph TD
    network_config[network_config]
    network_config_TeamNetwork[TeamNetwork]
    network_config --> network_config_TeamNetwork
    network_config_TeamNetwork___init__[__init__()]
    network_config_TeamNetwork --> network_config_TeamNetwork___init__
    network_config_TeamNetwork__load_config[_load_config()]
    network_config_TeamNetwork --> network_config_TeamNetwork__load_config
    network_config_TeamNetwork__create_default_config[_create_default_config()]
    network_config_TeamNetwork --> network_config_TeamNetwork__create_default_config
    network_config_TeamNetwork__save_config[_save_config()]
    network_config_TeamNetwork --> network_config_TeamNetwork__save_config
    network_config_TeamNetwork_get_team_member_info[get_team_member_info()]
    network_config_TeamNetwork --> network_config_TeamNetwork_get_team_member_info
    style network_config fill:#ffcc99
```