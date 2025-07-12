# swarmprotocol Module Flowchart

```mermaid
graph TD
    swarmprotocol[swarmprotocol]
    swarmprotocol_MessageBroker[MessageBroker]
    swarmprotocol --> swarmprotocol_MessageBroker
    swarmprotocol_MessageBroker___init__[__init__()]
    swarmprotocol_MessageBroker --> swarmprotocol_MessageBroker___init__
    swarmprotocol_MessageBroker_subscribe[subscribe()]
    swarmprotocol_MessageBroker --> swarmprotocol_MessageBroker_subscribe
    swarmprotocol_RLAgent[RLAgent]
    swarmprotocol --> swarmprotocol_RLAgent
    swarmprotocol_RLAgent___init__[__init__()]
    swarmprotocol_RLAgent --> swarmprotocol_RLAgent___init__
    swarmprotocol_RLAgent_choose_action[choose_action()]
    swarmprotocol_RLAgent --> swarmprotocol_RLAgent_choose_action
    swarmprotocol_RLAgent_update[update()]
    swarmprotocol_RLAgent --> swarmprotocol_RLAgent_update
    swarmprotocol_RLAgent_set_last[set_last()]
    swarmprotocol_RLAgent --> swarmprotocol_RLAgent_set_last
    swarmprotocol_RLAgent_get_policy[get_policy()]
    swarmprotocol_RLAgent --> swarmprotocol_RLAgent_get_policy
    swarmprotocol_PixieRobot[PixieRobot]
    swarmprotocol --> swarmprotocol_PixieRobot
    swarmprotocol_PixieRobot___init__[__init__()]
    swarmprotocol_PixieRobot --> swarmprotocol_PixieRobot___init__
    swarmprotocol_PixieRobot_simulate_action[simulate_action()]
    swarmprotocol_PixieRobot --> swarmprotocol_PixieRobot_simulate_action
    swarmprotocol_PixieRobot_get_new_state[get_new_state()]
    swarmprotocol_PixieRobot --> swarmprotocol_PixieRobot_get_new_state
    style swarmprotocol fill:#99ccff
```