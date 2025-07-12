# web_agent_interface Module Flowchart

```mermaid
graph TD
    web_agent_interface[web_agent_interface]
    web_agent_interface_WebAgentType[WebAgentType]
    web_agent_interface --> web_agent_interface_WebAgentType
    web_agent_interface_WebInteractionType[WebInteractionType]
    web_agent_interface --> web_agent_interface_WebInteractionType
    web_agent_interface_WebAgentState[WebAgentState]
    web_agent_interface --> web_agent_interface_WebAgentState
    web_agent_interface_WebInteractionEvent[WebInteractionEvent]
    web_agent_interface --> web_agent_interface_WebInteractionEvent
    web_agent_interface_CognitiveWebResponse[CognitiveWebResponse]
    web_agent_interface --> web_agent_interface_CognitiveWebResponse
    web_agent_interface_WebAgentInterface[WebAgentInterface]
    web_agent_interface --> web_agent_interface_WebAgentInterface
    web_agent_interface_WebAgentInterface___init__[__init__()]
    web_agent_interface_WebAgentInterface --> web_agent_interface_WebAgentInterface___init__
    web_agent_interface_WebAgentInterface__setup_flask_app[_setup_flask_app()]
    web_agent_interface_WebAgentInterface --> web_agent_interface_WebAgentInterface__setup_flask_app
    web_agent_interface_WebAgentInterface__register_routes[_register_routes()]
    web_agent_interface_WebAgentInterface --> web_agent_interface_WebAgentInterface__register_routes
    web_agent_interface_WebAgentInterface__register_socketio_events[_register_socketio_events()]
    web_agent_interface_WebAgentInterface --> web_agent_interface_WebAgentInterface__register_socketio_events
    web_agent_interface_WebAgentInterface__register_web_agent[_register_web_agent()]
    web_agent_interface_WebAgentInterface --> web_agent_interface_WebAgentInterface__register_web_agent
    web_agent_interface_create_web_agent_interface[create_web_agent_interface()]
    web_agent_interface --> web_agent_interface_create_web_agent_interface
    style web_agent_interface fill:#ffcc99
```