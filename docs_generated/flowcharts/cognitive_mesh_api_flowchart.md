# cognitive_mesh_api Module Flowchart

```mermaid
graph TD
    cognitive_mesh_api[cognitive_mesh_api]
    cognitive_mesh_api_EmbodimentPlatform[EmbodimentPlatform]
    cognitive_mesh_api --> cognitive_mesh_api_EmbodimentPlatform
    cognitive_mesh_api_CognitiveEndpointType[CognitiveEndpointType]
    cognitive_mesh_api --> cognitive_mesh_api_CognitiveEndpointType
    cognitive_mesh_api_CognitiveStateRequest[CognitiveStateRequest]
    cognitive_mesh_api --> cognitive_mesh_api_CognitiveStateRequest
    cognitive_mesh_api_CognitiveStateResponse[CognitiveStateResponse]
    cognitive_mesh_api --> cognitive_mesh_api_CognitiveStateResponse
    cognitive_mesh_api_MemoryQueryRequest[MemoryQueryRequest]
    cognitive_mesh_api --> cognitive_mesh_api_MemoryQueryRequest
    cognitive_mesh_api_EmbodimentSyncRequest[EmbodimentSyncRequest]
    cognitive_mesh_api --> cognitive_mesh_api_EmbodimentSyncRequest
    cognitive_mesh_api_WebSocketMessage[WebSocketMessage]
    cognitive_mesh_api --> cognitive_mesh_api_WebSocketMessage
    cognitive_mesh_api_CognitiveConnection[CognitiveConnection]
    cognitive_mesh_api --> cognitive_mesh_api_CognitiveConnection
    cognitive_mesh_api_CognitiveMeshAPI[CognitiveMeshAPI]
    cognitive_mesh_api --> cognitive_mesh_api_CognitiveMeshAPI
    cognitive_mesh_api_CognitiveMeshAPI___init__[__init__()]
    cognitive_mesh_api_CognitiveMeshAPI --> cognitive_mesh_api_CognitiveMeshAPI___init__
    cognitive_mesh_api_CognitiveMeshAPI__setup_app[_setup_app()]
    cognitive_mesh_api_CognitiveMeshAPI --> cognitive_mesh_api_CognitiveMeshAPI__setup_app
    cognitive_mesh_api_CognitiveMeshAPI__setup_cognitive_network[_setup_cognitive_network()]
    cognitive_mesh_api_CognitiveMeshAPI --> cognitive_mesh_api_CognitiveMeshAPI__setup_cognitive_network
    cognitive_mesh_api_CognitiveMeshAPI__register_routes[_register_routes()]
    cognitive_mesh_api_CognitiveMeshAPI --> cognitive_mesh_api_CognitiveMeshAPI__register_routes
    cognitive_mesh_api_CognitiveMeshAPI__process_embodiment_data[_process_embodiment_data()]
    cognitive_mesh_api_CognitiveMeshAPI --> cognitive_mesh_api_CognitiveMeshAPI__process_embodiment_data
    cognitive_mesh_api_create_cognitive_mesh_api[create_cognitive_mesh_api()]
    cognitive_mesh_api --> cognitive_mesh_api_create_cognitive_mesh_api
    style cognitive_mesh_api fill:#99ccff
```