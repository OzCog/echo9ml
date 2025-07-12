#!/usr/bin/env python3
"""
Distributed Cognitive Mesh API Server for Phase 4 Echo9ML

This module implements REST/WebSocket APIs to expose the distributed cognitive
grammar network for embodied cognition applications including Unity3D, ROS,
and web agents.

Key Features:
- FastAPI-based REST endpoints for cognitive state access
- WebSocket server for real-time bidirectional communication
- Authentication and rate limiting
- Integration with existing cognitive architecture
- Support for embodiment platforms (Unity3D, ROS, Web)
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

# FastAPI and WebSocket imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Security
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available")

# Import existing cognitive components
try:
    from distributed_cognitive_grammar import DistributedCognitiveNetwork, Echo9MLNode
    from neural_symbolic_synthesis import NeuralSymbolicSynthesisEngine
    from cognitive_architecture import CognitiveArchitecture
    from memory_management import HypergraphMemory
    from echoself_introspection import EchoselfIntrospector
    COGNITIVE_COMPONENTS_AVAILABLE = True
except ImportError:
    COGNITIVE_COMPONENTS_AVAILABLE = False
    logging.warning("Some cognitive components not available")

logger = logging.getLogger(__name__)

class EmbodimentPlatform(Enum):
    """Supported embodiment platforms"""
    UNITY3D = "unity3d"
    ROS = "ros"
    WEB = "web"
    SIMULATION = "simulation"
    MOBILE = "mobile"

class CognitiveEndpointType(Enum):
    """Types of cognitive endpoints"""
    STATE_QUERY = "state_query"
    MEMORY_ACCESS = "memory_access"
    ATTENTION_CONTROL = "attention_control"
    REASONING_REQUEST = "reasoning_request"
    LEARNING_UPDATE = "learning_update"
    EMBODIMENT_SYNC = "embodiment_sync"

# Pydantic models for API
class CognitiveStateRequest(BaseModel):
    """Request model for cognitive state queries"""
    agent_id: str = Field(..., description="ID of the cognitive agent")
    state_type: str = Field(..., description="Type of state to query")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")

class CognitiveStateResponse(BaseModel):
    """Response model for cognitive state"""
    agent_id: str
    state_type: str
    timestamp: datetime
    state_data: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)

class MemoryQueryRequest(BaseModel):
    """Request model for memory queries"""
    query_type: str = Field(..., description="Type of memory query")
    query_parameters: Dict[str, Any] = Field(default_factory=dict)
    salience_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=100)

class EmbodimentSyncRequest(BaseModel):
    """Request model for embodiment synchronization"""
    platform: EmbodimentPlatform
    embodiment_data: Dict[str, Any]
    sync_type: str = Field(..., description="Type of synchronization")
    timestamp: datetime = Field(default_factory=datetime.now)

class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    message_type: str
    sender_id: str
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

@dataclass
class CognitiveConnection:
    """Represents a connection to the cognitive mesh"""
    connection_id: str
    platform: EmbodimentPlatform
    websocket: Optional[Any] = None
    last_ping: datetime = field(default_factory=datetime.now)
    authenticated: bool = False
    rate_limit_tokens: int = 100
    rate_limit_reset: datetime = field(default_factory=datetime.now)

class CognitiveMeshAPI:
    """Main API server for the distributed cognitive mesh"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.app = None
        self.cognitive_network = None
        self.connections: Dict[str, CognitiveConnection] = {}
        self.security = HTTPBearer() if FASTAPI_AVAILABLE else None
        
        # Rate limiting
        self.rate_limits = {
            "default": 100,  # requests per minute
            "embodiment": 200,  # higher for real-time embodiment
            "memory": 50,  # lower for expensive memory operations
        }
        
        self._setup_app()
        self._setup_cognitive_network()
    
    def _setup_app(self):
        """Initialize FastAPI application"""
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available - cannot create API server")
            return
            
        self.app = FastAPI(
            title="Echo9ML Cognitive Mesh API",
            description="REST/WebSocket API for distributed cognitive grammar embodiment",
            version="4.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS middleware for web applications
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._register_routes()
    
    def _setup_cognitive_network(self):
        """Initialize the cognitive network backend"""
        if not COGNITIVE_COMPONENTS_AVAILABLE:
            logger.warning("Cognitive components not available - using mock backend")
            return
            
        try:
            self.cognitive_network = DistributedCognitiveNetwork()
            # Add a primary cognitive agent
            self.primary_agent = Echo9MLNode("cognitive_mesh_primary", None)
            self.cognitive_network.add_agent(self.primary_agent)
            logger.info("Cognitive network initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cognitive network: {e}")
    
    def _register_routes(self):
        """Register API routes"""
        if not self.app:
            return
        
        # Health and status endpoints
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now(),
                "cognitive_network": "available" if self.cognitive_network else "unavailable",
                "active_connections": len(self.connections)
            }
        
        @self.app.get("/api/v1/status")
        async def get_api_status():
            return {
                "api_version": "4.0.0",
                "cognitive_agents": len(self.cognitive_network.agents) if self.cognitive_network else 0,
                "active_connections": len(self.connections),
                "supported_platforms": [p.value for p in EmbodimentPlatform],
                "endpoints": {
                    "cognitive_state": "/api/v1/cognitive/state",
                    "memory": "/api/v1/cognitive/memory",
                    "embodiment": "/api/v1/embodiment",
                    "websocket": "/ws/{connection_id}"
                }
            }
        
        # Cognitive state endpoints
        @self.app.post("/api/v1/cognitive/state", response_model=CognitiveStateResponse)
        async def query_cognitive_state(
            request: CognitiveStateRequest,
            auth: HTTPAuthorizationCredentials = Security(self.security)
        ):
            if not await self._check_rate_limit("default", auth.credentials if auth else "anonymous"):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            return await self._handle_cognitive_state_query(request)
        
        @self.app.post("/api/v1/cognitive/memory")
        async def query_memory(
            request: MemoryQueryRequest,
            auth: HTTPAuthorizationCredentials = Security(self.security)
        ):
            if not await self._check_rate_limit("memory", auth.credentials if auth else "anonymous"):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            return await self._handle_memory_query(request)
        
        # Embodiment endpoints
        @self.app.post("/api/v1/embodiment/sync")
        async def sync_embodiment(
            request: EmbodimentSyncRequest,
            auth: HTTPAuthorizationCredentials = Security(self.security)
        ):
            if not await self._check_rate_limit("embodiment", auth.credentials if auth else "anonymous"):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            return await self._handle_embodiment_sync(request)
        
        @self.app.get("/api/v1/embodiment/platforms")
        async def get_supported_platforms():
            return {
                "platforms": [
                    {
                        "name": platform.value,
                        "description": self._get_platform_description(platform),
                        "endpoints": self._get_platform_endpoints(platform)
                    }
                    for platform in EmbodimentPlatform
                ]
            }
        
        # WebSocket endpoint
        @self.app.websocket("/ws/{connection_id}")
        async def websocket_endpoint(websocket: WebSocket, connection_id: str):
            await self._handle_websocket_connection(websocket, connection_id)
    
    async def _handle_cognitive_state_query(self, request: CognitiveStateRequest) -> CognitiveStateResponse:
        """Handle cognitive state query requests"""
        try:
            # Mock implementation - replace with actual cognitive network query
            state_data = {
                "attention_level": 0.85,
                "memory_activation": 0.72,
                "reasoning_state": "active",
                "emotional_valence": 0.65,
                "cognitive_load": 0.58,
                "parameters_processed": request.parameters
            }
            
            if self.cognitive_network and request.agent_id in [agent.agent_id for agent in self.cognitive_network.agents]:
                # Use actual cognitive network if available
                # This would be replaced with actual agent state query
                pass
            
            return CognitiveStateResponse(
                agent_id=request.agent_id,
                state_type=request.state_type,
                timestamp=datetime.now(),
                state_data=state_data,
                confidence=0.87
            )
        except Exception as e:
            logger.error(f"Error handling cognitive state query: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _handle_memory_query(self, request: MemoryQueryRequest) -> Dict[str, Any]:
        """Handle memory query requests"""
        try:
            # Mock memory query - replace with actual memory system
            memory_results = [
                {
                    "memory_id": f"mem_{i}",
                    "content": f"Memory content for {request.query_type}",
                    "salience": 0.8 - (i * 0.1),
                    "timestamp": datetime.now() - timedelta(hours=i),
                    "memory_type": "episodic" if i % 2 == 0 else "semantic"
                }
                for i in range(min(request.max_results, 5))
            ]
            
            return {
                "query_type": request.query_type,
                "results": memory_results,
                "total_found": len(memory_results),
                "query_time": 0.02,  # seconds
                "timestamp": datetime.now()
            }
        except Exception as e:
            logger.error(f"Error handling memory query: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _handle_embodiment_sync(self, request: EmbodimentSyncRequest) -> Dict[str, Any]:
        """Handle embodiment synchronization requests"""
        try:
            # Process embodiment data based on platform
            sync_result = {
                "platform": request.platform.value,
                "sync_type": request.sync_type,
                "status": "success",
                "processed_data": self._process_embodiment_data(request.platform, request.embodiment_data),
                "timestamp": datetime.now(),
                "cognitive_response": self._generate_cognitive_response(request)
            }
            
            # Broadcast to connected WebSocket clients if needed
            await self._broadcast_embodiment_update(sync_result)
            
            return sync_result
        except Exception as e:
            logger.error(f"Error handling embodiment sync: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _handle_websocket_connection(self, websocket: WebSocket, connection_id: str):
        """Handle WebSocket connections for real-time communication"""
        await websocket.accept()
        
        # Create connection record
        connection = CognitiveConnection(
            connection_id=connection_id,
            platform=EmbodimentPlatform.WEB,  # Default to web
            websocket=websocket
        )
        self.connections[connection_id] = connection
        
        logger.info(f"WebSocket connection established: {connection_id}")
        
        try:
            # Send welcome message
            await websocket.send_json({
                "message_type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat(),
                "capabilities": ["cognitive_state", "memory_query", "embodiment_sync"]
            })
            
            # Handle incoming messages
            while True:
                data = await websocket.receive_json()
                await self._handle_websocket_message(connection, data)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
        finally:
            if connection_id in self.connections:
                del self.connections[connection_id]
    
    async def _handle_websocket_message(self, connection: CognitiveConnection, data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        try:
            message_type = data.get("message_type")
            payload = data.get("payload", {})
            
            if message_type == "ping":
                await connection.websocket.send_json({
                    "message_type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
                connection.last_ping = datetime.now()
            
            elif message_type == "cognitive_query":
                response = await self._process_websocket_cognitive_query(payload)
                await connection.websocket.send_json({
                    "message_type": "cognitive_response",
                    "payload": response,
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message_type == "embodiment_update":
                await self._process_websocket_embodiment_update(connection, payload)
            
            else:
                logger.warning(f"Unknown WebSocket message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _process_websocket_cognitive_query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process cognitive queries from WebSocket"""
        # Mock cognitive processing
        return {
            "query_id": payload.get("query_id", str(uuid.uuid4())),
            "result": "cognitive_processing_complete",
            "cognitive_state": {
                "attention": 0.82,
                "memory_activation": 0.71,
                "reasoning": "active"
            },
            "processing_time": 0.015
        }
    
    async def _process_websocket_embodiment_update(self, connection: CognitiveConnection, payload: Dict[str, Any]):
        """Process embodiment updates from WebSocket"""
        # Update connection platform if specified
        if "platform" in payload:
            try:
                connection.platform = EmbodimentPlatform(payload["platform"])
            except ValueError:
                logger.warning(f"Unknown platform: {payload['platform']}")
        
        # Process embodiment data
        processed_data = self._process_embodiment_data(connection.platform, payload)
        
        # Send confirmation
        await connection.websocket.send_json({
            "message_type": "embodiment_update_ack",
            "payload": {
                "status": "processed",
                "platform": connection.platform.value,
                "data_processed": len(str(processed_data))
            },
            "timestamp": datetime.now().isoformat()
        })
    
    def _process_embodiment_data(self, platform: EmbodimentPlatform, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process embodiment data based on platform"""
        if platform == EmbodimentPlatform.UNITY3D:
            return self._process_unity3d_data(data)
        elif platform == EmbodimentPlatform.ROS:
            return self._process_ros_data(data)
        elif platform == EmbodimentPlatform.WEB:
            return self._process_web_data(data)
        else:
            return {"processed": True, "platform": platform.value, "data": data}
    
    def _process_unity3d_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Unity3D specific embodiment data"""
        return {
            "unity_processed": True,
            "transform": data.get("transform", {}),
            "animation_state": data.get("animation", {}),
            "physics": data.get("physics", {}),
            "cognitive_mapping": {
                "position_attention": 0.8,
                "movement_intention": 0.7,
                "environmental_awareness": 0.85
            }
        }
    
    def _process_ros_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ROS specific embodiment data"""
        return {
            "ros_processed": True,
            "pose": data.get("pose", {}),
            "twist": data.get("twist", {}),
            "sensor_data": data.get("sensors", {}),
            "cognitive_mapping": {
                "navigation_intention": 0.82,
                "obstacle_awareness": 0.88,
                "goal_orientation": 0.75
            }
        }
    
    def _process_web_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process web-based embodiment data"""
        return {
            "web_processed": True,
            "interaction": data.get("interaction", {}),
            "user_input": data.get("user_input", {}),
            "display_state": data.get("display", {}),
            "cognitive_mapping": {
                "user_attention": 0.79,
                "interface_adaptation": 0.73,
                "response_preparation": 0.86
            }
        }
    
    def _generate_cognitive_response(self, request: EmbodimentSyncRequest) -> Dict[str, Any]:
        """Generate cognitive response for embodiment sync"""
        return {
            "attention_shift": 0.15,
            "memory_activation": ["embodiment_state", "platform_adaptation"],
            "reasoning_trigger": f"{request.platform.value}_interaction",
            "emotional_response": 0.68,
            "action_intention": self._suggest_action(request.platform, request.embodiment_data)
        }
    
    def _suggest_action(self, platform: EmbodimentPlatform, data: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest cognitive actions based on embodiment platform and data"""
        if platform == EmbodimentPlatform.UNITY3D:
            return {"type": "movement", "direction": "forward", "intensity": 0.7}
        elif platform == EmbodimentPlatform.ROS:
            return {"type": "navigation", "goal": "exploration", "speed": 0.5}
        elif platform == EmbodimentPlatform.WEB:
            return {"type": "response", "content": "interactive_feedback", "urgency": 0.6}
        else:
            return {"type": "generic", "action": "observe", "intensity": 0.5}
    
    async def _broadcast_embodiment_update(self, update: Dict[str, Any]):
        """Broadcast embodiment updates to connected WebSocket clients"""
        message = {
            "message_type": "embodiment_broadcast",
            "payload": update,
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected = []
        for connection_id, connection in self.connections.items():
            try:
                await connection.websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            del self.connections[connection_id]
    
    async def _check_rate_limit(self, endpoint_type: str, identifier: str) -> bool:
        """Check rate limiting for API requests"""
        # Simple rate limiting implementation
        # In production, use Redis or similar for distributed rate limiting
        return True  # For now, allow all requests
    
    def _get_platform_description(self, platform: EmbodimentPlatform) -> str:
        """Get description for embodiment platform"""
        descriptions = {
            EmbodimentPlatform.UNITY3D: "Unity3D game engine integration for 3D embodiment",
            EmbodimentPlatform.ROS: "Robot Operating System integration for robotic embodiment",
            EmbodimentPlatform.WEB: "Web browser integration for web-based embodiment",
            EmbodimentPlatform.SIMULATION: "Simulation environment for testing embodiment",
            EmbodimentPlatform.MOBILE: "Mobile device integration for mobile embodiment"
        }
        return descriptions.get(platform, "Unknown platform")
    
    def _get_platform_endpoints(self, platform: EmbodimentPlatform) -> List[str]:
        """Get specific endpoints for embodiment platform"""
        return [
            f"/api/v1/embodiment/{platform.value}/sync",
            f"/api/v1/embodiment/{platform.value}/status",
            f"/ws/{platform.value}"
        ]
    
    async def start_server(self):
        """Start the API server"""
        if not FASTAPI_AVAILABLE:
            logger.error("Cannot start server - FastAPI not available")
            return
        
        logger.info(f"Starting Cognitive Mesh API server on {self.host}:{self.port}")
        
        # Start cognitive network if available
        if self.cognitive_network and COGNITIVE_COMPONENTS_AVAILABLE:
            try:
                await self.cognitive_network.start_network()
                logger.info("Cognitive network started")
            except Exception as e:
                logger.warning(f"Failed to start cognitive network: {e}")
        
        # Start the API server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

def create_cognitive_mesh_api(host: str = "localhost", port: int = 8000) -> CognitiveMeshAPI:
    """Factory function to create a cognitive mesh API instance"""
    return CognitiveMeshAPI(host=host, port=port)

async def main():
    """Main entry point for the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Echo9ML Cognitive Mesh API Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    api = create_cognitive_mesh_api(host=args.host, port=args.port)
    await api.start_server()

if __name__ == "__main__":
    asyncio.run(main())