#!/usr/bin/env python3
"""
Web Agent Embodiment Interface for Echo9ML Cognitive Mesh

This module provides web browser-based embodiment for the distributed
cognitive grammar network, enabling real-time interaction through web
interfaces, JavaScript agents, and browser automation.

Key Features:
- JavaScript SDK for web agent development
- Real-time WebSocket communication with cognitive mesh
- Browser automation integration
- Web-based cognitive interaction interfaces
- Multi-user collaborative cognitive environments
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Web framework imports
try:
    from flask import Flask, render_template_string, request, jsonify, session
    from flask_socketio import SocketIO, emit, join_room, leave_room
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logging.warning("Flask/SocketIO not available")

# WebSocket and HTTP clients
try:
    import websockets
    import requests
    WEB_CLIENTS_AVAILABLE = True
except ImportError:
    WEB_CLIENTS_AVAILABLE = False
    logging.warning("Web client libraries not available")

logger = logging.getLogger(__name__)

class WebAgentType(Enum):
    """Types of web agents"""
    BROWSER_USER = "browser_user"
    JAVASCRIPT_AGENT = "javascript_agent"
    AUTOMATED_BROWSER = "automated_browser"
    COLLABORATIVE_USER = "collaborative_user"
    MOBILE_USER = "mobile_user"
    API_CLIENT = "api_client"

class WebInteractionType(Enum):
    """Types of web interactions"""
    CLICK = "click"
    HOVER = "hover"
    KEYBOARD_INPUT = "keyboard_input"
    SCROLL = "scroll"
    DRAG_DROP = "drag_drop"
    VOICE_INPUT = "voice_input"
    GESTURE = "gesture"
    EYE_TRACKING = "eye_tracking"

@dataclass
class WebAgentState:
    """Web agent state information"""
    agent_id: str
    agent_type: WebAgentType
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    current_page: str = "/"
    viewport_size: Tuple[int, int] = (1920, 1080)
    user_attention: float = 0.5
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    cognitive_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class WebInteractionEvent:
    """Web interaction event data"""
    event_type: WebInteractionType
    element_id: Optional[str] = None
    element_type: Optional[str] = None
    coordinates: Optional[Tuple[int, int]] = None
    input_data: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CognitiveWebResponse:
    """Cognitive response for web interactions"""
    response_type: str
    content: Dict[str, Any]
    ui_updates: List[Dict[str, Any]] = field(default_factory=list)
    cognitive_confidence: float = 0.8
    requires_user_action: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

class WebAgentInterface:
    """Web agent embodiment interface for cognitive mesh"""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 5000,
                 cognitive_api_url: str = "ws://localhost:8000/ws/web"):
        self.host = host
        self.port = port
        self.cognitive_api_url = cognitive_api_url
        
        # Flask application
        self.app = None
        self.socketio = None
        
        # Web agents tracking
        self.active_agents: Dict[str, WebAgentState] = {}
        self.cognitive_websocket = None
        self.is_connected_to_mesh = False
        
        # Cognitive mesh connection
        self.connection_id = f"web_interface_{str(uuid.uuid4())[:8]}"
        
        # Configuration
        self.max_agents = 100
        self.interaction_buffer_size = 1000
        self.attention_threshold = 0.6
        
        self._setup_flask_app()
        logger.info(f"Web agent interface initialized on {host}:{port}")
    
    def _setup_flask_app(self):
        """Setup Flask application with SocketIO"""
        if not FLASK_AVAILABLE:
            logger.error("Flask not available - cannot create web interface")
            return
        
        self.app = Flask(__name__)
        self.app.secret_key = str(uuid.uuid4())
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self._register_routes()
        self._register_socketio_events()
    
    def _register_routes(self):
        """Register Flask routes"""
        if not self.app:
            return
        
        @self.app.route('/')
        def index():
            """Main cognitive web interface"""
            return render_template_string(self._get_main_interface_html())
        
        @self.app.route('/agent/<agent_type>')
        def agent_interface(agent_type):
            """Specialized agent interfaces"""
            if agent_type == "javascript":
                return render_template_string(self._get_javascript_agent_html())
            elif agent_type == "collaborative":
                return render_template_string(self._get_collaborative_interface_html())
            elif agent_type == "mobile":
                return render_template_string(self._get_mobile_interface_html())
            else:
                return render_template_string(self._get_generic_agent_html(agent_type))
        
        @self.app.route('/api/agent/register', methods=['POST'])
        def register_agent():
            """Register new web agent"""
            data = request.get_json()
            agent_id = self._register_web_agent(
                agent_type=WebAgentType(data.get('agent_type', 'browser_user')),
                user_id=data.get('user_id'),
                metadata=data.get('metadata', {})
            )
            return jsonify({"agent_id": agent_id, "status": "registered"})
        
        @self.app.route('/api/cognitive/query', methods=['POST'])
        def cognitive_query():
            """Handle cognitive queries from web agents"""
            data = request.get_json()
            response = asyncio.run(self._handle_cognitive_query(data))
            return jsonify(response)
        
        @self.app.route('/api/interaction/report', methods=['POST'])
        def report_interaction():
            """Report interaction event"""
            data = request.get_json()
            agent_id = data.get('agent_id')
            if agent_id in self.active_agents:
                event = WebInteractionEvent(
                    event_type=WebInteractionType(data.get('event_type')),
                    element_id=data.get('element_id'),
                    element_type=data.get('element_type'),
                    coordinates=tuple(data.get('coordinates', [])) if data.get('coordinates') else None,
                    input_data=data.get('input_data'),
                    metadata=data.get('metadata', {})
                )
                asyncio.run(self._process_interaction_event(agent_id, event))
                return jsonify({"status": "processed"})
            return jsonify({"status": "agent_not_found"}), 404
        
        @self.app.route('/sdk/echo9ml-web-agent.js')
        def web_agent_sdk():
            """Serve JavaScript SDK for web agents"""
            return self._get_javascript_sdk(), 200, {'Content-Type': 'application/javascript'}
    
    def _register_socketio_events(self):
        """Register SocketIO event handlers"""
        if not self.socketio:
            return
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            agent_id = str(uuid.uuid4())
            session['agent_id'] = agent_id
            
            # Register web agent
            self._register_web_agent(
                agent_type=WebAgentType.BROWSER_USER,
                user_id=request.sid,
                metadata={"session_id": request.sid}
            )
            
            emit('agent_registered', {"agent_id": agent_id})
            logger.info(f"Web agent connected: {agent_id}")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            agent_id = session.get('agent_id')
            if agent_id and agent_id in self.active_agents:
                del self.active_agents[agent_id]
                logger.info(f"Web agent disconnected: {agent_id}")
        
        @self.socketio.on('interaction_event')
        def handle_interaction_event(data):
            """Handle real-time interaction events"""
            agent_id = session.get('agent_id')
            if agent_id:
                event = WebInteractionEvent(
                    event_type=WebInteractionType(data.get('event_type')),
                    element_id=data.get('element_id'),
                    element_type=data.get('element_type'),
                    coordinates=tuple(data.get('coordinates', [])) if data.get('coordinates') else None,
                    input_data=data.get('input_data'),
                    metadata=data.get('metadata', {})
                )
                asyncio.run(self._process_interaction_event(agent_id, event))
        
        @self.socketio.on('cognitive_query')
        def handle_cognitive_query_socketio(data):
            """Handle cognitive queries via SocketIO"""
            agent_id = session.get('agent_id')
            if agent_id:
                data['agent_id'] = agent_id
                response = asyncio.run(self._handle_cognitive_query(data))
                emit('cognitive_response', response)
        
        @self.socketio.on('join_collaborative_room')
        def handle_join_room(data):
            """Handle joining collaborative rooms"""
            room_id = data.get('room_id', 'default')
            join_room(room_id)
            agent_id = session.get('agent_id')
            emit('joined_room', {"room_id": room_id, "agent_id": agent_id}, room=room_id)
    
    def _register_web_agent(self, 
                          agent_type: WebAgentType, 
                          user_id: Optional[str] = None,
                          metadata: Dict[str, Any] = None) -> str:
        """Register a new web agent"""
        if len(self.active_agents) >= self.max_agents:
            raise Exception("Maximum number of agents reached")
        
        agent_id = str(uuid.uuid4())
        agent_state = WebAgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            user_id=user_id,
            session_id=metadata.get('session_id') if metadata else None,
            cognitive_context=metadata or {}
        )
        
        self.active_agents[agent_id] = agent_state
        
        # Report to cognitive mesh
        if self.is_connected_to_mesh:
            asyncio.create_task(self._report_agent_registration(agent_state))
        
        logger.info(f"Registered web agent: {agent_id} ({agent_type.value})")
        return agent_id
    
    async def _process_interaction_event(self, agent_id: str, event: WebInteractionEvent):
        """Process web interaction event"""
        if agent_id not in self.active_agents:
            return
        
        agent = self.active_agents[agent_id]
        agent.interaction_history.append({
            "event_type": event.event_type.value,
            "element_id": event.element_id,
            "element_type": event.element_type,
            "coordinates": event.coordinates,
            "input_data": event.input_data,
            "metadata": event.metadata,
            "timestamp": event.timestamp.isoformat()
        })
        
        # Keep history manageable
        if len(agent.interaction_history) > self.interaction_buffer_size:
            agent.interaction_history = agent.interaction_history[-500:]
        
        # Update attention based on interaction
        agent.user_attention = min(1.0, agent.user_attention + 0.1)
        
        # Send to cognitive mesh for processing
        await self._send_interaction_to_mesh(agent_id, event)
        
        # Generate cognitive response if appropriate
        if agent.user_attention > self.attention_threshold:
            cognitive_response = await self._generate_cognitive_response(agent, event)
            if cognitive_response:
                await self._send_cognitive_response_to_agent(agent_id, cognitive_response)
    
    async def _handle_cognitive_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cognitive query from web agent"""
        agent_id = data.get('agent_id')
        query_type = data.get('query_type', 'general')
        query_parameters = data.get('parameters', {})
        
        if not agent_id or agent_id not in self.active_agents:
            return {"error": "Invalid agent ID"}
        
        # Send query to cognitive mesh
        cognitive_response = await self._query_cognitive_mesh(query_type, query_parameters, agent_id)
        
        return {
            "agent_id": agent_id,
            "query_type": query_type,
            "response": cognitive_response,
            "timestamp": datetime.now().isoformat()
        }
    
    async def connect_to_cognitive_mesh(self):
        """Connect to the cognitive mesh API"""
        if not WEB_CLIENTS_AVAILABLE:
            logger.error("Web client libraries not available")
            return False
        
        try:
            self.cognitive_websocket = await websockets.connect(
                f"{self.cognitive_api_url.replace('/web', '')}/{self.connection_id}"
            )
            self.is_connected_to_mesh = True
            logger.info("Connected to cognitive mesh")
            
            # Send platform identification
            await self._send_to_cognitive_mesh({
                "message_type": "platform_identification",
                "platform": "web",
                "capabilities": [
                    "user_interaction",
                    "real_time_communication",
                    "collaborative_environments",
                    "browser_automation",
                    "mobile_support"
                ],
                "active_agents": len(self.active_agents)
            })
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to cognitive mesh: {e}")
            self.is_connected_to_mesh = False
            return False
    
    async def disconnect_from_cognitive_mesh(self):
        """Disconnect from cognitive mesh"""
        if self.cognitive_websocket:
            await self.cognitive_websocket.close()
            self.is_connected_to_mesh = False
            logger.info("Disconnected from cognitive mesh")
    
    async def _send_to_cognitive_mesh(self, payload: Dict[str, Any]):
        """Send message to cognitive mesh"""
        if not self.is_connected_to_mesh or not self.cognitive_websocket:
            return
        
        try:
            message = {
                "message_type": "web_agent_data",
                "sender_id": self.connection_id,
                "payload": payload,
                "timestamp": datetime.now().isoformat()
            }
            await self.cognitive_websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send to cognitive mesh: {e}")
    
    async def _report_agent_registration(self, agent_state: WebAgentState):
        """Report agent registration to cognitive mesh"""
        await self._send_to_cognitive_mesh({
            "type": "agent_registration",
            "agent_id": agent_state.agent_id,
            "agent_type": agent_state.agent_type.value,
            "user_id": agent_state.user_id,
            "metadata": agent_state.cognitive_context
        })
    
    async def _send_interaction_to_mesh(self, agent_id: str, event: WebInteractionEvent):
        """Send interaction event to cognitive mesh"""
        await self._send_to_cognitive_mesh({
            "type": "interaction_event",
            "agent_id": agent_id,
            "event_type": event.event_type.value,
            "element_id": event.element_id,
            "element_type": event.element_type,
            "coordinates": event.coordinates,
            "input_data": event.input_data,
            "metadata": event.metadata,
            "timestamp": event.timestamp.isoformat()
        })
    
    async def _query_cognitive_mesh(self, query_type: str, parameters: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        """Query cognitive mesh for cognitive processing"""
        # Mock implementation - replace with actual mesh query
        return {
            "cognitive_state": {
                "attention_level": 0.75,
                "memory_activation": 0.68,
                "reasoning_state": "processing",
                "response_confidence": 0.82
            },
            "suggestions": [
                "Focus on the highlighted element",
                "Consider exploring related content",
                "Review previous interactions"
            ],
            "ui_adaptations": {
                "highlight_elements": ["#main-content"],
                "suggest_actions": ["scroll", "click"],
                "attention_indicators": True
            }
        }
    
    async def _generate_cognitive_response(self, agent: WebAgentState, event: WebInteractionEvent) -> Optional[CognitiveWebResponse]:
        """Generate cognitive response based on interaction"""
        # Mock cognitive response generation
        response_content = {
            "interaction_acknowledged": True,
            "cognitive_interpretation": f"User {event.event_type.value} on {event.element_type}",
            "suggested_next_actions": ["explore", "focus", "remember"],
            "attention_update": agent.user_attention
        }
        
        ui_updates = []
        if event.event_type == WebInteractionType.CLICK:
            ui_updates.append({
                "type": "highlight",
                "target": event.element_id,
                "duration": 2000
            })
        
        return CognitiveWebResponse(
            response_type="interaction_response",
            content=response_content,
            ui_updates=ui_updates,
            cognitive_confidence=0.8
        )
    
    async def _send_cognitive_response_to_agent(self, agent_id: str, response: CognitiveWebResponse):
        """Send cognitive response to web agent"""
        if self.socketio:
            self.socketio.emit('cognitive_response', {
                "agent_id": agent_id,
                "response_type": response.response_type,
                "content": response.content,
                "ui_updates": response.ui_updates,
                "cognitive_confidence": response.cognitive_confidence,
                "timestamp": response.timestamp.isoformat()
            })
    
    def _get_main_interface_html(self) -> str:
        """Get main web interface HTML"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Echo9ML Cognitive Web Interface</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="/sdk/echo9ml-web-agent.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #1e3a8a; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .cognitive-panel { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .attention-meter { width: 100%; height: 20px; background: #e5e7eb; border-radius: 10px; overflow: hidden; }
        .attention-fill { height: 100%; background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981); transition: width 0.3s; }
        .interaction-log { max-height: 300px; overflow-y: auto; border: 1px solid #d1d5db; padding: 10px; border-radius: 4px; }
        .cognitive-response { background: #f3f4f6; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .button { background: #3b82f6; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        .button:hover { background: #2563eb; }
        .highlighted { box-shadow: 0 0 10px #3b82f6; border: 2px solid #3b82f6; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Echo9ML Cognitive Web Interface</h1>
            <p>Real-time embodied cognition through web interaction</p>
        </div>
        
        <div class="cognitive-panel">
            <h3>Cognitive State</h3>
            <div>
                <label>Attention Level:</label>
                <div class="attention-meter">
                    <div class="attention-fill" id="attention-meter" style="width: 50%;"></div>
                </div>
                <span id="attention-value">50%</span>
            </div>
            <div style="margin-top: 15px;">
                <strong>Agent ID:</strong> <span id="agent-id">Connecting...</span><br>
                <strong>Cognitive Confidence:</strong> <span id="cognitive-confidence">--</span><br>
                <strong>Memory Activation:</strong> <span id="memory-activation">--</span>
            </div>
        </div>
        
        <div class="cognitive-panel">
            <h3>Interactive Elements</h3>
            <button class="button" onclick="testInteraction('exploration')">🔍 Explore Content</button>
            <button class="button" onclick="testInteraction('focus')">🎯 Focus Attention</button>
            <button class="button" onclick="testInteraction('memory')">💭 Access Memory</button>
            <button class="button" onclick="testInteraction('reasoning')">🧮 Trigger Reasoning</button>
            
            <div style="margin-top: 20px;">
                <input type="text" id="cognitive-input" placeholder="Enter cognitive query..." style="width: 300px; padding: 8px;">
                <button class="button" onclick="sendCognitiveQuery()">Send Query</button>
            </div>
        </div>
        
        <div class="cognitive-panel">
            <h3>Cognitive Responses</h3>
            <div id="cognitive-responses"></div>
        </div>
        
        <div class="cognitive-panel">
            <h3>Interaction Log</h3>
            <div class="interaction-log" id="interaction-log"></div>
        </div>
    </div>

    <script>
        const cognitiveAgent = new Echo9MLWebAgent();
        
        cognitiveAgent.onRegistered = function(agentId) {
            document.getElementById('agent-id').textContent = agentId;
            logInteraction('Agent registered: ' + agentId);
        };
        
        cognitiveAgent.onCognitiveResponse = function(response) {
            updateCognitiveState(response);
            displayCognitiveResponse(response);
            
            // Handle UI updates
            if (response.ui_updates) {
                response.ui_updates.forEach(update => {
                    if (update.type === 'highlight') {
                        highlightElement(update.target, update.duration);
                    }
                });
            }
        };
        
        function testInteraction(type) {
            cognitiveAgent.reportInteraction('click', type + '-button', 'button');
            logInteraction('Testing: ' + type);
        }
        
        function sendCognitiveQuery() {
            const input = document.getElementById('cognitive-input');
            const query = input.value.trim();
            if (query) {
                cognitiveAgent.queryCognitive('text_analysis', { text: query });
                logInteraction('Query: ' + query);
                input.value = '';
            }
        }
        
        function updateCognitiveState(response) {
            if (response.content && response.content.cognitive_state) {
                const state = response.content.cognitive_state;
                if (state.attention_level !== undefined) {
                    const attention = Math.round(state.attention_level * 100);
                    document.getElementById('attention-meter').style.width = attention + '%';
                    document.getElementById('attention-value').textContent = attention + '%';
                }
                if (state.response_confidence !== undefined) {
                    document.getElementById('cognitive-confidence').textContent = 
                        Math.round(state.response_confidence * 100) + '%';
                }
                if (state.memory_activation !== undefined) {
                    document.getElementById('memory-activation').textContent = 
                        Math.round(state.memory_activation * 100) + '%';
                }
            }
        }
        
        function displayCognitiveResponse(response) {
            const container = document.getElementById('cognitive-responses');
            const responseDiv = document.createElement('div');
            responseDiv.className = 'cognitive-response';
            responseDiv.innerHTML = `
                <strong>${response.response_type}</strong><br>
                <em>Confidence: ${Math.round(response.cognitive_confidence * 100)}%</em><br>
                ${JSON.stringify(response.content, null, 2)}
            `;
            container.insertBefore(responseDiv, container.firstChild);
            
            // Keep only last 5 responses
            while (container.children.length > 5) {
                container.removeChild(container.lastChild);
            }
        }
        
        function logInteraction(message) {
            const log = document.getElementById('interaction-log');
            const timestamp = new Date().toLocaleTimeString();
            log.innerHTML += `<div>[${timestamp}] ${message}</div>`;
            log.scrollTop = log.scrollHeight;
        }
        
        function highlightElement(elementId, duration) {
            const element = document.getElementById(elementId) || document.querySelector(elementId);
            if (element) {
                element.classList.add('highlighted');
                setTimeout(() => {
                    element.classList.remove('highlighted');
                }, duration);
            }
        }
        
        // Auto-track user interactions
        document.addEventListener('click', function(e) {
            cognitiveAgent.reportInteraction('click', e.target.id, e.target.tagName.toLowerCase(), 
                                           [e.clientX, e.clientY]);
        });
        
        document.addEventListener('mouseover', function(e) {
            if (e.target.classList.contains('button')) {
                cognitiveAgent.reportInteraction('hover', e.target.id, e.target.tagName.toLowerCase());
            }
        });
    </script>
</body>
</html>
'''
    
    def _get_javascript_sdk(self) -> str:
        """Get JavaScript SDK for web agents"""
        return '''
/**
 * Echo9ML Web Agent SDK
 * JavaScript library for integrating with the Echo9ML cognitive mesh
 */
class Echo9MLWebAgent {
    constructor(serverUrl = null) {
        this.serverUrl = serverUrl || window.location.origin;
        this.socket = null;
        this.agentId = null;
        this.isConnected = false;
        
        // Event callbacks
        this.onRegistered = null;
        this.onCognitiveResponse = null;
        this.onDisconnected = null;
        this.onError = null;
        
        this.init();
    }
    
    init() {
        // Initialize SocketIO connection
        if (typeof io !== 'undefined') {
            this.socket = io(this.serverUrl);
            this.setupSocketHandlers();
        } else {
            console.error('Socket.IO not available');
        }
    }
    
    setupSocketHandlers() {
        this.socket.on('connect', () => {
            this.isConnected = true;
            console.log('Connected to Echo9ML cognitive mesh');
        });
        
        this.socket.on('agent_registered', (data) => {
            this.agentId = data.agent_id;
            console.log('Agent registered:', this.agentId);
            if (this.onRegistered) {
                this.onRegistered(this.agentId);
            }
        });
        
        this.socket.on('cognitive_response', (response) => {
            console.log('Cognitive response received:', response);
            if (this.onCognitiveResponse) {
                this.onCognitiveResponse(response);
            }
        });
        
        this.socket.on('disconnect', () => {
            this.isConnected = false;
            console.log('Disconnected from cognitive mesh');
            if (this.onDisconnected) {
                this.onDisconnected();
            }
        });
        
        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
            if (this.onError) {
                this.onError(error);
            }
        });
    }
    
    /**
     * Report user interaction event
     */
    reportInteraction(eventType, elementId, elementType, coordinates = null, inputData = null, metadata = {}) {
        if (!this.isConnected || !this.socket) {
            console.warn('Not connected to cognitive mesh');
            return;
        }
        
        const interactionData = {
            event_type: eventType,
            element_id: elementId,
            element_type: elementType,
            coordinates: coordinates,
            input_data: inputData,
            metadata: {
                ...metadata,
                timestamp: new Date().toISOString(),
                page_url: window.location.href,
                viewport_size: [window.innerWidth, window.innerHeight]
            }
        };
        
        this.socket.emit('interaction_event', interactionData);
    }
    
    /**
     * Send cognitive query
     */
    queryCognitive(queryType, parameters = {}) {
        if (!this.isConnected || !this.socket) {
            console.warn('Not connected to cognitive mesh');
            return;
        }
        
        const queryData = {
            query_type: queryType,
            parameters: parameters,
            timestamp: new Date().toISOString()
        };
        
        this.socket.emit('cognitive_query', queryData);
    }
    
    /**
     * Join collaborative room
     */
    joinCollaborativeRoom(roomId) {
        if (!this.isConnected || !this.socket) {
            console.warn('Not connected to cognitive mesh');
            return;
        }
        
        this.socket.emit('join_collaborative_room', { room_id: roomId });
    }
    
    /**
     * Auto-track page interactions
     */
    enableAutoTracking() {
        // Track clicks
        document.addEventListener('click', (e) => {
            this.reportInteraction('click', e.target.id || 'unnamed', e.target.tagName.toLowerCase(), 
                                 [e.clientX, e.clientY]);
        });
        
        // Track form submissions
        document.addEventListener('submit', (e) => {
            this.reportInteraction('submit', e.target.id || 'unnamed', 'form');
        });
        
        // Track scroll events (throttled)
        let scrollTimeout;
        document.addEventListener('scroll', () => {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                this.reportInteraction('scroll', 'window', 'document', 
                                     [window.scrollX, window.scrollY]);
            }, 100);
        });
        
        // Track keyboard input (on input elements)
        document.addEventListener('input', (e) => {
            if (e.target.tagName.toLowerCase() === 'input' || 
                e.target.tagName.toLowerCase() === 'textarea') {
                this.reportInteraction('keyboard_input', e.target.id || 'unnamed', 
                                     e.target.tagName.toLowerCase(), null, e.target.value);
            }
        });
    }
    
    /**
     * Send HTTP request to cognitive mesh API
     */
    async apiRequest(endpoint, data = {}) {
        try {
            const response = await fetch(`${this.serverUrl}/api/${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    ...data,
                    agent_id: this.agentId
                })
            });
            
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            return null;
        }
    }
    
    /**
     * Register agent with specific type and metadata
     */
    async registerAgent(agentType = 'browser_user', userId = null, metadata = {}) {
        return await this.apiRequest('agent/register', {
            agent_type: agentType,
            user_id: userId,
            metadata: metadata
        });
    }
    
    /**
     * Get agent state
     */
    getAgentState() {
        return {
            agentId: this.agentId,
            isConnected: this.isConnected,
            serverUrl: this.serverUrl,
            timestamp: new Date().toISOString()
        };
    }
}

// Auto-initialize if in browser environment
if (typeof window !== 'undefined') {
    window.Echo9MLWebAgent = Echo9MLWebAgent;
}

// Export for Node.js environment
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Echo9MLWebAgent;
}
'''
    
    def _get_javascript_agent_html(self) -> str:
        """Get JavaScript agent interface HTML"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Echo9ML JavaScript Agent</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="/sdk/echo9ml-web-agent.js"></script>
</head>
<body>
    <h1>JavaScript Agent Interface</h1>
    <div id="agent-status"></div>
    <div id="cognitive-output"></div>
    
    <script>
        const agent = new Echo9MLWebAgent();
        agent.enableAutoTracking();
        
        agent.onRegistered = (agentId) => {
            document.getElementById('agent-status').innerHTML = 
                `Agent registered: ${agentId}`;
        };
        
        agent.onCognitiveResponse = (response) => {
            document.getElementById('cognitive-output').innerHTML += 
                `<div>Response: ${JSON.stringify(response)}</div>`;
        };
    </script>
</body>
</html>
'''
    
    def _get_collaborative_interface_html(self) -> str:
        """Get collaborative interface HTML"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Echo9ML Collaborative Interface</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="/sdk/echo9ml-web-agent.js"></script>
</head>
<body>
    <h1>Collaborative Cognitive Environment</h1>
    <div>
        <input type="text" id="room-id" placeholder="Room ID" value="default">
        <button onclick="joinRoom()">Join Room</button>
    </div>
    <div id="collaboration-area"></div>
    
    <script>
        const agent = new Echo9MLWebAgent();
        
        function joinRoom() {
            const roomId = document.getElementById('room-id').value;
            agent.joinCollaborativeRoom(roomId);
        }
    </script>
</body>
</html>
'''
    
    def _get_mobile_interface_html(self) -> str:
        """Get mobile interface HTML"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Echo9ML Mobile Interface</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="/sdk/echo9ml-web-agent.js"></script>
    <style>
        body { padding: 10px; font-size: 18px; }
        button { padding: 15px; font-size: 16px; margin: 5px; }
    </style>
</head>
<body>
    <h1>Mobile Cognitive Interface</h1>
    <div id="mobile-content">
        <button onclick="voiceCommand()">🎤 Voice Command</button>
        <button onclick="gestureInput()">👋 Gesture Input</button>
    </div>
    
    <script>
        const agent = new Echo9MLWebAgent();
        
        function voiceCommand() {
            agent.reportInteraction('voice_input', 'voice-button', 'button');
        }
        
        function gestureInput() {
            agent.reportInteraction('gesture', 'gesture-button', 'button');
        }
    </script>
</body>
</html>
'''
    
    def _get_generic_agent_html(self, agent_type: str) -> str:
        """Get generic agent interface HTML"""
        return f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Echo9ML {agent_type.title()} Agent</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="/sdk/echo9ml-web-agent.js"></script>
</head>
<body>
    <h1>{agent_type.title()} Agent Interface</h1>
    <div id="agent-content">
        <p>Specialized interface for {agent_type} agents</p>
    </div>
    
    <script>
        const agent = new Echo9MLWebAgent();
        
        agent.onRegistered = (agentId) => {{
            console.log('Agent registered:', agentId);
        }};
    </script>
</body>
</html>
'''
    
    async def start_server(self):
        """Start the web agent interface server"""
        if not FLASK_AVAILABLE:
            logger.error("Flask not available - cannot start web interface")
            return
        
        # Connect to cognitive mesh
        await self.connect_to_cognitive_mesh()
        
        logger.info(f"Starting web agent interface on {self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=False)

def create_web_agent_interface(host: str = "localhost", 
                             port: int = 5000,
                             cognitive_api_url: str = "ws://localhost:8000/ws/web") -> WebAgentInterface:
    """Factory function to create web agent interface"""
    return WebAgentInterface(host=host, port=port, cognitive_api_url=cognitive_api_url)

async def main():
    """Test web agent interface"""
    interface = create_web_agent_interface()
    await interface.start_server()

if __name__ == "__main__":
    asyncio.run(main())