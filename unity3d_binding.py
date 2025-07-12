#!/usr/bin/env python3
"""
Unity3D Embodiment Binding for Echo9ML Cognitive Mesh

This module provides Unity3D integration for embodied cognition, enabling
real-time communication between Unity3D environments and the distributed
cognitive grammar network.

Key Features:
- JSON-based communication protocol for Unity3D
- Real-time cognitive state synchronization
- Transform and animation data processing
- Physics and interaction event handling
- Cognitive intention to Unity action mapping
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
import websockets

logger = logging.getLogger(__name__)

class Unity3DMessageType(Enum):
    """Unity3D message types for cognitive communication"""
    TRANSFORM_UPDATE = "transform_update"
    ANIMATION_STATE = "animation_state"
    PHYSICS_EVENT = "physics_event"
    INTERACTION_EVENT = "interaction_event"
    COGNITIVE_QUERY = "cognitive_query"
    COGNITIVE_RESPONSE = "cognitive_response"
    INTENTION_ACTION = "intention_action"
    ENVIRONMENT_STATE = "environment_state"
    CAMERA_DATA = "camera_data"
    AUDIO_EVENT = "audio_event"

class Unity3DObjectType(Enum):
    """Unity3D object types for cognitive processing"""
    PLAYER_CHARACTER = "player_character"
    NPC = "npc"
    INTERACTIVE_OBJECT = "interactive_object"
    ENVIRONMENT = "environment"
    UI_ELEMENT = "ui_element"
    CAMERA = "camera"
    LIGHT = "light"
    PHYSICS_OBJECT = "physics_object"

@dataclass
class Unity3DTransform:
    """Unity3D transform data structure"""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # Quaternion
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Unity3DAnimationState:
    """Unity3D animation state data"""
    current_animation: str = ""
    animation_time: float = 0.0
    animation_speed: float = 1.0
    blend_weights: Dict[str, float] = field(default_factory=dict)
    is_playing: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Unity3DPhysicsEvent:
    """Unity3D physics event data"""
    event_type: str  # collision, trigger_enter, trigger_exit
    object_id: str
    other_object_id: str
    contact_point: Optional[Tuple[float, float, float]] = None
    force: Optional[Tuple[float, float, float]] = None
    velocity: Optional[Tuple[float, float, float]] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Unity3DInteractionEvent:
    """Unity3D interaction event data"""
    interaction_type: str  # click, hover, select, use
    object_id: str
    user_id: Optional[str] = None
    interaction_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Unity3DCognitiveIntention:
    """Cognitive intention for Unity3D actions"""
    intention_type: str
    target_object: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    duration: Optional[float] = None
    cognitive_confidence: float = 0.8
    timestamp: datetime = field(default_factory=datetime.now)

class Unity3DBinding:
    """Unity3D embodiment binding for cognitive mesh"""
    
    def __init__(self, cognitive_api_url: str = "ws://localhost:8000/ws/unity3d"):
        self.cognitive_api_url = cognitive_api_url
        self.websocket = None
        self.connection_id = "unity3d_" + str(uuid.uuid4())[:8]
        self.is_connected = False
        self.object_registry: Dict[str, Unity3DObjectType] = {}
        self.cognitive_state: Dict[str, Any] = {}
        self.intention_queue: List[Unity3DCognitiveIntention] = []
        
        # Cognitive mapping parameters
        self.attention_threshold = 0.7
        self.action_confidence_threshold = 0.6
        self.environment_awareness_decay = 0.95
        
        # Unity3D state tracking
        self.last_transform_update = {}
        self.active_animations = {}
        self.physics_events_buffer = []
        self.interaction_history = []
        
        logger.info(f"Unity3D binding initialized with connection ID: {self.connection_id}")
    
    async def connect_to_cognitive_mesh(self):
        """Connect to the cognitive mesh API"""
        try:
            self.websocket = await websockets.connect(
                f"{self.cognitive_api_url.replace('/unity3d', '')}/{self.connection_id}"
            )
            self.is_connected = True
            logger.info("Connected to cognitive mesh")
            
            # Send platform identification
            await self.send_message(Unity3DMessageType.COGNITIVE_QUERY, {
                "type": "platform_identification",
                "platform": "unity3d",
                "capabilities": [
                    "transform_tracking",
                    "animation_control",
                    "physics_simulation",
                    "interaction_handling",
                    "camera_input",
                    "audio_processing"
                ]
            })
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to cognitive mesh: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from cognitive mesh"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("Disconnected from cognitive mesh")
    
    async def send_message(self, message_type: Unity3DMessageType, payload: Dict[str, Any]):
        """Send message to cognitive mesh"""
        if not self.is_connected or not self.websocket:
            logger.warning("Not connected to cognitive mesh")
            return False
        
        try:
            message = {
                "message_type": message_type.value,
                "sender_id": self.connection_id,
                "payload": payload,
                "timestamp": datetime.now().isoformat()
            }
            await self.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from cognitive mesh"""
        if not self.is_connected or not self.websocket:
            return None
        
        try:
            message_raw = await self.websocket.recv()
            message = json.loads(message_raw)
            return message
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None
    
    # Unity3D specific methods
    
    def register_object(self, object_id: str, object_type: Unity3DObjectType):
        """Register Unity3D object for cognitive tracking"""
        self.object_registry[object_id] = object_type
        logger.debug(f"Registered Unity3D object: {object_id} ({object_type.value})")
    
    async def update_transform(self, object_id: str, transform: Unity3DTransform):
        """Update object transform and send to cognitive mesh"""
        self.last_transform_update[object_id] = transform
        
        # Send to cognitive mesh for spatial reasoning
        await self.send_message(Unity3DMessageType.TRANSFORM_UPDATE, {
            "object_id": object_id,
            "object_type": self.object_registry.get(object_id, Unity3DObjectType.PHYSICS_OBJECT).value,
            "position": transform.position,
            "rotation": transform.rotation,
            "scale": transform.scale,
            "timestamp": transform.timestamp.isoformat()
        })
    
    async def update_animation(self, object_id: str, animation_state: Unity3DAnimationState):
        """Update animation state and send to cognitive mesh"""
        self.active_animations[object_id] = animation_state
        
        await self.send_message(Unity3DMessageType.ANIMATION_STATE, {
            "object_id": object_id,
            "animation": animation_state.current_animation,
            "time": animation_state.animation_time,
            "speed": animation_state.animation_speed,
            "blend_weights": animation_state.blend_weights,
            "is_playing": animation_state.is_playing,
            "timestamp": animation_state.timestamp.isoformat()
        })
    
    async def report_physics_event(self, event: Unity3DPhysicsEvent):
        """Report physics event to cognitive mesh"""
        self.physics_events_buffer.append(event)
        
        # Keep buffer size manageable
        if len(self.physics_events_buffer) > 100:
            self.physics_events_buffer = self.physics_events_buffer[-50:]
        
        await self.send_message(Unity3DMessageType.PHYSICS_EVENT, {
            "event_type": event.event_type,
            "object_id": event.object_id,
            "other_object_id": event.other_object_id,
            "contact_point": event.contact_point,
            "force": event.force,
            "velocity": event.velocity,
            "timestamp": event.timestamp.isoformat()
        })
    
    async def report_interaction(self, interaction: Unity3DInteractionEvent):
        """Report user interaction to cognitive mesh"""
        self.interaction_history.append(interaction)
        
        # Keep history manageable
        if len(self.interaction_history) > 50:
            self.interaction_history = self.interaction_history[-25:]
        
        await self.send_message(Unity3DMessageType.INTERACTION_EVENT, {
            "interaction_type": interaction.interaction_type,
            "object_id": interaction.object_id,
            "user_id": interaction.user_id,
            "interaction_data": interaction.interaction_data,
            "timestamp": interaction.timestamp.isoformat()
        })
    
    async def send_environment_state(self, environment_data: Dict[str, Any]):
        """Send complete environment state for cognitive analysis"""
        await self.send_message(Unity3DMessageType.ENVIRONMENT_STATE, {
            "scene_name": environment_data.get("scene_name", "unknown"),
            "object_count": environment_data.get("object_count", 0),
            "lighting": environment_data.get("lighting", {}),
            "physics_settings": environment_data.get("physics", {}),
            "player_count": environment_data.get("players", 0),
            "timestamp": datetime.now().isoformat()
        })
    
    async def send_camera_data(self, camera_id: str, camera_data: Dict[str, Any]):
        """Send camera/vision data for cognitive processing"""
        await self.send_message(Unity3DMessageType.CAMERA_DATA, {
            "camera_id": camera_id,
            "resolution": camera_data.get("resolution", [1920, 1080]),
            "field_of_view": camera_data.get("fov", 60.0),
            "view_direction": camera_data.get("direction", [0, 0, 1]),
            "visible_objects": camera_data.get("visible_objects", []),
            "depth_data": camera_data.get("depth", None),  # Optional depth buffer
            "timestamp": datetime.now().isoformat()
        })
    
    async def query_cognitive_state(self, query_type: str, parameters: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Query cognitive state from the mesh"""
        query_id = str(uuid.uuid4())
        
        await self.send_message(Unity3DMessageType.COGNITIVE_QUERY, {
            "query_id": query_id,
            "query_type": query_type,
            "parameters": parameters or {},
            "timestamp": datetime.now().isoformat()
        })
        
        # Wait for response (with timeout)
        timeout = 5.0  # seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            message = await self.receive_message()
            if message and message.get("message_type") == "cognitive_response":
                payload = message.get("payload", {})
                if payload.get("query_id") == query_id:
                    return payload
            await asyncio.sleep(0.1)
        
        logger.warning(f"Cognitive query timed out: {query_type}")
        return None
    
    async def process_cognitive_intentions(self) -> List[Unity3DCognitiveIntention]:
        """Process cognitive intentions into Unity3D actions"""
        if not self.intention_queue:
            return []
        
        # Filter intentions by confidence threshold
        actionable_intentions = [
            intention for intention in self.intention_queue
            if intention.cognitive_confidence >= self.action_confidence_threshold
        ]
        
        # Clear processed intentions
        self.intention_queue = []
        
        return actionable_intentions
    
    async def handle_cognitive_response(self, message: Dict[str, Any]):
        """Handle cognitive response from mesh"""
        payload = message.get("payload", {})
        message_type = payload.get("type", "unknown")
        
        if message_type == "intention_action":
            intention = Unity3DCognitiveIntention(
                intention_type=payload.get("intention_type", "observe"),
                target_object=payload.get("target_object"),
                parameters=payload.get("parameters", {}),
                priority=payload.get("priority", 0.5),
                duration=payload.get("duration"),
                cognitive_confidence=payload.get("confidence", 0.8)
            )
            self.intention_queue.append(intention)
            logger.debug(f"Received cognitive intention: {intention.intention_type}")
        
        elif message_type == "state_update":
            self.cognitive_state.update(payload.get("state", {}))
            logger.debug("Updated cognitive state from mesh")
        
        elif message_type == "attention_shift":
            attention_data = payload.get("attention", {})
            await self._process_attention_shift(attention_data)
    
    async def _process_attention_shift(self, attention_data: Dict[str, Any]):
        """Process attention shift from cognitive mesh"""
        focus_object = attention_data.get("focus_object")
        attention_level = attention_data.get("level", 0.5)
        
        if focus_object and attention_level > self.attention_threshold:
            # Request Unity3D to highlight or focus on the object
            await self.send_message(Unity3DMessageType.INTENTION_ACTION, {
                "action_type": "focus_attention",
                "target_object": focus_object,
                "intensity": attention_level,
                "timestamp": datetime.now().isoformat()
            })
    
    # Unity3D communication protocol methods
    
    def create_unity3d_message_handler(self):
        """Create message handler for Unity3D communication"""
        
        async def message_handler():
            """Handle incoming messages from cognitive mesh"""
            while self.is_connected:
                try:
                    message = await self.receive_message()
                    if message:
                        await self.handle_cognitive_response(message)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
                    await asyncio.sleep(1.0)
        
        return message_handler
    
    def get_unity3d_integration_code(self) -> str:
        """Generate Unity3D C# integration code template"""
        return '''
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using WebSocketSharp;
using Newtonsoft.Json;

public class Echo9MLCognitiveMesh : MonoBehaviour
{
    [System.Serializable]
    public class TransformData
    {
        public float[] position;
        public float[] rotation;
        public float[] scale;
        public string timestamp;
    }
    
    [System.Serializable]
    public class CognitiveMessage
    {
        public string message_type;
        public string sender_id;
        public Dictionary<string, object> payload;
        public string timestamp;
    }
    
    private WebSocket ws;
    private string cognitiveAPIURL = "ws://localhost:8000/ws/unity3d_connection";
    private bool isConnected = false;
    
    void Start()
    {
        ConnectToCognitiveMesh();
    }
    
    void ConnectToCognitiveMesh()
    {
        ws = new WebSocket(cognitiveAPIURL);
        
        ws.OnOpen += (sender, e) =>
        {
            Debug.Log("Connected to Cognitive Mesh");
            isConnected = true;
            
            // Send platform identification
            SendCognitiveMessage("cognitive_query", new Dictionary<string, object>
            {
                {"type", "platform_identification"},
                {"platform", "unity3d"},
                {"scene_name", UnityEngine.SceneManagement.SceneManager.GetActiveScene().name}
            });
        };
        
        ws.OnMessage += (sender, e) =>
        {
            HandleCognitiveResponse(e.Data);
        };
        
        ws.OnClose += (sender, e) =>
        {
            Debug.Log("Disconnected from Cognitive Mesh");
            isConnected = false;
        };
        
        ws.Connect();
    }
    
    void SendCognitiveMessage(string messageType, Dictionary<string, object> payload)
    {
        if (!isConnected) return;
        
        var message = new CognitiveMessage
        {
            message_type = messageType,
            sender_id = "unity3d_" + System.Guid.NewGuid().ToString("N")[..8],
            payload = payload,
            timestamp = System.DateTime.Now.ToString("o")
        };
        
        string json = JsonConvert.SerializeObject(message);
        ws.Send(json);
    }
    
    void HandleCognitiveResponse(string messageData)
    {
        try
        {
            var message = JsonConvert.DeserializeObject<CognitiveMessage>(messageData);
            
            switch (message.message_type)
            {
                case "intention_action":
                    ProcessIntentionAction(message.payload);
                    break;
                case "attention_shift":
                    ProcessAttentionShift(message.payload);
                    break;
                default:
                    Debug.Log($"Received cognitive message: {message.message_type}");
                    break;
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error processing cognitive response: {ex.Message}");
        }
    }
    
    void ProcessIntentionAction(Dictionary<string, object> payload)
    {
        string actionType = payload.ContainsKey("action_type") ? payload["action_type"].ToString() : "";
        string targetObject = payload.ContainsKey("target_object") ? payload["target_object"].ToString() : "";
        
        switch (actionType)
        {
            case "focus_attention":
                HighlightObject(targetObject);
                break;
            case "move_to":
                MoveToTarget(targetObject);
                break;
            case "interact_with":
                InteractWithObject(targetObject);
                break;
        }
    }
    
    void ProcessAttentionShift(Dictionary<string, object> payload)
    {
        // Handle attention shift from cognitive mesh
        // Adjust camera, lighting, or object highlighting
    }
    
    public void SendTransformUpdate(GameObject obj)
    {
        var transform = obj.transform;
        var transformData = new TransformData
        {
            position = new float[] { transform.position.x, transform.position.y, transform.position.z },
            rotation = new float[] { transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w },
            scale = new float[] { transform.localScale.x, transform.localScale.y, transform.localScale.z },
            timestamp = System.DateTime.Now.ToString("o")
        };
        
        SendCognitiveMessage("transform_update", new Dictionary<string, object>
        {
            {"object_id", obj.name},
            {"object_type", "game_object"},
            {"position", transformData.position},
            {"rotation", transformData.rotation},
            {"scale", transformData.scale},
            {"timestamp", transformData.timestamp}
        });
    }
    
    public void SendInteractionEvent(string interactionType, GameObject obj, string userId = null)
    {
        SendCognitiveMessage("interaction_event", new Dictionary<string, object>
        {
            {"interaction_type", interactionType},
            {"object_id", obj.name},
            {"user_id", userId},
            {"timestamp", System.DateTime.Now.ToString("o")}
        });
    }
    
    void HighlightObject(string objectName)
    {
        GameObject obj = GameObject.Find(objectName);
        if (obj != null)
        {
            // Add highlighting effect
            Renderer renderer = obj.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material.color = Color.yellow;
            }
        }
    }
    
    void MoveToTarget(string targetName)
    {
        // Implement movement logic
        Debug.Log($"Moving to target: {targetName}");
    }
    
    void InteractWithObject(string objectName)
    {
        // Implement interaction logic
        Debug.Log($"Interacting with: {objectName}");
    }
    
    void OnDestroy()
    {
        if (ws != null)
        {
            ws.Close();
        }
    }
}
'''
    
    def get_unity3d_setup_instructions(self) -> str:
        """Get setup instructions for Unity3D integration"""
        return '''
# Unity3D Echo9ML Cognitive Mesh Integration Setup

## Prerequisites
1. Unity 2021.3 LTS or later
2. Newtonsoft.Json package (Window > Package Manager > Add package by name: com.unity.nuget.newtonsoft-json)
3. WebSocketSharp package (download from NuGet or Unity Package Manager)

## Setup Steps
1. Create a new GameObject in your scene called "CognitiveMesh"
2. Attach the Echo9MLCognitiveMesh script to this GameObject
3. Configure the cognitive API URL in the script inspector
4. Add the following components to objects you want to track:
   - Echo9MLObjectTracker for transform updates
   - Echo9MLInteractionHandler for interaction events

## Usage Examples

### Transform Tracking
```csharp
// Track player movement
public class PlayerController : MonoBehaviour
{
    private Echo9MLCognitiveMesh cognitiveMesh;
    
    void Start()
    {
        cognitiveMesh = FindObjectOfType<Echo9MLCognitiveMesh>();
    }
    
    void Update()
    {
        if (transform.hasChanged)
        {
            cognitiveMesh.SendTransformUpdate(gameObject);
            transform.hasChanged = false;
        }
    }
}
```

### Interaction Handling
```csharp
// Handle object interactions
public class InteractableObject : MonoBehaviour, IPointerClickHandler
{
    private Echo9MLCognitiveMesh cognitiveMesh;
    
    void Start()
    {
        cognitiveMesh = FindObjectOfType<Echo9MLCognitiveMesh>();
    }
    
    public void OnPointerClick(PointerEventData eventData)
    {
        cognitiveMesh.SendInteractionEvent("click", gameObject, "player");
    }
}
```

### Physics Events
```csharp
// Track physics collisions
public class PhysicsTracker : MonoBehaviour
{
    private Echo9MLCognitiveMesh cognitiveMesh;
    
    void Start()
    {
        cognitiveMesh = FindObjectOfType<Echo9MLCognitiveMesh>();
    }
    
    void OnCollisionEnter(Collision collision)
    {
        // Send collision data to cognitive mesh
        var payload = new Dictionary<string, object>
        {
            {"event_type", "collision"},
            {"object_id", gameObject.name},
            {"other_object_id", collision.gameObject.name},
            {"contact_point", new float[] { 
                collision.contacts[0].point.x, 
                collision.contacts[0].point.y, 
                collision.contacts[0].point.z 
            }}
        };
        
        cognitiveMesh.SendCognitiveMessage("physics_event", payload);
    }
}
```

## Configuration
- Set cognitive API URL to match your Echo9ML server
- Adjust message frequency to balance performance and responsiveness
- Configure object filtering to track only relevant game objects
- Set up authentication if required by your cognitive mesh

## Performance Considerations
- Limit transform update frequency for performance
- Use object pooling for frequent message sending
- Implement message batching for high-frequency events
- Consider LOD (Level of Detail) for cognitive attention allocation
'''

def create_unity3d_binding(cognitive_api_url: str = "ws://localhost:8000/ws/unity3d") -> Unity3DBinding:
    """Factory function to create Unity3D binding"""
    return Unity3DBinding(cognitive_api_url=cognitive_api_url)

async def main():
    """Test Unity3D binding"""
    binding = create_unity3d_binding()
    
    # Connect to cognitive mesh
    if await binding.connect_to_cognitive_mesh():
        print("Connected to cognitive mesh")
        
        # Register some test objects
        binding.register_object("player", Unity3DObjectType.PLAYER_CHARACTER)
        binding.register_object("enemy1", Unity3DObjectType.NPC)
        binding.register_object("door", Unity3DObjectType.INTERACTIVE_OBJECT)
        
        # Send some test data
        transform = Unity3DTransform(position=(1.0, 0.0, 5.0))
        await binding.update_transform("player", transform)
        
        animation = Unity3DAnimationState(current_animation="walking", is_playing=True)
        await binding.update_animation("player", animation)
        
        # Query cognitive state
        cognitive_state = await binding.query_cognitive_state("attention_level")
        if cognitive_state:
            print(f"Cognitive state: {cognitive_state}")
        
        # Keep running for a bit to receive messages
        message_handler = binding.create_unity3d_message_handler()
        handler_task = asyncio.create_task(message_handler())
        
        await asyncio.sleep(10)  # Run for 10 seconds
        
        await binding.disconnect()
        handler_task.cancel()
    else:
        print("Failed to connect to cognitive mesh")

if __name__ == "__main__":
    asyncio.run(main())