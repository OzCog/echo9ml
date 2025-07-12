#!/usr/bin/env python3
"""
ROS (Robot Operating System) Embodiment Binding for Echo9ML Cognitive Mesh

This module provides ROS integration for embodied cognition, enabling
real-time communication between ROS-based robotic systems and the
distributed cognitive grammar network.

Key Features:
- ROS message integration for cognitive communication
- Navigation and path planning cognitive interface
- Sensor data processing and cognitive interpretation
- Action server integration for cognitive intentions
- Multi-robot coordination through cognitive mesh
"""

import asyncio
import json
import logging
import time
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# ROS imports (mock implementation if ROS not available)
try:
    import rospy
    from std_msgs.msg import String, Float64, Bool, Header
    from geometry_msgs.msg import Twist, Pose, PoseStamped, Transform, Vector3, Quaternion
    from sensor_msgs.msg import LaserScan, Image, PointCloud2, Joy, CompressedImage
    from nav_msgs.msg import OccupancyGrid, Odometry, Path
    from actionlib_msgs.msg import GoalStatus
    from tf2_msgs.msg import TFMessage
    import tf2_ros
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logging.warning("ROS not available - using mock implementation")
    
    # Mock ROS classes for development without ROS
    class rospy:
        @staticmethod
        def init_node(name): pass
        @staticmethod
        def Publisher(topic, msg_type, queue_size=10): return MockPublisher()
        @staticmethod
        def Subscriber(topic, msg_type, callback): return MockSubscriber()
        @staticmethod
        def Service(service, msg_type, callback): return MockService()
        @staticmethod
        def Rate(hz): return MockRate()
        @staticmethod
        def is_shutdown(): return False
        @staticmethod
        def loginfo(msg): logging.info(msg)
        @staticmethod
        def logwarn(msg): logging.warning(msg)
        @staticmethod
        def logerr(msg): logging.error(msg)
    
    class MockPublisher:
        def publish(self, msg): pass
    
    class MockSubscriber:
        pass
    
    class MockService:
        pass
    
    class MockRate:
        def sleep(self): time.sleep(0.1)

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("WebSockets not available")

logger = logging.getLogger(__name__)

class ROSMessageType(Enum):
    """ROS message types for cognitive communication"""
    NAVIGATION_GOAL = "navigation_goal"
    SENSOR_DATA = "sensor_data"
    ROBOT_STATE = "robot_state"
    COGNITIVE_COMMAND = "cognitive_command"
    COGNITIVE_RESPONSE = "cognitive_response"
    MULTI_ROBOT_COORDINATION = "multi_robot_coordination"
    ENVIRONMENT_MAP = "environment_map"
    OBSTACLE_DETECTION = "obstacle_detection"
    MISSION_STATUS = "mission_status"

class ROSCognitiveIntention(Enum):
    """Cognitive intentions that can be executed by ROS systems"""
    NAVIGATE_TO_POINT = "navigate_to_point"
    EXPLORE_AREA = "explore_area"
    FOLLOW_PATH = "follow_path"
    AVOID_OBSTACLE = "avoid_obstacle"
    SEARCH_OBJECT = "search_object"
    COORDINATE_WITH_ROBOT = "coordinate_with_robot"
    STOP_MISSION = "stop_mission"
    RETURN_HOME = "return_home"

@dataclass
class ROSRobotState:
    """ROS robot state information"""
    robot_id: str
    pose: Tuple[float, float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0, 1)  # x,y,z,qx,qy,qz,qw
    velocity: Tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)  # linear and angular
    battery_level: float = 1.0
    mission_status: str = "idle"
    current_goal: Optional[str] = None
    obstacles_detected: List[Tuple[float, float, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ROSSensorReading:
    """ROS sensor data structure"""
    sensor_type: str  # laser, camera, imu, gps, etc.
    sensor_id: str
    data: Dict[str, Any]
    frame_id: str = "base_link"
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ROSNavigationGoal:
    """ROS navigation goal with cognitive context"""
    goal_id: str
    target_pose: Tuple[float, float, float, float, float, float, float]  # x,y,z,qx,qy,qz,qw
    cognitive_priority: float = 0.5
    cognitive_context: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    approach_tolerance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

class ROSBinding:
    """ROS embodiment binding for cognitive mesh"""
    
    def __init__(self, 
                 robot_name: str = "echo9ml_robot",
                 cognitive_api_url: str = "ws://localhost:8000/ws/ros",
                 ros_namespace: str = "/echo9ml"):
        self.robot_name = robot_name
        self.cognitive_api_url = cognitive_api_url
        self.ros_namespace = ros_namespace
        self.connection_id = f"ros_{robot_name}_{str(uuid.uuid4())[:8]}"
        
        # ROS components
        self.node_initialized = False
        self.publishers = {}
        self.subscribers = {}
        self.services = {}
        
        # Cognitive mesh connection
        self.websocket = None
        self.is_connected = False
        self.cognitive_thread = None
        
        # Robot state
        self.robot_state = ROSRobotState(robot_id=robot_name)
        self.sensor_readings = {}
        self.navigation_goals = []
        self.cognitive_intentions = []
        
        # Cognitive parameters
        self.exploration_radius = 10.0  # meters
        self.obstacle_threshold = 1.0  # meters
        self.coordination_range = 50.0  # meters
        self.attention_decay = 0.95
        
        logger.info(f"ROS binding initialized for robot: {robot_name}")
    
    def initialize_ros_node(self):
        """Initialize ROS node and components"""
        if not ROS_AVAILABLE:
            logger.warning("ROS not available - using mock mode")
            return True
        
        try:
            rospy.init_node(f"echo9ml_cognitive_binding_{self.robot_name}")
            self.node_initialized = True
            
            # Initialize publishers
            self._setup_publishers()
            
            # Initialize subscribers
            self._setup_subscribers()
            
            # Initialize services
            self._setup_services()
            
            logger.info(f"ROS node initialized: echo9ml_cognitive_binding_{self.robot_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ROS node: {e}")
            return False
    
    def _setup_publishers(self):
        """Setup ROS publishers"""
        self.publishers = {
            'cmd_vel': rospy.Publisher(f'{self.ros_namespace}/cmd_vel', Twist, queue_size=10),
            'cognitive_cmd': rospy.Publisher(f'{self.ros_namespace}/cognitive_command', String, queue_size=10),
            'robot_state': rospy.Publisher(f'{self.ros_namespace}/robot_state', String, queue_size=10),
            'cognitive_intention': rospy.Publisher(f'{self.ros_namespace}/cognitive_intention', String, queue_size=10)
        }
    
    def _setup_subscribers(self):
        """Setup ROS subscribers"""
        self.subscribers = {
            'odom': rospy.Subscriber('/odom', Odometry, self._odometry_callback),
            'scan': rospy.Subscriber('/scan', LaserScan, self._laser_callback),
            'camera': rospy.Subscriber('/camera/image_raw/compressed', CompressedImage, self._camera_callback),
            'goal_status': rospy.Subscriber('/move_base/status', GoalStatus, self._goal_status_callback)
        }
    
    def _setup_services(self):
        """Setup ROS services"""
        # Services would be set up here for more complex interactions
        pass
    
    def _odometry_callback(self, msg):
        """Handle odometry updates"""
        pose = msg.pose.pose
        twist = msg.twist.twist
        
        self.robot_state.pose = (
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        )
        
        self.robot_state.velocity = (
            twist.linear.x, twist.linear.y, twist.linear.z,
            twist.angular.x, twist.angular.y, twist.angular.z
        )
        
        self.robot_state.timestamp = datetime.now()
        
        # Send to cognitive mesh
        asyncio.create_task(self._send_robot_state_update())
    
    def _laser_callback(self, msg):
        """Handle laser scan data"""
        # Process laser scan for obstacle detection
        ranges = msg.ranges
        min_range = min([r for r in ranges if r > msg.range_min and r < msg.range_max])
        
        if min_range < self.obstacle_threshold:
            # Obstacle detected - update robot state
            angle_increment = msg.angle_increment
            min_index = ranges.index(min_range)
            obstacle_angle = msg.angle_min + (min_index * angle_increment)
            
            # Calculate obstacle position relative to robot
            obstacle_x = min_range * np.cos(obstacle_angle)
            obstacle_y = min_range * np.sin(obstacle_angle)
            
            self.robot_state.obstacles_detected = [(obstacle_x, obstacle_y, 0.0)]
        else:
            self.robot_state.obstacles_detected = []
        
        # Create sensor reading
        sensor_reading = ROSSensorReading(
            sensor_type="laser",
            sensor_id="base_scan",
            data={
                "ranges": list(ranges),
                "angle_min": msg.angle_min,
                "angle_max": msg.angle_max,
                "angle_increment": msg.angle_increment,
                "range_min": msg.range_min,
                "range_max": msg.range_max,
                "obstacles_detected": len(self.robot_state.obstacles_detected) > 0
            },
            frame_id=msg.header.frame_id
        )
        
        self.sensor_readings["laser"] = sensor_reading
        
        # Send to cognitive mesh
        asyncio.create_task(self._send_sensor_data(sensor_reading))
    
    def _camera_callback(self, msg):
        """Handle camera data"""
        # Process camera image for visual cognition
        sensor_reading = ROSSensorReading(
            sensor_type="camera",
            sensor_id="main_camera",
            data={
                "image_size": len(msg.data),
                "encoding": msg.format,
                "timestamp": msg.header.stamp.to_sec() if hasattr(msg.header.stamp, 'to_sec') else time.time()
            },
            frame_id=msg.header.frame_id
        )
        
        self.sensor_readings["camera"] = sensor_reading
        
        # Send to cognitive mesh (without actual image data for now)
        asyncio.create_task(self._send_sensor_data(sensor_reading))
    
    def _goal_status_callback(self, msg):
        """Handle navigation goal status updates"""
        # Update mission status based on goal status
        status_map = {
            0: "pending",
            1: "active", 
            2: "preempted",
            3: "succeeded",
            4: "aborted",
            5: "rejected",
            6: "preempting",
            7: "recalling",
            8: "recalled",
            9: "lost"
        }
        
        if hasattr(msg, 'status'):
            self.robot_state.mission_status = status_map.get(msg.status, "unknown")
        
        # Send mission status update to cognitive mesh
        asyncio.create_task(self._send_mission_status_update())
    
    async def connect_to_cognitive_mesh(self):
        """Connect to the cognitive mesh API"""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets not available - cannot connect to cognitive mesh")
            return False
        
        try:
            self.websocket = await websockets.connect(
                f"{self.cognitive_api_url.replace('/ros', '')}/{self.connection_id}"
            )
            self.is_connected = True
            logger.info("Connected to cognitive mesh")
            
            # Send platform identification
            await self.send_cognitive_message(ROSMessageType.ROBOT_STATE, {
                "type": "platform_identification",
                "platform": "ros",
                "robot_name": self.robot_name,
                "capabilities": [
                    "navigation",
                    "laser_scanning",
                    "camera_vision",
                    "odometry",
                    "multi_robot_coordination"
                ],
                "robot_state": self._robot_state_to_dict()
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
    
    async def send_cognitive_message(self, message_type: ROSMessageType, payload: Dict[str, Any]):
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
            logger.error(f"Failed to send cognitive message: {e}")
            return False
    
    async def receive_cognitive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from cognitive mesh"""
        if not self.is_connected or not self.websocket:
            return None
        
        try:
            message_raw = await self.websocket.recv()
            message = json.loads(message_raw)
            return message
        except Exception as e:
            logger.error(f"Failed to receive cognitive message: {e}")
            return None
    
    async def _send_robot_state_update(self):
        """Send robot state update to cognitive mesh"""
        await self.send_cognitive_message(ROSMessageType.ROBOT_STATE, {
            "robot_state": self._robot_state_to_dict(),
            "timestamp": datetime.now().isoformat()
        })
    
    async def _send_sensor_data(self, sensor_reading: ROSSensorReading):
        """Send sensor data to cognitive mesh"""
        await self.send_cognitive_message(ROSMessageType.SENSOR_DATA, {
            "sensor_type": sensor_reading.sensor_type,
            "sensor_id": sensor_reading.sensor_id,
            "data": sensor_reading.data,
            "frame_id": sensor_reading.frame_id,
            "timestamp": sensor_reading.timestamp.isoformat()
        })
    
    async def _send_mission_status_update(self):
        """Send mission status update to cognitive mesh"""
        await self.send_cognitive_message(ROSMessageType.MISSION_STATUS, {
            "robot_id": self.robot_name,
            "mission_status": self.robot_state.mission_status,
            "current_goal": self.robot_state.current_goal,
            "timestamp": datetime.now().isoformat()
        })
    
    def _robot_state_to_dict(self) -> Dict[str, Any]:
        """Convert robot state to dictionary"""
        return {
            "robot_id": self.robot_state.robot_id,
            "pose": {
                "position": {"x": self.robot_state.pose[0], "y": self.robot_state.pose[1], "z": self.robot_state.pose[2]},
                "orientation": {"x": self.robot_state.pose[3], "y": self.robot_state.pose[4], 
                              "z": self.robot_state.pose[5], "w": self.robot_state.pose[6]}
            },
            "velocity": {
                "linear": {"x": self.robot_state.velocity[0], "y": self.robot_state.velocity[1], "z": self.robot_state.velocity[2]},
                "angular": {"x": self.robot_state.velocity[3], "y": self.robot_state.velocity[4], "z": self.robot_state.velocity[5]}
            },
            "battery_level": self.robot_state.battery_level,
            "mission_status": self.robot_state.mission_status,
            "current_goal": self.robot_state.current_goal,
            "obstacles_detected": self.robot_state.obstacles_detected,
            "timestamp": self.robot_state.timestamp.isoformat()
        }
    
    # ROS-specific cognitive interface methods
    
    async def execute_navigation_goal(self, goal: ROSNavigationGoal):
        """Execute navigation goal from cognitive intention"""
        self.navigation_goals.append(goal)
        self.robot_state.current_goal = goal.goal_id
        
        # Publish navigation goal (simplified - real implementation would use move_base)
        if ROS_AVAILABLE and 'cmd_vel' in self.publishers:
            # Calculate direction to goal
            current_pos = self.robot_state.pose[:2]
            goal_pos = goal.target_pose[:2]
            
            # Simple proportional navigation (real implementation would use path planning)
            direction = [goal_pos[0] - current_pos[0], goal_pos[1] - current_pos[1]]
            distance = (direction[0]**2 + direction[1]**2)**0.5
            
            if distance > goal.approach_tolerance:
                # Create Twist message for movement
                cmd_vel = Twist()
                cmd_vel.linear.x = min(0.5, distance * 0.1)  # Proportional speed
                cmd_vel.angular.z = direction[1] / max(abs(direction[0]), 0.1) * 0.5  # Simple turning
                
                self.publishers['cmd_vel'].publish(cmd_vel)
                
                logger.info(f"Executing navigation to goal: {goal.goal_id}")
            else:
                logger.info(f"Reached navigation goal: {goal.goal_id}")
                self.robot_state.mission_status = "goal_reached"
    
    async def process_cognitive_intention(self, intention_type: str, parameters: Dict[str, Any]):
        """Process cognitive intention and execute corresponding ROS actions"""
        intention_enum = None
        try:
            intention_enum = ROSCognitiveIntention(intention_type)
        except ValueError:
            logger.warning(f"Unknown cognitive intention: {intention_type}")
            return
        
        if intention_enum == ROSCognitiveIntention.NAVIGATE_TO_POINT:
            target = parameters.get("target", [0, 0, 0, 0, 0, 0, 1])
            goal = ROSNavigationGoal(
                goal_id=str(uuid.uuid4()),
                target_pose=tuple(target),
                cognitive_priority=parameters.get("priority", 0.5),
                cognitive_context=parameters.get("context", {})
            )
            await self.execute_navigation_goal(goal)
        
        elif intention_enum == ROSCognitiveIntention.EXPLORE_AREA:
            await self._execute_exploration(parameters)
        
        elif intention_enum == ROSCognitiveIntention.AVOID_OBSTACLE:
            await self._execute_obstacle_avoidance(parameters)
        
        elif intention_enum == ROSCognitiveIntention.STOP_MISSION:
            await self._stop_current_mission()
        
        else:
            logger.info(f"Cognitive intention not implemented: {intention_type}")
    
    async def _execute_exploration(self, parameters: Dict[str, Any]):
        """Execute exploration behavior"""
        exploration_radius = parameters.get("radius", self.exploration_radius)
        
        # Simple exploration: move to random points within radius
        import random
        current_pos = self.robot_state.pose[:2]
        
        # Generate random exploration point
        angle = random.uniform(0, 2 * 3.14159)
        distance = random.uniform(2.0, exploration_radius)
        target_x = current_pos[0] + distance * np.cos(angle)
        target_y = current_pos[1] + distance * np.sin(angle)
        
        goal = ROSNavigationGoal(
            goal_id=f"explore_{int(time.time())}",
            target_pose=(target_x, target_y, 0, 0, 0, 0, 1),
            cognitive_priority=0.6,
            cognitive_context={"behavior": "exploration", "radius": exploration_radius}
        )
        
        await self.execute_navigation_goal(goal)
        logger.info(f"Exploring area: target ({target_x:.2f}, {target_y:.2f})")
    
    async def _execute_obstacle_avoidance(self, parameters: Dict[str, Any]):
        """Execute obstacle avoidance behavior"""
        if not self.robot_state.obstacles_detected:
            return
        
        # Simple avoidance: stop and turn
        if ROS_AVAILABLE and 'cmd_vel' in self.publishers:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0  # Stop forward movement
            cmd_vel.angular.z = 0.5  # Turn to avoid
            
            self.publishers['cmd_vel'].publish(cmd_vel)
            logger.info("Executing obstacle avoidance maneuver")
    
    async def _stop_current_mission(self):
        """Stop current mission and return to idle"""
        if ROS_AVAILABLE and 'cmd_vel' in self.publishers:
            cmd_vel = Twist()  # Zero velocity
            self.publishers['cmd_vel'].publish(cmd_vel)
        
        self.robot_state.mission_status = "stopped"
        self.robot_state.current_goal = None
        self.navigation_goals.clear()
        
        logger.info("Mission stopped by cognitive intention")
    
    async def handle_cognitive_response(self, message: Dict[str, Any]):
        """Handle cognitive response from mesh"""
        payload = message.get("payload", {})
        message_type = payload.get("type", "unknown")
        
        if message_type == "cognitive_intention":
            intention_type = payload.get("intention_type")
            parameters = payload.get("parameters", {})
            await self.process_cognitive_intention(intention_type, parameters)
        
        elif message_type == "multi_robot_coordination":
            await self._handle_multi_robot_coordination(payload)
        
        elif message_type == "environment_update":
            await self._handle_environment_update(payload)
        
        else:
            logger.debug(f"Unhandled cognitive response type: {message_type}")
    
    async def _handle_multi_robot_coordination(self, payload: Dict[str, Any]):
        """Handle multi-robot coordination messages"""
        coordination_type = payload.get("coordination_type", "")
        target_robot = payload.get("target_robot", "")
        
        if coordination_type == "formation":
            # Implement formation flying/driving
            formation_data = payload.get("formation_data", {})
            logger.info(f"Forming coordination with {target_robot}: {formation_data}")
        
        elif coordination_type == "task_handoff":
            # Handle task handoff from another robot
            task_data = payload.get("task_data", {})
            logger.info(f"Receiving task handoff from {target_robot}: {task_data}")
    
    async def _handle_environment_update(self, payload: Dict[str, Any]):
        """Handle environment update from cognitive mesh"""
        environment_data = payload.get("environment", {})
        
        # Update local environment model
        if "obstacles" in environment_data:
            # Process new obstacle information
            obstacles = environment_data["obstacles"]
            logger.info(f"Environment update: {len(obstacles)} obstacles detected")
        
        if "points_of_interest" in environment_data:
            # Process points of interest for exploration
            poi_list = environment_data["points_of_interest"]
            logger.info(f"Environment update: {len(poi_list)} points of interest")
    
    def create_ros_message_handler(self):
        """Create message handler for cognitive mesh communication"""
        
        async def message_handler():
            """Handle incoming messages from cognitive mesh"""
            while self.is_connected:
                try:
                    message = await self.receive_cognitive_message()
                    if message:
                        await self.handle_cognitive_response(message)
                except Exception as e:
                    logger.error(f"Error in ROS message handler: {e}")
                    await asyncio.sleep(1.0)
        
        return message_handler
    
    def get_ros_launch_file(self) -> str:
        """Generate ROS launch file for Echo9ML integration"""
        return f'''<?xml version="1.0"?>
<launch>
    <!-- Echo9ML Cognitive Mesh ROS Integration -->
    
    <!-- Robot parameters -->
    <param name="robot_name" value="{self.robot_name}" />
    <param name="cognitive_api_url" value="{self.cognitive_api_url}" />
    <param name="ros_namespace" value="{self.ros_namespace}" />
    
    <!-- Cognitive binding node -->
    <node name="echo9ml_cognitive_binding" pkg="echo9ml_ros" type="ros_binding.py" output="screen">
        <param name="robot_name" value="{self.robot_name}" />
        <param name="cognitive_api_url" value="{self.cognitive_api_url}" />
    </node>
    
    <!-- Navigation and mapping -->
    <include file="$(find turtlebot3_navigation)/launch/turtlebot3_navigation.launch" if="$(arg use_navigation)">
        <arg name="map_file" value="$(arg map_file)" />
    </include>
    
    <!-- Sensor processing -->
    <node name="laser_processor" pkg="echo9ml_ros" type="laser_processor.py" output="screen">
        <remap from="scan" to="/scan" />
        <remap from="cognitive_scan" to="{self.ros_namespace}/cognitive_scan" />
    </node>
    
    <node name="camera_processor" pkg="echo9ml_ros" type="camera_processor.py" output="screen">
        <remap from="image_raw" to="/camera/image_raw" />
        <remap from="cognitive_vision" to="{self.ros_namespace}/cognitive_vision" />
    </node>
    
    <!-- Multi-robot coordination -->
    <node name="multi_robot_coordinator" pkg="echo9ml_ros" type="multi_robot_coordinator.py" output="screen">
        <param name="coordination_range" value="50.0" />
        <param name="coordination_topic" value="/echo9ml/coordination" />
    </node>
    
    <!-- Cognitive intention executor -->
    <node name="intention_executor" pkg="echo9ml_ros" type="intention_executor.py" output="screen">
        <remap from="cmd_vel" to="/cmd_vel" />
        <remap from="cognitive_intentions" to="{self.ros_namespace}/cognitive_intentions" />
    </node>
    
</launch>'''
    
    def get_ros_package_setup(self) -> str:
        """Generate ROS package setup instructions"""
        return f'''# Echo9ML ROS Package Setup Instructions

## Prerequisites
- ROS Noetic (recommended) or ROS Melodic
- Python 3.8+
- echo9ml cognitive mesh API server running

## Installation Steps

1. Create ROS workspace (if not exists):
```bash
mkdir -p ~/echo9ml_ws/src
cd ~/echo9ml_ws/src
```

2. Create echo9ml_ros package:
```bash
catkin_create_pkg echo9ml_ros rospy std_msgs geometry_msgs sensor_msgs nav_msgs
```

3. Copy integration files:
```bash
cp ros_binding.py ~/echo9ml_ws/src/echo9ml_ros/scripts/
cp *.launch ~/echo9ml_ws/src/echo9ml_ros/launch/
```

4. Make scripts executable:
```bash
chmod +x ~/echo9ml_ws/src/echo9ml_ros/scripts/*.py
```

5. Build workspace:
```bash
cd ~/echo9ml_ws
catkin_make
source devel/setup.bash
```

## Usage

1. Start Echo9ML cognitive mesh API:
```bash
python cognitive_mesh_api.py --host 0.0.0.0 --port 8000
```

2. Launch ROS integration:
```bash
roslaunch echo9ml_ros echo9ml_integration.launch robot_name:={self.robot_name}
```

3. Test cognitive commands:
```bash
# Navigate to point
rostopic pub /echo9ml/cognitive_command std_msgs/String "data: '{{\\"intention\\": \\"navigate_to_point\\", \\"target\\": [2.0, 3.0, 0.0]}}"

# Start exploration
rostopic pub /echo9ml/cognitive_command std_msgs/String "data: '{{\\"intention\\": \\"explore_area\\", \\"radius\\": 10.0}}"

# Stop mission
rostopic pub /echo9ml/cognitive_command std_msgs/String "data: '{{\\"intention\\": \\"stop_mission\\"}}"
```

## Integration with Existing ROS Systems

### TurtleBot3 Integration:
```bash
export TURTLEBOT3_MODEL=burger
roslaunch echo9ml_ros echo9ml_integration.launch use_navigation:=true map_file:=/path/to/map.yaml
```

### Custom Robot Integration:
1. Modify topic remappings in launch file
2. Adjust sensor processing nodes for your robot's sensors
3. Configure navigation parameters for your robot's capabilities

## Multi-Robot Coordination:
```bash
# Robot 1
roslaunch echo9ml_ros echo9ml_integration.launch robot_name:=robot1 cognitive_api_url:=ws://192.168.1.100:8000/ws/ros

# Robot 2
roslaunch echo9ml_ros echo9ml_integration.launch robot_name:=robot2 cognitive_api_url:=ws://192.168.1.100:8000/ws/ros
```

## Troubleshooting

1. Connection issues:
   - Check cognitive mesh API server is running
   - Verify network connectivity
   - Check firewall settings

2. Navigation issues:
   - Ensure move_base is properly configured
   - Check map and localization
   - Verify coordinate frame transformations

3. Sensor issues:
   - Check sensor topic names and types
   - Verify sensor drivers are running
   - Check frame_id consistency
'''

def create_ros_binding(robot_name: str = "echo9ml_robot",
                      cognitive_api_url: str = "ws://localhost:8000/ws/ros",
                      ros_namespace: str = "/echo9ml") -> ROSBinding:
    """Factory function to create ROS binding"""
    return ROSBinding(robot_name=robot_name, 
                     cognitive_api_url=cognitive_api_url,
                     ros_namespace=ros_namespace)

async def main():
    """Test ROS binding"""
    # Note: numpy import for mathematical operations
    global np
    try:
        import numpy as np
    except ImportError:
        # Simple math fallback
        class np:
            @staticmethod
            def cos(x): return math.cos(x)
            @staticmethod
            def sin(x): return math.sin(x)
        import math
    
    binding = create_ros_binding(robot_name="test_robot")
    
    # Initialize ROS node
    if binding.initialize_ros_node():
        print("ROS node initialized")
        
        # Connect to cognitive mesh
        if await binding.connect_to_cognitive_mesh():
            print("Connected to cognitive mesh")
            
            # Simulate some robot updates
            binding.robot_state.pose = (1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            binding.robot_state.battery_level = 0.85
            binding.robot_state.mission_status = "exploring"
            
            await binding._send_robot_state_update()
            
            # Simulate sensor data
            laser_reading = ROSSensorReading(
                sensor_type="laser",
                sensor_id="base_scan",
                data={"ranges": [1.0, 1.5, 2.0, 1.2], "obstacles_detected": False}
            )
            await binding._send_sensor_data(laser_reading)
            
            # Test cognitive intention
            await binding.process_cognitive_intention("navigate_to_point", 
                                                    {"target": [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0]})
            
            # Keep running for a bit to receive messages
            message_handler = binding.create_ros_message_handler()
            handler_task = asyncio.create_task(message_handler())
            
            await asyncio.sleep(10)  # Run for 10 seconds
            
            await binding.disconnect()
            handler_task.cancel()
        else:
            print("Failed to connect to cognitive mesh")
    else:
        print("Failed to initialize ROS node")

if __name__ == "__main__":
    asyncio.run(main())