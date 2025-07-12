# ros_binding Module Flowchart

```mermaid
graph TD
    ros_binding[ros_binding]
    ros_binding_ROSMessageType[ROSMessageType]
    ros_binding --> ros_binding_ROSMessageType
    ros_binding_ROSCognitiveIntention[ROSCognitiveIntention]
    ros_binding --> ros_binding_ROSCognitiveIntention
    ros_binding_ROSRobotState[ROSRobotState]
    ros_binding --> ros_binding_ROSRobotState
    ros_binding_ROSSensorReading[ROSSensorReading]
    ros_binding --> ros_binding_ROSSensorReading
    ros_binding_ROSNavigationGoal[ROSNavigationGoal]
    ros_binding --> ros_binding_ROSNavigationGoal
    ros_binding_ROSBinding[ROSBinding]
    ros_binding --> ros_binding_ROSBinding
    ros_binding_ROSBinding___init__[__init__()]
    ros_binding_ROSBinding --> ros_binding_ROSBinding___init__
    ros_binding_ROSBinding_initialize_ros_node[initialize_ros_node()]
    ros_binding_ROSBinding --> ros_binding_ROSBinding_initialize_ros_node
    ros_binding_ROSBinding__setup_publishers[_setup_publishers()]
    ros_binding_ROSBinding --> ros_binding_ROSBinding__setup_publishers
    ros_binding_ROSBinding__setup_subscribers[_setup_subscribers()]
    ros_binding_ROSBinding --> ros_binding_ROSBinding__setup_subscribers
    ros_binding_ROSBinding__setup_services[_setup_services()]
    ros_binding_ROSBinding --> ros_binding_ROSBinding__setup_services
    ros_binding_create_ros_binding[create_ros_binding()]
    ros_binding --> ros_binding_create_ros_binding
    style ros_binding fill:#ffcc99
```