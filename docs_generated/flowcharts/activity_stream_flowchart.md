# activity_stream Module Flowchart

```mermaid
graph TD
    activity_stream[activity_stream]
    activity_stream_StreamType[StreamType]
    activity_stream --> activity_stream_StreamType
    activity_stream_ActivityStream[ActivityStream]
    activity_stream --> activity_stream_ActivityStream
    activity_stream_ActivityStream___init__[__init__()]
    activity_stream_ActivityStream --> activity_stream_ActivityStream___init__
    activity_stream_ActivityStream__screen_state_changed[_screen_state_changed()]
    activity_stream_ActivityStream --> activity_stream_ActivityStream__screen_state_changed
    activity_stream_ActivityStream_update_activities[update_activities()]
    activity_stream_ActivityStream --> activity_stream_ActivityStream_update_activities
    activity_stream_ActivityStream_update_system_stats[update_system_stats()]
    activity_stream_ActivityStream --> activity_stream_ActivityStream_update_system_stats
    activity_stream_ActivityStream_get_activity_color[get_activity_color()]
    activity_stream_ActivityStream --> activity_stream_ActivityStream_get_activity_color
    activity_stream_main[main()]
    activity_stream --> activity_stream_main
```