# chat_session_manager Module Flowchart

```mermaid
graph TD
    chat_session_manager[chat_session_manager]
    chat_session_manager_ChatPlatform[ChatPlatform]
    chat_session_manager --> chat_session_manager_ChatPlatform
    chat_session_manager_SessionStatus[SessionStatus]
    chat_session_manager --> chat_session_manager_SessionStatus
    chat_session_manager_ChatMessage[ChatMessage]
    chat_session_manager --> chat_session_manager_ChatMessage
    chat_session_manager_ChatMessage_to_dict[to_dict()]
    chat_session_manager_ChatMessage --> chat_session_manager_ChatMessage_to_dict
    chat_session_manager_ChatMessage_from_dict[from_dict()]
    chat_session_manager_ChatMessage --> chat_session_manager_ChatMessage_from_dict
    chat_session_manager_ChatSession[ChatSession]
    chat_session_manager --> chat_session_manager_ChatSession
    chat_session_manager_ChatSession_to_dict[to_dict()]
    chat_session_manager_ChatSession --> chat_session_manager_ChatSession_to_dict
    chat_session_manager_ChatSession_from_dict[from_dict()]
    chat_session_manager_ChatSession --> chat_session_manager_ChatSession_from_dict
    chat_session_manager_ChatSession_add_message[add_message()]
    chat_session_manager_ChatSession --> chat_session_manager_ChatSession_add_message
    chat_session_manager_ChatSession__update_statistics[_update_statistics()]
    chat_session_manager_ChatSession --> chat_session_manager_ChatSession__update_statistics
    chat_session_manager_ChatSessionManager[ChatSessionManager]
    chat_session_manager --> chat_session_manager_ChatSessionManager
    chat_session_manager_ChatSessionManager___init__[__init__()]
    chat_session_manager_ChatSessionManager --> chat_session_manager_ChatSessionManager___init__
    chat_session_manager_ChatSessionManager_start_auto_save[start_auto_save()]
    chat_session_manager_ChatSessionManager --> chat_session_manager_ChatSessionManager_start_auto_save
    chat_session_manager_ChatSessionManager_stop_auto_save[stop_auto_save()]
    chat_session_manager_ChatSessionManager --> chat_session_manager_ChatSessionManager_stop_auto_save
    chat_session_manager_ChatSessionManager__auto_save_loop[_auto_save_loop()]
    chat_session_manager_ChatSessionManager --> chat_session_manager_ChatSessionManager__auto_save_loop
    chat_session_manager_ChatSessionManager_create_session[create_session()]
    chat_session_manager_ChatSessionManager --> chat_session_manager_ChatSessionManager_create_session
    chat_session_manager_initialize_session_manager[initialize_session_manager()]
    chat_session_manager --> chat_session_manager_initialize_session_manager
    chat_session_manager_create_chat_session[create_chat_session()]
    chat_session_manager --> chat_session_manager_create_chat_session
    chat_session_manager_log_chat_message[log_chat_message()]
    chat_session_manager --> chat_session_manager_log_chat_message
    chat_session_manager_end_chat_session[end_chat_session()]
    chat_session_manager --> chat_session_manager_end_chat_session
    chat_session_manager_get_chat_history[get_chat_history()]
    chat_session_manager --> chat_session_manager_get_chat_history
```