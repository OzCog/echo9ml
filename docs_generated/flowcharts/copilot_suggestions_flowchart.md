# copilot_suggestions Module Flowchart

```mermaid
graph TD
    copilot_suggestions[copilot_suggestions]
    copilot_suggestions_fetch_suggestions_from_azure_openai[fetch_suggestions_from_azure_openai()]
    copilot_suggestions --> copilot_suggestions_fetch_suggestions_from_azure_openai
    copilot_suggestions_update_note_with_suggestions[update_note_with_suggestions()]
    copilot_suggestions --> copilot_suggestions_update_note_with_suggestions
    copilot_suggestions_main[main()]
    copilot_suggestions --> copilot_suggestions_main
    style copilot_suggestions fill:#ffcc99
```