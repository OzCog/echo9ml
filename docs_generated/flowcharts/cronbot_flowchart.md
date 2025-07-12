# cronbot Module Flowchart

```mermaid
graph TD
    cronbot[cronbot]
    cronbot_read_note[read_note()]
    cronbot --> cronbot_read_note
    cronbot_write_note[write_note()]
    cronbot --> cronbot_write_note
    cronbot_call_github_copilot[call_github_copilot()]
    cronbot --> cronbot_call_github_copilot
    cronbot_introspect_repo[introspect_repo()]
    cronbot --> cronbot_introspect_repo
    cronbot_apply_improvement[apply_improvement()]
    cronbot --> cronbot_apply_improvement
```