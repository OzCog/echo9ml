# validate_cognitive_integration Module Flowchart

```mermaid
graph TD
    validate_cognitive_integration[validate_cognitive_integration]
    validate_cognitive_integration_validate_workflow_file[validate_workflow_file()]
    validate_cognitive_integration --> validate_cognitive_integration_validate_workflow_file
    validate_cognitive_integration_validate_orchestrator_script[validate_orchestrator_script()]
    validate_cognitive_integration --> validate_cognitive_integration_validate_orchestrator_script
    validate_cognitive_integration_validate_cognitive_components[validate_cognitive_components()]
    validate_cognitive_integration --> validate_cognitive_integration_validate_cognitive_components
    validate_cognitive_integration_validate_issue_template[validate_issue_template()]
    validate_cognitive_integration --> validate_cognitive_integration_validate_issue_template
    validate_cognitive_integration_validate_documentation[validate_documentation()]
    validate_cognitive_integration --> validate_cognitive_integration_validate_documentation
    style validate_cognitive_integration fill:#99ccff
```