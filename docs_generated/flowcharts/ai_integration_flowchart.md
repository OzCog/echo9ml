# ai_integration Module Flowchart

```mermaid
graph TD
    ai_integration[ai_integration]
    ai_integration_AIService[AIService]
    ai_integration --> ai_integration_AIService
    ai_integration_AIService___init__[__init__()]
    ai_integration_AIService --> ai_integration_AIService___init__
    ai_integration_AIService__check_rate_limit[_check_rate_limit()]
    ai_integration_AIService --> ai_integration_AIService__check_rate_limit
    ai_integration_AIService__record_usage[_record_usage()]
    ai_integration_AIService --> ai_integration_AIService__record_usage
    ai_integration_OpenAIService[OpenAIService]
    ai_integration --> ai_integration_OpenAIService
    ai_integration_OpenAIService___init__[__init__()]
    ai_integration_OpenAIService --> ai_integration_OpenAIService___init__
    ai_integration_OpenAIService_generate_embedding[generate_embedding()]
    ai_integration_OpenAIService --> ai_integration_OpenAIService_generate_embedding
    ai_integration_OpenAIService_complete_text[complete_text()]
    ai_integration_OpenAIService --> ai_integration_OpenAIService_complete_text
    ai_integration_AnthropicService[AnthropicService]
    ai_integration --> ai_integration_AnthropicService
    ai_integration_AnthropicService___init__[__init__()]
    ai_integration_AnthropicService --> ai_integration_AnthropicService___init__
    ai_integration_AnthropicService_complete_text[complete_text()]
    ai_integration_AnthropicService --> ai_integration_AnthropicService_complete_text
    ai_integration_AIIntegration[AIIntegration]
    ai_integration --> ai_integration_AIIntegration
    ai_integration_AIIntegration___init__[__init__()]
    ai_integration_AIIntegration --> ai_integration_AIIntegration___init__
    ai_integration_AIIntegration_setup_services[setup_services()]
    ai_integration_AIIntegration --> ai_integration_AIIntegration_setup_services
    ai_integration_AIIntegration__load_cache[_load_cache()]
    ai_integration_AIIntegration --> ai_integration_AIIntegration__load_cache
    ai_integration_AIIntegration__save_cache[_save_cache()]
    ai_integration_AIIntegration --> ai_integration_AIIntegration__save_cache
    ai_integration_AIIntegration_get_embedding[get_embedding()]
    ai_integration_AIIntegration --> ai_integration_AIIntegration_get_embedding
    style ai_integration fill:#ffcc99
```