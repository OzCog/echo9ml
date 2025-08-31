# Tiny Model Setup for Echo9ML

This document describes the tiny model integration for the Echo9ML project, providing a permanent fallback and testing solution using TinyLlama models.

## Overview

Two tiny models have been integrated into the node-llama-cpp package:

1. **TinyLlama Stories 260K** (~280KB, 260K parameters)
   - Ultra-minimal model for basic testing
   - Perfect for CI/CD pipelines and resource-constrained environments
   - Trained on children's stories

2. **TinyLlama Stories 15M** (~15MB, 15M parameters)  
   - Small but more capable model
   - Better for demonstrations and educational purposes
   - Supports more complex story generation

## Use Cases

### Primary Use Cases
- **Fallback Model**: When larger models fail to load or are unavailable
- **Testing & Development**: Quick validation of inference pipelines
- **CI/CD Integration**: Fast, lightweight tests in automated environments
- **Educational Purposes**: Learning llama.cpp without large downloads
- **Prototyping**: Rapid development and iteration

### Benefits
- **Fast Download**: Models download in seconds rather than minutes/hours
- **Low Memory**: Minimal RAM and storage requirements
- **Quick Inference**: Very fast token generation
- **Reliable**: Simple models are less likely to fail
- **Offline Capable**: Small enough to bundle with applications

## Integration Points

### 1. Test Infrastructure (`test/utils/modelFiles.ts`)
```typescript
const supportedModels = {
    // ... existing models
    "stories260K.gguf": "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf",
    "stories15M-q4_0.gguf": "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf"
} as const;
```

### 2. Recommended Models (`src/cli/recommendedModels.ts`)
Both models are now available in the CLI model selection with appropriate descriptions and use case guidance.

### 3. Test Suite (`test/modelDependent/tinyLlama/basic.test.ts`)
Comprehensive tests ensuring:
- Basic text completion works
- Chat sessions function correctly
- Tokenization is working
- Model metadata is accessible
- Performance is acceptable

### 4. Demo Script (`examples/tiny-model-demo.js`)
Interactive demonstration showing:
- Model loading and initialization
- Text completion
- Chat sessions (for 15M model)
- Performance metrics
- Use case examples

## Usage Examples

### Basic Text Completion
```javascript
import {getLlama, LlamaCompletion} from "node-llama-cpp";

const llama = await getLlama();
const model = await llama.loadModel({
    modelPath: "hf:ggml-org/models:tinyllamas/stories260K.gguf"
});

const context = await model.createContext({contextSize: 512});
const completion = new LlamaCompletion({
    contextSequence: context.getSequence()
});

const result = await completion.generateCompletion("Once upon a time", {
    maxTokens: 50,
    temperature: 0.1
});
```

### Chat Session
```javascript
import {getLlama, LlamaChatSession} from "node-llama-cpp";

const llama = await getLlama();
const model = await llama.loadModel({
    modelPath: "hf:ggml-org/models:tinyllamas/stories15M-q4_0.gguf"
});

const context = await model.createContext({contextSize: 1024});
const chatSession = new LlamaChatSession({
    contextSequence: context.getSequence(),
    systemPrompt: "You are a helpful storyteller."
});

const response = await chatSession.prompt("Tell me a short story", {
    maxTokens: 100
});
```

### Fallback Pattern
```javascript
async function loadModelWithFallback(preferredModel) {
    try {
        return await llama.loadModel({modelPath: preferredModel});
    } catch (error) {
        console.warn("Preferred model failed, using fallback:", error.message);
        return await llama.loadModel({
            modelPath: "hf:ggml-org/models:tinyllamas/stories260K.gguf"
        });
    }
}
```

## Running the Demo

```bash
# Build the project
npm run build

# Run with default settings (stories260K)
node examples/tiny-model-demo.js

# Use the larger model with custom prompt
node examples/tiny-model-demo.js stories15M "Tell me about a dragon"

# Quick test with ultra-tiny model
node examples/tiny-model-demo.js stories260K "Hello"
```

## Running Tests

```bash
# Run all tests (including tiny model tests)
npm test

# Run only tiny model tests
npm run test:modelDependent -- test/modelDependent/tinyLlama

# Run specific tiny model test
npx vitest test/modelDependent/tinyLlama/basic.test.ts
```

## Model Specifications

| Model | Parameters | File Size | Context | Best For |
|-------|------------|-----------|---------|----------|
| stories260K | 260K | ~280KB | 512 | Unit tests, CI/CD |
| stories15M | 15M | ~15MB | 1024 | Demos, development |

## Performance Expectations

### stories260K
- Load time: < 1 second
- First token: < 100ms
- Generation speed: 100+ tokens/second
- Memory usage: < 50MB

### stories15M  
- Load time: < 5 seconds
- First token: < 200ms
- Generation speed: 50+ tokens/second
- Memory usage: < 200MB

## Limitations

1. **Limited Vocabulary**: Trained primarily on simple stories
2. **Basic Capabilities**: No complex reasoning or knowledge
3. **Story Domain**: Works best with narrative/story prompts
4. **No Function Calling**: stories260K doesn't support advanced features
5. **Simple Responses**: Output quality is limited by model size

## Integration with Echo9ML

The tiny models integrate seamlessly with the broader Echo9ML ecosystem:

1. **Testing**: Validate cognitive architectures without large model overhead
2. **Development**: Quick iteration on neural-symbolic integration
3. **Fallback**: Ensure system reliability when resources are constrained
4. **Demonstration**: Show system capabilities in resource-limited environments

## Troubleshooting

### Common Issues

1. **Download Failures**: Check internet connectivity, try alternative endpoints
2. **Memory Errors**: Even tiny models need some RAM, ensure minimum requirements
3. **Slow Generation**: Normal for CPU inference, consider GPU acceleration
4. **Poor Quality**: Expected with tiny models, adjust expectations or use larger fallback

### Solutions

```javascript
// Robust error handling
try {
    const model = await llama.loadModel({
        modelPath: "hf:ggml-org/models:tinyllamas/stories260K.gguf"
    });
} catch (error) {
    if (error.message.includes("network")) {
        console.log("Network issue, using cached model if available");
    } else if (error.message.includes("memory")) {
        console.log("Insufficient memory, try reducing context size");
    }
    throw error;
}
```

## Future Enhancements

1. **Pre-bundled Models**: Include models in the repository for offline use
2. **Model Switching**: Dynamic fallback chain based on available resources
3. **Performance Optimization**: GGML optimization for tiny models
4. **Custom Training**: Domain-specific tiny models for Echo9ML use cases
5. **Quantization**: Even smaller variants for extreme resource constraints

## Contributing

When working with tiny models:

1. Keep expectations realistic for model capabilities
2. Focus on infrastructure validation rather than output quality
3. Use appropriate context sizes and token limits
4. Test both models to ensure compatibility
5. Document any model-specific behaviors or limitations