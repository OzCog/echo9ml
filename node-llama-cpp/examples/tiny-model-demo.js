#!/usr/bin/env node

/**
 * Tiny Model Demo Script
 * 
 * This script demonstrates the TinyLlama models capability as a fallback 
 * and testing solution for the echo9ml project.
 * 
 * Usage:
 *   node examples/tiny-model-demo.js [stories260K|stories15M] [prompt]
 * 
 * Examples:
 *   node examples/tiny-model-demo.js stories260K "Once upon a time"
 *   node examples/tiny-model-demo.js stories15M "Tell me a story about a cat"
 */

import path from "path";
import {fileURLToPath} from "url";
import {getLlama, LlamaChatSession, LlamaCompletion} from "../dist/index.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function main() {
    const args = process.argv.slice(2);
    const modelType = args[0] || "stories260K";
    const prompt = args[1] || "Once upon a time there was a";
    
    console.log("🚀 Echo9ML Tiny Model Demo");
    console.log("=" .repeat(50));
    console.log(`Model: ${modelType}`);
    console.log(`Prompt: "${prompt}"`);
    console.log("");

    try {
        // Get llama instance
        console.log("⚡ Initializing llama.cpp...");
        const llama = await getLlama();

        // Determine model URL based on type
        const modelUrls = {
            "stories260K": "hf:ggml-org/models:tinyllamas/stories260K.gguf",
            "stories15M": "hf:ggml-org/models:tinyllamas/stories15M-q4_0.gguf"
        };

        const modelUrl = modelUrls[modelType];
        if (!modelUrl) {
            throw new Error(`Unknown model type: ${modelType}. Use 'stories260K' or 'stories15M'`);
        }

        console.log(`📥 Loading model: ${modelUrl}`);
        const model = await llama.loadModel({
            modelPath: modelUrl
        });

        console.log(`🧠 Creating context (${modelType === "stories260K" ? "512" : "1024"} tokens)...`);
        const context = await model.createContext({
            contextSize: modelType === "stories260K" ? 512 : 1024
        });

        // Demo 1: Basic completion
        console.log("\n📝 Demo 1: Basic Text Completion");
        console.log("-" .repeat(30));
        
        const completion = new LlamaCompletion({
            contextSequence: context.getSequence()
        });

        const startTime = Date.now();
        const result = await completion.generateCompletion(prompt, {
            maxTokens: 50,
            temperature: 0.1
        });
        const endTime = Date.now();

        console.log(`Input: "${prompt}"`);
        console.log(`Output: "${result}"`);
        console.log(`⏱️  Generation time: ${endTime - startTime}ms`);

        // Demo 2: Chat session (only for stories15M as it's more capable)
        if (modelType === "stories15M") {
            console.log("\n💬 Demo 2: Chat Session");
            console.log("-" .repeat(30));

            const chatSession = new LlamaChatSession({
                contextSequence: context.getSequence(),
                systemPrompt: "You are a helpful storyteller who tells very short stories."
            });

            const chatStartTime = Date.now();
            const chatResponse = await chatSession.prompt("Tell me a very short story about friendship", {
                maxTokens: 80,
                temperature: 0.2
            });
            const chatEndTime = Date.now();

            console.log(`Chat Response: "${chatResponse}"`);
            console.log(`⏱️  Chat time: ${chatEndTime - chatStartTime}ms`);
        }

        // Demo 3: Model info
        console.log("\n📊 Model Information");
        console.log("-" .repeat(30));
        
        const tokenizer = model.tokenizer;
        const testTokens = tokenizer.encode("Hello, world!");
        const decodedTest = tokenizer.decode(testTokens);
        
        console.log(`Vocabulary size: ~${tokenizer.vocabularySize} tokens`);
        console.log(`Tokenizer test: "Hello, world!" -> ${testTokens.length} tokens -> "${decodedTest}"`);
        console.log(`Model type: ${modelType === "stories260K" ? "Ultra-tiny (260K params)" : "Tiny (15M params)"}`);
        console.log(`File size: ${modelType === "stories260K" ? "~280KB" : "~15MB"}`);

        console.log("\n✅ Demo completed successfully!");
        console.log("\n💡 Use cases for tiny models:");
        console.log("  • Testing and development");
        console.log("  • CI/CD pipelines");
        console.log("  • Resource-constrained environments");
        console.log("  • Fallback when larger models fail");
        console.log("  • Educational purposes");
        console.log("  • Quick prototyping");

    } catch (error) {
        console.error("❌ Error:", error.message);
        if (error.message.includes("Could not resolve host")) {
            console.log("\n💡 Tip: This demo requires internet access to download models.");
            console.log("   In production, models should be pre-downloaded and cached.");
        }
        process.exit(1);
    }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log("\n\n👋 Goodbye!");
    process.exit(0);
});

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}