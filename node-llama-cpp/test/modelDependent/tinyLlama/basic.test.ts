import {describe, expect, test} from "vitest";
import {LlamaChatSession, LlamaCompletion} from "../../../src/index.js";
import {getModelFile} from "../../utils/modelFiles.js";
import {getTestLlama} from "../../utils/getTestLlama.js";

describe("TinyLlama Stories Models", () => {
    describe("stories260K", () => {
        test("basic inference", {timeout: 1000 * 60 * 5}, async () => {
            const modelPath = await getModelFile("stories260K.gguf");
            const llama = await getTestLlama();

            const model = await llama.loadModel({
                modelPath
            });
            const context = await model.createContext({
                contextSize: 512 // Small context for tiny model
            });

            // Test basic completion
            const completion = new LlamaCompletion({
                contextSequence: context.getSequence()
            });

            const result = await completion.generateCompletion("Once upon a time", {
                maxTokens: 20,
                temperature: 0.1 // Low temperature for more deterministic output
            });

            expect(result).toBeDefined();
            expect(result.length).toBeGreaterThan(0);
            expect(typeof result).toBe("string");
            
            // Should contain some story-like continuation
            expect(result.toLowerCase()).toMatch(/\b(there|was|were|lived|went|said|looked|saw|came|had)\b/);
        });

        test("chat session", {timeout: 1000 * 60 * 5}, async () => {
            const modelPath = await getModelFile("stories260K.gguf");
            const llama = await getTestLlama();

            const model = await llama.loadModel({
                modelPath
            });
            const context = await model.createContext({
                contextSize: 512
            });

            const chatSession = new LlamaChatSession({
                contextSequence: context.getSequence()
            });

            const response = await chatSession.prompt("Tell me a very short story", {
                maxTokens: 30,
                temperature: 0.1
            });

            expect(response).toBeDefined();
            expect(response.length).toBeGreaterThan(0);
            expect(typeof response).toBe("string");
        });
    });

    describe("stories15M", () => {
        test("basic inference", {timeout: 1000 * 60 * 5}, async () => {
            const modelPath = await getModelFile("stories15M-q4_0.gguf");
            const llama = await getTestLlama();

            const model = await llama.loadModel({
                modelPath
            });
            const context = await model.createContext({
                contextSize: 1024 // Slightly larger context for 15M model
            });

            // Test basic completion
            const completion = new LlamaCompletion({
                contextSequence: context.getSequence()
            });

            const result = await completion.generateCompletion("Once upon a time there was a", {
                maxTokens: 30,
                temperature: 0.1
            });

            expect(result).toBeDefined();
            expect(result.length).toBeGreaterThan(0);
            expect(typeof result).toBe("string");
            
            // Should be a coherent story continuation
            expect(result.toLowerCase()).toMatch(/\b(little|big|small|young|old|good|bad|happy|sad|brave|wise)\b/);
        });

        test("chat session with system prompt", {timeout: 1000 * 60 * 5}, async () => {
            const modelPath = await getModelFile("stories15M-q4_0.gguf");
            const llama = await getTestLlama();

            const model = await llama.loadModel({
                modelPath
            });
            const context = await model.createContext({
                contextSize: 1024
            });

            const chatSession = new LlamaChatSession({
                contextSequence: context.getSequence(),
                systemPrompt: "You are a storyteller who tells very short children's stories."
            });

            const response = await chatSession.prompt("Tell me about a cat", {
                maxTokens: 50,
                temperature: 0.2
            });

            expect(response).toBeDefined();
            expect(response.length).toBeGreaterThan(0);
            expect(typeof response).toBe("string");
            
            // Should mention a cat
            expect(response.toLowerCase()).toMatch(/\bcat\b/);
        });

        test("model capabilities verification", {timeout: 1000 * 60 * 5}, async () => {
            const modelPath = await getModelFile("stories15M-q4_0.gguf");
            const llama = await getTestLlama();

            const model = await llama.loadModel({
                modelPath
            });

            // Verify model metadata
            const tokenizer = model.tokenizer;
            expect(tokenizer).toBeDefined();
            
            // Test tokenization
            const tokens = tokenizer.encode("Hello, world!");
            expect(tokens).toBeDefined();
            expect(tokens.length).toBeGreaterThan(0);
            
            const decoded = tokenizer.decode(tokens);
            expect(decoded).toBe("Hello, world!");
        });
    });
});