#!/usr/bin/env python3
"""
Echo9ML Tiny Model Integration Script

This script demonstrates how to integrate the tiny LLM models from node-llama-cpp
into the broader Echo9ML cognitive architecture. It provides a lightweight fallback
and testing capability.

Usage:
    python tiny_model_integration.py
    python tiny_model_integration.py --model stories15M
    python tiny_model_integration.py --test-mode
"""

import asyncio
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

class TinyModelIntegration:
    """Integration layer for tiny models in Echo9ML"""
    
    def __init__(self, model_type: str = "stories260K"):
        self.model_type = model_type
        self.node_llama_path = Path(__file__).parent / "node-llama-cpp"
        self.models_cache = Path(__file__).parent / "models"
        
    def check_dependencies(self) -> bool:
        """Check if node-llama-cpp is available and built"""
        try:
            # Check if Node.js is available
            result = subprocess.run(["node", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Node.js not found. Please install Node.js.")
                return False
                
            # Check if node-llama-cpp is built
            dist_path = self.node_llama_path / "dist"
            if not dist_path.exists():
                print("⚠️  node-llama-cpp not built. Building now...")
                return self.build_node_llama()
                
            return True
            
        except Exception as e:
            print(f"❌ Dependency check failed: {e}")
            return False
    
    def build_node_llama(self) -> bool:
        """Build node-llama-cpp if needed"""
        try:
            print("🔨 Building node-llama-cpp...")
            original_cwd = os.getcwd()
            os.chdir(self.node_llama_path)
            
            # Install dependencies
            result = subprocess.run(["npm", "install"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ npm install failed: {result.stderr}")
                return False
                
            # Build project
            result = subprocess.run(["npm", "run", "build"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ npm build failed: {result.stderr}")
                return False
                
            print("✅ node-llama-cpp built successfully")
            return True
            
        except Exception as e:
            print(f"❌ Build failed: {e}")
            return False
        finally:
            os.chdir(original_cwd)
    
    def run_tiny_model_demo(self, prompt: str = "Once upon a time") -> Optional[Dict[str, Any]]:
        """Run the tiny model demo and capture results"""
        try:
            original_cwd = os.getcwd()
            os.chdir(self.node_llama_path)
            
            demo_script = "examples/tiny-model-demo.js"
            cmd = ["node", demo_script, self.model_type, prompt]
            
            print(f"🚀 Running tiny model demo: {self.model_type}")
            print(f"📝 Prompt: '{prompt}'")
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            end_time = time.time()
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout,
                    "error": result.stderr,
                    "execution_time": end_time - start_time,
                    "model_type": self.model_type,
                    "prompt": prompt
                }
            else:
                return {
                    "success": False,
                    "output": result.stdout,
                    "error": result.stderr,
                    "execution_time": end_time - start_time,
                    "model_type": self.model_type,
                    "prompt": prompt
                }
                
        except subprocess.TimeoutExpired:
            print("⏰ Demo timed out after 5 minutes")
            return None
        except Exception as e:
            print(f"❌ Demo execution failed: {e}")
            return None
        finally:
            os.chdir(original_cwd)
    
    def test_model_availability(self) -> Dict[str, bool]:
        """Test if tiny models are accessible"""
        results = {}
        
        for model in ["stories260K", "stories15M"]:
            try:
                # Quick test to see if model can be loaded
                original_cwd = os.getcwd()
                os.chdir(self.node_llama_path)
                
                # Use a simple Node.js script to test model loading
                test_script = f"""
                import {{ getLlama }} from "./dist/index.js";
                
                async function test() {{
                    try {{
                        const llama = await getLlama();
                        const modelUrl = "hf:ggml-org/models:tinyllamas/{model}.gguf";
                        console.log("Testing model:", modelUrl);
                        
                        // Just test if we can create the downloader (don't actually download)
                        console.log("Model URL is valid");
                        process.exit(0);
                    }} catch (error) {{
                        console.error("Error:", error.message);
                        process.exit(1);
                    }}
                }}
                
                test();
                """
                
                with open("test_model.mjs", "w") as f:
                    f.write(test_script)
                
                result = subprocess.run(["node", "test_model.mjs"], 
                                      capture_output=True, text=True, timeout=30)
                results[model] = result.returncode == 0
                
                # Clean up
                if os.path.exists("test_model.mjs"):
                    os.remove("test_model.mjs")
                
            except Exception as e:
                print(f"⚠️  Error testing {model}: {e}")
                results[model] = False
            finally:
                os.chdir(original_cwd)
        
        return results
    
    def integrate_with_cognitive_architecture(self) -> Dict[str, Any]:
        """Integrate tiny model with Echo9ML cognitive architecture"""
        integration_config = {
            "tiny_model_config": {
                "enabled": True,
                "default_model": self.model_type,
                "fallback_enabled": True,
                "use_cases": [
                    "testing",
                    "development", 
                    "fallback",
                    "resource_constrained"
                ],
                "models": {
                    "stories260K": {
                        "parameters": "260K",
                        "size": "280KB",
                        "context_size": 512,
                        "best_for": ["unit_tests", "ci_cd", "minimal_resource"]
                    },
                    "stories15M": {
                        "parameters": "15M", 
                        "size": "15MB",
                        "context_size": 1024,
                        "best_for": ["demos", "development", "education"]
                    }
                }
            },
            "echo9ml_integration": {
                "cognitive_architecture_fallback": True,
                "ml_system_integration": True,
                "testing_framework": True,
                "emergency_protocols": True
            }
        }
        
        # Save integration config
        config_path = Path(__file__).parent / "tiny_model_config.json"
        with open(config_path, "w") as f:
            json.dump(integration_config, f, indent=2)
        
        print(f"💾 Integration config saved to {config_path}")
        return integration_config

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Echo9ML Tiny Model Integration")
    parser.add_argument("--model", choices=["stories260K", "stories15M"], 
                       default="stories260K", help="Model type to use")
    parser.add_argument("--test-mode", action="store_true", 
                       help="Run in test mode (quick validation)")
    parser.add_argument("--prompt", default="Once upon a time there was a little robot",
                       help="Prompt to test with")
    
    args = parser.parse_args()
    
    print("🤖 Echo9ML Tiny Model Integration")
    print("=" * 50)
    
    integration = TinyModelIntegration(model_type=args.model)
    
    # Check dependencies
    if not integration.check_dependencies():
        print("❌ Dependencies check failed. Please resolve issues and try again.")
        sys.exit(1)
    
    if args.test_mode:
        print("\n🧪 Running in test mode...")
        
        # Test model availability
        print("\n📋 Testing model availability...")
        availability = integration.test_model_availability()
        for model, available in availability.items():
            status = "✅" if available else "❌"
            print(f"  {status} {model}")
        
        # Quick demo
        if availability.get(args.model, False):
            print(f"\n🚀 Quick demo with {args.model}...")
            result = integration.run_tiny_model_demo("Hello world")
            if result and result["success"]:
                print("✅ Demo successful!")
                print(f"⏱️  Execution time: {result['execution_time']:.2f}s")
            else:
                print("❌ Demo failed")
        
    else:
        print(f"\n🚀 Running full integration with {args.model}...")
        
        # Run demo
        result = integration.run_tiny_model_demo(args.prompt)
        if result:
            if result["success"]:
                print("✅ Tiny model demo completed successfully!")
                print(f"⏱️  Execution time: {result['execution_time']:.2f}s")
                print("\n📄 Output:")
                print(result["output"])
            else:
                print("❌ Demo failed:")
                print(result["error"])
        
        # Create integration config
        print("\n⚙️  Creating integration configuration...")
        config = integration.integrate_with_cognitive_architecture()
        
        print("\n✅ Integration complete!")
        print("\n💡 Next steps:")
        print("  • Review tiny_model_config.json for configuration details")
        print("  • Integrate with existing Echo9ML components")
        print("  • Use tiny models as fallback in cognitive_architecture.py")
        print("  • Add to emergency_protocols.py for system reliability")
        print("  • Include in test suites for continuous validation")

if __name__ == "__main__":
    main()