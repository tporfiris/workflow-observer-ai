"""
FastAPI Backend for Workflow Observer AI

This is the main backend server that integrates all components:
- Screen Capture Agent
- VLM Analyzer 
- Pattern Detector
- Bot Generation & Execution
- Package Installation
- Real-time Monitoring

The server provides REST API endpoints and serves the web UI.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import os
import sys
import asyncio
import threading
import subprocess
from pathlib import Path
from datetime import datetime
import uvicorn

# Import our existing components
from capture_agent import ScreenCaptureAgent


# from cloud_vlm_analyzer import OptimizedVLMAnalyzer
from secure_vlm_analyzer import SecureVLMAnalyzer as OptimizedVLMAnalyzer


from llm_pattern_detector import load_data, create_prompt, analyze_with_claude
# from bot_generator import BotCodeGenerator
# from package_installer import PackageInstaller
from enhanced_bot_generator import EnhancedBotCodeGenerator
from enhanced_package_installer import PackageInstaller
from bot_executor import BotExecutor, BotStatus

# Initialize FastAPI app
app = FastAPI(
    title="Workflow Observer AI",
    description="Universal digital workforce automation platform",
    version="1.0.0"
)

# Global variables to manage state
capture_agent: Optional[ScreenCaptureAgent] = None
vlm_analyzer: Optional[OptimizedVLMAnalyzer] = None
# bot_generator: Optional[BotCodeGenerator] = None
bot_generator: Optional[EnhancedBotCodeGenerator] = None
package_installer: Optional[PackageInstaller] = None
bot_executor: Optional[BotExecutor] = None
monitoring_active = False
detected_patterns = []
generated_bots = {}

# Pydantic models for API requests/responses
class MonitoringRequest(BaseModel):
    duration_days: int = 7
    capture_interval: int = 10
    max_files: int = 1000

class MonitoringStatus(BaseModel):
    is_active: bool
    total_captures: int
    runtime_seconds: int
    disk_usage_mb: float
    last_capture: Optional[str]

class DetectedPattern(BaseModel):
    pattern_name: str
    pattern_type: str
    frequency: str
    applications_involved: List[str]
    workflow_steps: List[str]
    automation_potential: int
    business_value: str
    time_savings_estimate: str

class BotGenerationRequest(BaseModel):
    pattern_name: str
    pattern_description: str
    user_credentials: Dict[str, str] = {}

class BotExecutionRequest(BaseModel):
    bot_id: str
    config_overrides: Dict[str, Any] = {}
    timeout_minutes: int = 60

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global vlm_analyzer, bot_generator, package_installer, bot_executor, detected_patterns, generated_bots
    
    print("🚀 Starting Workflow Observer AI Backend...")
    
    # Initialize VLM Analyzer
    try:
        vlm_analyzer = OptimizedVLMAnalyzer()
        print("✅ VLM Analyzer initialized")
    except Exception as e:
        print(f"⚠️  VLM Analyzer initialization failed: {e}")
        vlm_analyzer = None
    
    # Initialize Bot Generator
    try:
        # bot_generator = BotCodeGenerator()
        bot_generator = EnhancedBotCodeGenerator()
        print("✅ Bot Generator initialized")
    except Exception as e:
        print(f"⚠️  Bot Generator initialization failed: {e}")
        bot_generator = None
    
    # Initialize Package Installer
    try:
        package_installer = PackageInstaller()
        print("✅ Package Installer initialized")
    except Exception as e:
        print(f"⚠️  Package Installer initialization failed: {e}")
        package_installer = None
    
    # Initialize Bot Executor
    try:
        bot_executor = BotExecutor()
        print("✅ Bot Executor initialized")
    except Exception as e:
        print(f"⚠️  Bot Executor initialization failed: {e}")
        bot_executor = None
    
    # Create necessary directories
    os.makedirs("screenshots", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("generated_bots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Load existing data
    await load_existing_data()
    
    print("✅ Workflow Observer AI Backend ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of all components"""
    global bot_executor
    
    print("🛑 Shutting down Workflow Observer AI Backend...")
    
    if bot_executor:
        bot_executor.shutdown()
    
    print("✅ Shutdown complete")

async def load_existing_data():
    """Load existing patterns and bots from files"""
    global detected_patterns, generated_bots, bot_generator
    
    # Load detected patterns
    patterns_file = Path("detected_patterns.json")
    if patterns_file.exists():
        try:
            with open(patterns_file, 'r') as f:
                detected_patterns = json.load(f)
            print(f"📋 Loaded {len(detected_patterns)} existing patterns")
        except Exception as e:
            print(f"⚠️  Error loading patterns file: {e}")
    
    # Load existing bots from generated_bots directory
    bots_dir = Path("generated_bots")
    if bots_dir.exists():
        bot_count = 0
        for bot_dir in bots_dir.iterdir():
            if bot_dir.is_dir() and not bot_dir.name.startswith('.'):
                try:
                    # Look for pattern_info.json to get bot details
                    pattern_info_file = bot_dir / "pattern_info.json"
                    environment_info_file = bot_dir / "environment_info.json"
                    
                    bot_info = {
                        "id": bot_dir.name,
                        "directory": str(bot_dir),
                        "status": "generated",
                        "created_at": datetime.fromtimestamp(bot_dir.stat().st_ctime).isoformat(),
                        "files": {}
                    }
                    
                    # Load pattern info if available
                    if pattern_info_file.exists():
                        with open(pattern_info_file, 'r') as f:
                            pattern_data = json.load(f)
                            bot_info.update({
                                "pattern_name": pattern_data.get("pattern_name", bot_dir.name),
                                "pattern_type": pattern_data.get("pattern_type", "unknown"),
                                "automation_potential": pattern_data.get("automation_potential", 0),
                                "time_savings": pattern_data.get("time_savings_estimate", "Unknown"),
                                "applications": pattern_data.get("applications_involved", []),
                                "business_value": pattern_data.get("business_value", ""),
                                "implementation_approach": pattern_data.get("implementation_approach", "")
                            })
                    else:
                        bot_info["pattern_name"] = bot_dir.name
                    
                    # Check environment status
                    if environment_info_file.exists():
                        with open(environment_info_file, 'r') as f:
                            env_data = json.load(f)
                            bot_info.update({
                                "installation_status": "success" if env_data.get("setup_completed") else "unknown",
                                "environment_ready": env_data.get("setup_completed", False),
                                "installation_result": env_data
                            })
                    
                    # List files in bot directory
                    for file_path in bot_dir.iterdir():
                        if file_path.is_file():
                            bot_info["files"][file_path.suffix or "other"] = str(file_path)
                    
                    # Add to generated_bots
                    generated_bots[bot_dir.name] = bot_info
                    
                    # Also add to bot_generator's tracking if available
                    if bot_generator:
                        bot_generator.generated_bots[bot_dir.name] = bot_info
                    
                    bot_count += 1
                    
                except Exception as e:
                    print(f"⚠️  Error loading bot {bot_dir.name}: {e}")
        
        print(f"🤖 Loaded {bot_count} existing bots")
    
    # Summary
    print(f"📊 Startup Summary:")
    print(f"   📋 Patterns: {len(detected_patterns)}")
    print(f"   🤖 Bots: {len(generated_bots)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "vlm_analyzer": vlm_analyzer is not None,
            "bot_generator": bot_generator is not None,
            "package_installer": package_installer is not None,
            "bot_executor": bot_executor is not None,
            "capture_agent": capture_agent is not None and capture_agent.is_running if capture_agent else False
        }
    }

# Monitoring endpoints
@app.post("/api/monitoring/start")
async def start_monitoring(request: MonitoringRequest):
    """Start screen capture monitoring"""
    global capture_agent, monitoring_active
    
    if monitoring_active and capture_agent and capture_agent.is_running:
        raise HTTPException(status_code=400, detail="Monitoring is already active")
    
    try:
        # Create new capture agent
        capture_agent = ScreenCaptureAgent(
            capture_interval=request.capture_interval,
            max_files=request.max_files,
            output_dir="screenshots",
            image_quality=85
        )
        
        # Start capturing
        success = capture_agent.start()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start screen capture")
        
        monitoring_active = True
        
        return {
            "status": "started",
            "message": f"Screen capture started for {request.duration_days} days",
            "capture_interval": request.capture_interval,
            "max_files": request.max_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting monitoring: {str(e)}")

@app.post("/api/monitoring/stop")
async def stop_monitoring():
    """Stop screen capture monitoring"""
    global capture_agent, monitoring_active
    
    if not monitoring_active or not capture_agent:
        raise HTTPException(status_code=400, detail="Monitoring is not active")
    
    try:
        capture_agent.stop()
        monitoring_active = False
        
        # Get final status
        final_status = capture_agent.get_status()
        
        return {
            "status": "stopped",
            "message": "Screen capture stopped",
            "final_stats": final_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping monitoring: {str(e)}")

@app.get("/api/monitoring/status")
async def get_monitoring_status() -> MonitoringStatus:
    """Get current monitoring status"""
    if not capture_agent or not monitoring_active:
        return MonitoringStatus(
            is_active=False,
            total_captures=0,
            runtime_seconds=0,
            disk_usage_mb=0.0,
            last_capture=None
        )
    
    status = capture_agent.get_status()
    return MonitoringStatus(
        is_active=status['is_running'],
        total_captures=status['total_captures'],
        runtime_seconds=status.get('runtime_seconds', 0),
        disk_usage_mb=status['disk_usage_mb'],
        last_capture=status.get('last_capture')
    )

# Pattern detection endpoints
@app.post("/api/patterns/analyze")
async def analyze_patterns(background_tasks: BackgroundTasks):
    """Analyze captured screenshots to detect workflow patterns"""
    global detected_patterns
    
    if not vlm_analyzer:
        raise HTTPException(status_code=500, detail="VLM Analyzer not available")
    
    # Check if we have screenshots
    screenshots_dir = Path("screenshots")
    screenshot_files = list(screenshots_dir.glob("screenshot_*.jpg"))
    
    if not screenshot_files:
        raise HTTPException(status_code=400, detail="No screenshots found to analyze")
    
    try:
        # Start analysis in background
        background_tasks.add_task(run_pattern_analysis, screenshot_files)
        
        return {
            "status": "started",
            "message": f"Pattern analysis started for {len(screenshot_files)} screenshots",
            "estimated_time_minutes": len(screenshot_files) * 0.1  # Rough estimate
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting pattern analysis: {str(e)}")

async def run_pattern_analysis(screenshot_files: List[Path]):
    """Background task to run the complete pattern analysis"""
    global detected_patterns, vlm_analyzer
    
    try:
        print(f"🔍 Starting VLM analysis of {len(screenshot_files)} screenshots...")
        
        # Step 1: VLM Analysis
        results = vlm_analyzer.analyze_batch(
            [str(f) for f in screenshot_files],
            max_images=None,
            delay_between_calls=1.0
        )
        
        # Step 2: Save VLM results
        vlm_analyzer.save_results(results, "vlm_analysis_cache.json")
        
        # Step 3: Pattern Detection with LLM
        print("🧠 Running pattern detection...")
        data = load_data("vlm_analysis_cache.json")
        prompt = create_prompt(data)
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise Exception("Anthropic API key not found")
        
        pattern_result = analyze_with_claude(prompt, api_key)
        
        # Step 4: Parse and store patterns
        try:
            # Try to extract JSON from the response
            start_idx = pattern_result.find('[')
            end_idx = pattern_result.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                patterns_json = pattern_result[start_idx:end_idx]
                detected_patterns = json.loads(patterns_json)
            else:
                detected_patterns = json.loads(pattern_result)
        except json.JSONDecodeError:
            print("⚠️  Could not parse patterns as JSON, storing as text")
            detected_patterns = [{"pattern_name": "Analysis Result", "raw_result": pattern_result}]
        
        # Save patterns to file
        with open("detected_patterns.json", "w") as f:
            json.dump(detected_patterns, f, indent=2)
        
        print(f"✅ Pattern analysis complete! Found {len(detected_patterns)} patterns")
        
    except Exception as e:
        print(f"❌ Error during pattern analysis: {e}")
        detected_patterns = [{"error": str(e)}]

@app.get("/api/patterns/detected")
async def get_detected_patterns():
    """Get list of detected workflow patterns"""
    global detected_patterns
    
    return {
        "patterns": detected_patterns,
        "total_count": len(detected_patterns),
        "last_analysis": datetime.now().isoformat()
    }

# Bot generation endpoints
@app.post("/api/bots/generate")
async def generate_bot(request: BotGenerationRequest):
    """Generate automation bot code for a detected pattern"""
    global bot_generator, detected_patterns
    
    if not bot_generator:
        raise HTTPException(status_code=500, detail="Bot Generator not available")
    
    # Find the pattern by name
    pattern = None
    for p in detected_patterns:
        if p.get("pattern_name") == request.pattern_name:
            pattern = p
            break
    
    if not pattern:
        raise HTTPException(status_code=404, detail=f"Pattern '{request.pattern_name}' not found")
    
    try:
        print(f"🤖 Generating bot for pattern: {request.pattern_name}")
        
        # Generate the bot using our BotCodeGenerator
        result = bot_generator.generate_bot(pattern)
        
        if result.get("status") == "generated":
            # Update our global bots tracking
            bot_id = result["id"]
            generated_bots[bot_id] = result
            
            return {
                "bot_id": bot_id,
                "status": "generated",
                "message": f"Bot successfully generated for pattern: {request.pattern_name}",
                "files_created": result.get("files", {}),
                "directory": result.get("directory"),
                "automation_potential": result.get("automation_potential"),
                "time_savings": result.get("time_savings"),
                "next_steps": [
                    "Review the generated code in the bot directory",
                    "Edit config.json with your credentials and settings", 
                    "Install requirements: pip install -r requirements.txt",
                    "Test the bot: python {}.py".format(bot_id),
                    "Schedule for automatic execution"
                ]
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Bot generation failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating bot: {str(e)}")

@app.post("/api/bots/generate-all")
async def generate_all_bots():
    """Generate automation bots for all detected patterns"""
    global bot_generator, detected_patterns
    
    if not bot_generator:
        raise HTTPException(status_code=500, detail="Bot Generator not available")
    
    if not detected_patterns:
        raise HTTPException(status_code=400, detail="No patterns detected. Run pattern analysis first.")
    
    try:
        print(f"🤖 Generating bots for {len(detected_patterns)} patterns...")
        
        # Filter out any invalid patterns
        valid_patterns = [p for p in detected_patterns if isinstance(p, dict) and p.get("pattern_name")]
        
        if not valid_patterns:
            raise HTTPException(status_code=400, detail="No valid patterns found for bot generation")
        
        # Generate bots for all patterns
        results = bot_generator.generate_multiple_bots(valid_patterns)
        
        # Update global tracking
        for result in results:
            if result.get("status") == "generated":
                bot_id = result["id"]
                generated_bots[bot_id] = result
        
        successful = len([r for r in results if r.get("status") == "generated"])
        failed = len(results) - successful
        
        return {
            "status": "completed",
            "message": f"Generated {successful} bots successfully, {failed} failed",
            "total_patterns": len(valid_patterns),
            "successful_generations": successful,
            "failed_generations": failed,
            "generated_bots": [r for r in results if r.get("status") == "generated"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating bots: {str(e)}")

@app.get("/api/bots/list")
async def list_bots():
    """Get list of all generated bots"""
    global bot_generator
    
    if bot_generator:
        # Get comprehensive bot information from the generator
        bot_info = bot_generator.get_generated_bots()
        return {
            "bots": list(bot_info["bots"].values()),
            "total_count": bot_info["total_bots"],
            "generation_stats": bot_info["generation_stats"]
        }
    else:
        return {
            "bots": list(generated_bots.values()),
            "total_count": len(generated_bots),
            "generation_stats": {"note": "Bot generator not initialized"}
        }

@app.get("/api/bots/{bot_id}/status")
async def get_bot_status(bot_id: str):
    """Get status of a specific bot"""
    if bot_id not in generated_bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    return generated_bots[bot_id]

@app.get("/api/bots/{bot_id}/files")
async def get_bot_files(bot_id: str):
    """Get list of files for a specific bot"""
    if bot_id not in generated_bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot_info = generated_bots[bot_id]
    bot_dir = Path(bot_info.get("directory", f"generated_bots/{bot_id}"))
    
    if not bot_dir.exists():
        raise HTTPException(status_code=404, detail="Bot directory not found")
    
    files = []
    for file_path in bot_dir.iterdir():
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
    
    return {
        "bot_id": bot_id,
        "directory": str(bot_dir),
        "files": files,
        "total_files": len(files)
    }

@app.get("/api/bots/{bot_id}/code")
async def get_bot_code(bot_id: str):
    """Get the generated Python code for a bot"""
    if bot_id not in generated_bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot_info = generated_bots[bot_id]
    bot_dir = Path(bot_info.get("directory", f"generated_bots/{bot_id}"))
    code_file = bot_dir / f"{bot_id}.py"
    
    if not code_file.exists():
        raise HTTPException(status_code=404, detail="Bot code file not found")
    
    try:
        with open(code_file, 'r') as f:
            code_content = f.read()
        
        return {
            "bot_id": bot_id,
            "filename": f"{bot_id}.py",
            "code": code_content,
            "file_size": len(code_content),
            "last_modified": datetime.fromtimestamp(code_file.stat().st_mtime).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading bot code: {str(e)}")

# Package installation endpoints
@app.post("/api/bots/{bot_id}/install")
async def install_bot_environment(bot_id: str, background_tasks: BackgroundTasks):
    """Install packages and setup environment for a specific bot"""
    global package_installer
    
    if not package_installer:
        raise HTTPException(status_code=500, detail="Package Installer not available")
    
    if bot_id not in generated_bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot_info = generated_bots[bot_id]
    bot_dir = Path(bot_info.get("directory", f"generated_bots/{bot_id}"))
    
    if not bot_dir.exists():
        raise HTTPException(status_code=404, detail="Bot directory not found")
    
    # Start installation in background
    background_tasks.add_task(install_bot_environment_task, bot_id, bot_dir)
    
    return {
        "bot_id": bot_id,
        "status": "installation_started",
        "message": f"Environment setup started for bot: {bot_id}",
        "estimated_time_minutes": 5
    }

async def install_bot_environment_task(bot_id: str, bot_dir: Path):
    """Background task to install bot environment"""
    global package_installer, generated_bots
    
    try:
        print(f"📦 Installing environment for bot: {bot_id}")
        
        # Update bot status
        generated_bots[bot_id]["installation_status"] = "installing"
        generated_bots[bot_id]["installation_started"] = datetime.now().isoformat()
        
        # Setup environment
        result = package_installer.setup_bot_environment(bot_dir)
        
        # Update bot status with results
        generated_bots[bot_id]["installation_status"] = result["status"]
        generated_bots[bot_id]["installation_completed"] = datetime.now().isoformat()
        generated_bots[bot_id]["installation_result"] = result
        
        if result["status"] == "success":
            generated_bots[bot_id]["environment_ready"] = True
            print(f"✅ Environment setup completed for bot: {bot_id}")
        else:
            generated_bots[bot_id]["environment_ready"] = False
            print(f"❌ Environment setup failed for bot: {bot_id}")
            
    except Exception as e:
        generated_bots[bot_id]["installation_status"] = "failed"
        generated_bots[bot_id]["installation_error"] = str(e)
        generated_bots[bot_id]["environment_ready"] = False
        print(f"❌ Environment setup error for bot {bot_id}: {e}")

@app.post("/api/bots/install-all")
async def install_all_bot_environments(background_tasks: BackgroundTasks):
    """Install packages and setup environments for all generated bots"""
    global package_installer, generated_bots
    
    if not package_installer:
        raise HTTPException(status_code=500, detail="Package Installer not available")
    
    if not generated_bots:
        raise HTTPException(status_code=400, detail="No bots found. Generate some bots first.")
    
    # Get valid bot directories
    valid_bots = []
    for bot_id, bot_info in generated_bots.items():
        bot_dir = Path(bot_info.get("directory", f"generated_bots/{bot_id}"))
        if bot_dir.exists():
            valid_bots.append((bot_id, bot_dir))
    
    if not valid_bots:
        raise HTTPException(status_code=400, detail="No valid bot directories found")
    
    # Start installation in background
    background_tasks.add_task(install_all_environments_task, valid_bots)
    
    return {
        "status": "installation_started",
        "message": f"Environment setup started for {len(valid_bots)} bots",
        "bot_count": len(valid_bots),
        "estimated_time_minutes": len(valid_bots) * 3
    }

async def install_all_environments_task(valid_bots: List[tuple]):
    """Background task to install all bot environments"""
    global package_installer, generated_bots
    
    try:
        print(f"📦 Installing environments for {len(valid_bots)} bots...")
        
        # Setup environments
        bot_names = [bot_id for bot_id, _ in valid_bots]
        results = package_installer.setup_multiple_bots(bot_names)
        
        # Update bot statuses
        for i, (bot_id, _) in enumerate(valid_bots):
            if i < len(results):
                result = results[i]
                generated_bots[bot_id]["installation_status"] = result["status"]
                generated_bots[bot_id]["installation_completed"] = datetime.now().isoformat()
                generated_bots[bot_id]["installation_result"] = result
                generated_bots[bot_id]["environment_ready"] = result["status"] == "success"
        
        successful = len([r for r in results if r["status"] == "success"])
        print(f"✅ Batch environment setup completed: {successful}/{len(valid_bots)} successful")
        
    except Exception as e:
        print(f"❌ Batch environment setup error: {e}")
        # Mark all as failed
        for bot_id, _ in valid_bots:
            generated_bots[bot_id]["installation_status"] = "failed"
            generated_bots[bot_id]["installation_error"] = str(e)
            generated_bots[bot_id]["environment_ready"] = False

@app.get("/api/bots/{bot_id}/verify")
async def verify_bot_environment(bot_id: str):
    """Verify that a bot's environment is properly set up"""
    global package_installer
    
    if not package_installer:
        raise HTTPException(status_code=500, detail="Package Installer not available")
    
    if bot_id not in generated_bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot_info = generated_bots[bot_id]
    bot_dir = Path(bot_info.get("directory", f"generated_bots/{bot_id}"))
    
    if not bot_dir.exists():
        raise HTTPException(status_code=404, detail="Bot directory not found")
    
    try:
        verification = package_installer.verify_bot_environment(bot_dir)
        return verification
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verifying environment: {str(e)}")

@app.get("/api/installation/status")
async def get_installation_status():
    """Get status of all package installations"""
    global package_installer
    
    if not package_installer:
        raise HTTPException(status_code=500, detail="Package Installer not available")
    
    return package_installer.get_installation_status()

# Bot execution endpoints
@app.post("/api/bots/{bot_id}/execute")
async def execute_bot(bot_id: str, request: BotExecutionRequest):
    """Execute a bot with optional configuration overrides"""
    global bot_executor
    
    if not bot_executor:
        raise HTTPException(status_code=500, detail="Bot Executor not available")
    
    if bot_id not in generated_bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    try:
        result = bot_executor.execute_bot(
            bot_id=bot_id,
            config_overrides=request.config_overrides,
            timeout_minutes=request.timeout_minutes
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing bot: {str(e)}")

@app.post("/api/bots/{bot_id}/stop")
async def stop_bot_execution(bot_id: str, execution_id: str):
    """Stop a running bot execution"""
    global bot_executor
    
    if not bot_executor:
        raise HTTPException(status_code=500, detail="Bot Executor not available")
    
    try:
        result = bot_executor.stop_bot(execution_id)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping bot: {str(e)}")

@app.get("/api/executions/running")
async def get_running_executions():
    """Get list of currently running bot executions"""
    global bot_executor
    
    if not bot_executor:
        raise HTTPException(status_code=500, detail="Bot Executor not available")
    
    try:
        running_bots = bot_executor.get_running_bots()
        return {
            "running_executions": running_bots,
            "count": len(running_bots),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting running executions: {str(e)}")

@app.get("/api/executions/{execution_id}/status")
async def get_execution_status(execution_id: str):
    """Get status of a specific execution"""
    global bot_executor
    
    if not bot_executor:
        raise HTTPException(status_code=500, detail="Bot Executor not available")
    
    try:
        status = bot_executor.get_execution_status(execution_id)
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting execution status: {str(e)}")

@app.get("/api/executions/{execution_id}/logs")
async def get_execution_logs(execution_id: str):
    """Get logs for a specific execution"""
    global bot_executor
    
    if not bot_executor:
        raise HTTPException(status_code=500, detail="Bot Executor not available")
    
    try:
        logs = bot_executor.get_bot_logs(execution_id)
        return logs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting execution logs: {str(e)}")

@app.get("/api/executions/history")
async def get_execution_history(bot_id: Optional[str] = None, limit: int = 50):
    """Get execution history"""
    global bot_executor
    
    if not bot_executor:
        raise HTTPException(status_code=500, detail="Bot Executor not available")
    
    try:
        history = bot_executor.get_execution_history(bot_id=bot_id, limit=limit)
        return {
            "execution_history": history,
            "total_count": len(history),
            "bot_filter": bot_id,
            "limit": limit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting execution history: {str(e)}")

@app.get("/api/executions/stats")
async def get_execution_stats():
    """Get execution statistics"""
    global bot_executor
    
    if not bot_executor:
        raise HTTPException(status_code=500, detail="Bot Executor not available")
    
    try:
        stats = bot_executor.get_execution_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting execution stats: {str(e)}")

@app.post("/api/bots/{bot_id}/validate")
async def validate_bot_environment(bot_id: str):
    """Validate that a bot's environment is ready for execution"""
    global bot_executor
    
    if not bot_executor:
        raise HTTPException(status_code=500, detail="Bot Executor not available")
    
    if bot_id not in generated_bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    try:
        is_valid, validation_message = bot_executor._validate_bot_environment(bot_id)
        
        return {
            "bot_id": bot_id,
            "is_valid": is_valid,
            "validation_message": validation_message,
            "environment_ready": is_valid,
            "validated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating bot environment: {str(e)}")

# Debug endpoint
@app.get("/api/bots/{bot_id}/debug")
async def debug_bot_environment(bot_id: str):
    """Debug bot environment paths and files"""
    if bot_id not in generated_bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot_info = generated_bots[bot_id]
    bot_dir = Path(bot_info.get("directory", f"generated_bots/{bot_id}"))
    
    debug_info = {
        "bot_id": bot_id,
        "bot_dir": str(bot_dir),
        "bot_dir_exists": bot_dir.exists(),
        "files_in_bot_dir": [],
        "venv_info": {},
        "python_executables": []
    }
    
    # List files in bot directory
    if bot_dir.exists():
        debug_info["files_in_bot_dir"] = [f.name for f in bot_dir.iterdir()]
        
        # Check venv directory
        venv_dir = bot_dir / "venv"
        debug_info["venv_info"]["venv_dir_exists"] = venv_dir.exists()
        
        if venv_dir.exists():
            debug_info["venv_info"]["venv_contents"] = [f.name for f in venv_dir.iterdir()]
            
            # Check bin directory
            bin_dir = venv_dir / "bin"
            if bin_dir.exists():
                debug_info["venv_info"]["bin_contents"] = [f.name for f in bin_dir.iterdir()]
                
                # Test all potential Python executables
                python_candidates = [
                    bin_dir / "python",
                    bin_dir / "python3", 
                    bin_dir / f"python{sys.version_info.major}.{sys.version_info.minor}",
                    bin_dir / f"python{sys.version_info.major}"
                ]
                
                for candidate in python_candidates:
                    test_info = {
                        "path": str(candidate),
                        "exists": candidate.exists(),
                        "is_file": candidate.is_file() if candidate.exists() else False,
                        "is_executable": False,
                        "version_test": None
                    }
                    
                    if candidate.exists() and candidate.is_file():
                        # Check if executable
                        try:
                            stat_result = candidate.stat()
                            test_info["is_executable"] = bool(stat_result.st_mode & 0o111)
                        except:
                            pass
                        
                        # Test version command
                        try:
                            result = subprocess.run([str(candidate), "--version"], 
                                                  capture_output=True, text=True, timeout=10)
                            test_info["version_test"] = {
                                "return_code": result.returncode,
                                "stdout": result.stdout.strip(),
                                "stderr": result.stderr.strip()
                            }
                        except Exception as e:
                            test_info["version_test"] = {"error": str(e)}
                    
                    debug_info["python_executables"].append(test_info)
    
    return debug_info

# Data management endpoints
@app.post("/api/data/refresh")
async def refresh_data():
    """Manually refresh patterns and bots from files"""
    try:
        await load_existing_data()
        return {
            "status": "success",
            "message": "Data refreshed successfully",
            "patterns_loaded": len(detected_patterns),
            "bots_loaded": len(generated_bots),
            "refreshed_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing data: {str(e)}")

@app.get("/api/data/summary")
async def get_data_summary():
    """Get summary of all data"""
    return {
        "patterns": {
            "count": len(detected_patterns),
            "file_exists": Path("detected_patterns.json").exists()
        },
        "bots": {
            "count": len(generated_bots),
            "directory_exists": Path("generated_bots").exists(),
            "bot_names": list(generated_bots.keys())
        },
        "screenshots": {
            "count": len(list(Path("screenshots").glob("*.jpg"))) if Path("screenshots").exists() else 0,
            "directory_exists": Path("screenshots").exists()
        }
    }

# Static file serving and web UI
@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Workflow Observer AI</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; }
            .status { padding: 15px; margin: 20px 0; border-radius: 5px; }
            .status.healthy { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .status.inactive { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 20px; margin: 10px 5px; border: none; border-radius: 5px; cursor: pointer; }
            .primary { background: #007acc; color: white; }
            .secondary { background: #6c757d; color: white; }
            .success { background: #28a745; color: white; }
            .danger { background: #dc3545; color: white; }
            pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Workflow Observer AI Dashboard</h1>
            
            <div class="status healthy">
                <strong>✅ System Status:</strong> Backend is running and ready
            </div>
            
            <div class="section">
                <h2>📷 Screen Monitoring</h2>
                <p>Capture screenshots to learn your workflows</p>
                <button class="primary" onclick="startMonitoring()">Start Monitoring</button>
                <button class="danger" onclick="stopMonitoring()">Stop Monitoring</button>
                <button class="secondary" onclick="checkStatus()">Check Status</button>
                <div id="monitoring-status"></div>
            </div>
            
            <div class="section">
                <h2>📊 Data Management</h2>
                <p>View and manage your workflow data</p>
                <button class="primary" onclick="refreshData()">Refresh Data</button>
                <button class="secondary" onclick="getDataSummary()">Data Summary</button>
                <div id="data-result"></div>
            </div>
            
            <div class="section">
                <h2>🧠 Pattern Analysis</h2>
                <p>Analyze captured screenshots to detect workflow patterns</p>
                <button class="primary" onclick="analyzePatterns()">Analyze Patterns</button>
                <button class="secondary" onclick="getPatterns()">View Detected Patterns</button>
                <div id="patterns-result"></div>
            </div>
            
            <div class="section">
                <h2>🤖 Automation Bots</h2>
                <p>Generate and manage automation bots</p>
                <button class="primary" onclick="generateAllBots()">Generate All Bots</button>
                <button class="success" onclick="installAllEnvironments()">Install All Environments</button>
                <button class="secondary" onclick="listBots()">List Bots</button>
                <button class="secondary" onclick="verifyEnvironments()">Verify Environments</button>
                <button class="secondary" onclick="getBotCode()">View Bot Code</button>
                <div id="bots-result"></div>
            </div>
            
            <div class="section">
                <h2>▶️ Bot Execution</h2>
                <p>Execute and monitor automation bots</p>
                <button class="primary" onclick="executeFirstBot()">Execute First Bot</button>
                <button class="secondary" onclick="debugFirstBot()">Debug First Bot</button>
                <button class="secondary" onclick="getLastExecutionLogs()">View Last Logs</button>
                <button class="secondary" onclick="getRunningExecutions()">Running Executions</button>
                <button class="secondary" onclick="getExecutionHistory()">Execution History</button>
                <button class="secondary" onclick="getExecutionStats()">Execution Stats</button>
                <div id="execution-result"></div>
            </div>
            
            <div class="section">
                <h2>🔧 API Endpoints</h2>
                <p>Available API endpoints for testing:</p>
                <ul>
                    <li><code>GET /health</code> - Health check</li>
                    <li><code>POST /api/monitoring/start</code> - Start monitoring</li>
                    <li><code>POST /api/monitoring/stop</code> - Stop monitoring</li>
                    <li><code>GET /api/monitoring/status</code> - Get monitoring status</li>
                    <li><code>POST /api/patterns/analyze</code> - Analyze patterns</li>
                    <li><code>GET /api/patterns/detected</code> - Get detected patterns</li>
                    <li><code>POST /api/bots/generate</code> - Generate bot for specific pattern</li>
                    <li><code>POST /api/bots/generate-all</code> - Generate bots for all patterns</li>
                    <li><code>GET /api/bots/list</code> - List bots</li>
                    <li><code>GET /api/bots/{id}/status</code> - Get bot status</li>
                    <li><code>GET /api/bots/{id}/code</code> - View generated bot code</li>
                    <li><code>POST /api/bots/{id}/install</code> - Install bot environment</li>
                    <li><code>POST /api/bots/install-all</code> - Install all bot environments</li>
                    <li><code>GET /api/bots/{id}/verify</code> - Verify bot environment</li>
                    <li><code>GET /api/installation/status</code> - Installation status</li>
                    <li><code>POST /api/bots/{id}/execute</code> - Execute bot</li>
                    <li><code>POST /api/bots/{id}/stop</code> - Stop bot execution</li>
                    <li><code>GET /api/executions/running</code> - Get running executions</li>
                    <li><code>GET /api/executions/{id}/status</code> - Get execution status</li>
                    <li><code>GET /api/executions/{id}/logs</code> - Get execution logs</li>
                    <li><code>GET /api/executions/history</code> - Get execution history</li>
                    <li><code>GET /api/executions/stats</code> - Get execution statistics</li>
                    <li><code>POST /api/data/refresh</code> - Refresh data from files</li>
                    <li><code>GET /api/data/summary</code> - Get data summary</li>
                </ul>
            </div>
        </div>
        
        <script>
            async function makeRequest(url, method = 'GET', body = null) {
                try {
                    const options = { method };
                    if (body) {
                        options.headers = { 'Content-Type': 'application/json' };
                        options.body = JSON.stringify(body);
                    }
                    const response = await fetch(url, options);
                    const data = await response.json();
                    return { success: response.ok, data };
                } catch (error) {
                    return { success: false, error: error.message };
                }
            }
            
            async function startMonitoring() {
                const result = await makeRequest('/api/monitoring/start', 'POST', {
                    duration_days: 7,
                    capture_interval: 10,
                    max_files: 1000
                });
                document.getElementById('monitoring-status').innerHTML = 
                    `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            async function stopMonitoring() {
                const result = await makeRequest('/api/monitoring/stop', 'POST');
                document.getElementById('monitoring-status').innerHTML = 
                    `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            async function checkStatus() {
                const result = await makeRequest('/api/monitoring/status');
                document.getElementById('monitoring-status').innerHTML = 
                    `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            async function refreshData() {
                const result = await makeRequest('/api/data/refresh', 'POST');
                document.getElementById('data-result').innerHTML = 
                    `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            async function getDataSummary() {
                const result = await makeRequest('/api/data/summary');
                document.getElementById('data-result').innerHTML = 
                    `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            async function analyzePatterns() {
                const result = await makeRequest('/api/patterns/analyze', 'POST');
                document.getElementById('patterns-result').innerHTML = 
                    `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            async function getPatterns() {
                const result = await makeRequest('/api/patterns/detected');
                document.getElementById('patterns-result').innerHTML = 
                    `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            async function generateAllBots() {
                const result = await makeRequest('/api/bots/generate-all', 'POST');
                document.getElementById('bots-result').innerHTML = 
                    `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            async function installAllEnvironments() {
                const result = await makeRequest('/api/bots/install-all', 'POST');
                document.getElementById('bots-result').innerHTML = 
                    `<h4>🔧 Installing Environments...</h4><pre>${JSON.stringify(result, null, 2)}</pre>`;
                
                // Auto-refresh status every 10 seconds during installation
                if (result.success) {
                    setTimeout(checkInstallationStatus, 10000);
                }
            }
            
            async function checkInstallationStatus() {
                const result = await makeRequest('/api/installation/status');
                if (result.success) {
                    document.getElementById('bots-result').innerHTML += 
                        `<h4>📊 Installation Status:</h4><pre>${JSON.stringify(result, null, 2)}</pre>`;
                }
            }
            
            async function verifyEnvironments() {
                const bots = await makeRequest('/api/bots/list');
                if (bots.data && bots.data.bots && bots.data.bots.length > 0) {
                    let verificationResults = '<h4>🔍 Environment Verification:</h4>';
                    
                    for (const bot of bots.data.bots) {
                        const verification = await makeRequest(`/api/bots/${bot.id}/verify`);
                        const status = verification.data?.environment_ready ? '✅' : '❌';
                        verificationResults += `<p>${status} ${bot.pattern_name || bot.id}</p>`;
                    }
                    
                    document.getElementById('bots-result').innerHTML = verificationResults;
                } else {
                    document.getElementById('bots-result').innerHTML = 
                        '<p>No bots found. Generate some bots first!</p>';
                }
            }
            
            async function listBots() {
                const result = await makeRequest('/api/bots/list');
                document.getElementById('bots-result').innerHTML = 
                    `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            async function getBotCode() {
                const bots = await makeRequest('/api/bots/list');
                if (bots.data && bots.data.bots && bots.data.bots.length > 0) {
                    const firstBot = bots.data.bots[0];
                    const code = await makeRequest(`/api/bots/${firstBot.id}/code`);
                    document.getElementById('bots-result').innerHTML = 
                        `<h4>Code for ${firstBot.pattern_name}:</h4><pre>${JSON.stringify(code, null, 2)}</pre>`;
                } else {
                    document.getElementById('bots-result').innerHTML = 
                        '<p>No bots found. Generate some bots first!</p>';
                }
            }
            
            async function executeFirstBot() {
                const bots = await makeRequest('/api/bots/list');
                if (bots.data && bots.data.bots && bots.data.bots.length > 0) {
                    const firstBot = bots.data.bots[0];
                    
                    // First validate the bot
                    const validation = await makeRequest(`/api/bots/${firstBot.id}/validate`, 'POST');
                    
                    if (validation.data && validation.data.is_valid) {
                        // Execute the bot
                        const execution = await makeRequest(`/api/bots/${firstBot.id}/execute`, 'POST', {
                            bot_id: firstBot.id,
                            timeout_minutes: 10
                        });
                        
                        document.getElementById('execution-result').innerHTML = 
                            `<h4>🚀 Executing ${firstBot.pattern_name || firstBot.id}:</h4><pre>${JSON.stringify(execution, null, 2)}</pre>`;
                        
                        // Auto-refresh execution status
                        if (execution.success && execution.data.execution_id) {
                            setTimeout(() => checkExecutionStatus(execution.data.execution_id), 5000);
                        }
                    } else {
                        document.getElementById('execution-result').innerHTML = 
                            `<h4>❌ Validation Failed:</h4><pre>${JSON.stringify(validation, null, 2)}</pre>`;
                    }
                } else {
                    document.getElementById('execution-result').innerHTML = 
                        '<p>No bots found. Generate and install some bots first!</p>';
                }
            }
            
            async function debugFirstBot() {
                const bots = await makeRequest('/api/bots/list');
                if (bots.data && bots.data.bots && bots.data.bots.length > 0) {
                    const firstBot = bots.data.bots[0];
                    const debug = await makeRequest(`/api/bots/${firstBot.id}/debug`);
                    document.getElementById('execution-result').innerHTML = 
                        `<h4>🔍 Debug Info for ${firstBot.pattern_name || firstBot.id}:</h4><pre>${JSON.stringify(debug, null, 2)}</pre>`;
                } else {
                    document.getElementById('execution-result').innerHTML = 
                        '<p>No bots found. Generate some bots first!</p>';
                }
            }
            
            async function checkExecutionStatus(executionId) {
                const status = await makeRequest(`/api/executions/${executionId}/status`);
                document.getElementById('execution-result').innerHTML += 
                    `<h4>📊 Status Update:</h4><pre>${JSON.stringify(status, null, 2)}</pre>`;
                
                // Continue checking if still running
                if (status.data && status.data.status === 'running') {
                    setTimeout(() => checkExecutionStatus(executionId), 5000);
                }
            }
            
            async function getRunningExecutions() {
                const result = await makeRequest('/api/executions/running');
                document.getElementById('execution-result').innerHTML = 
                    `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            async function getExecutionHistory() {
                const result = await makeRequest('/api/executions/history');
                document.getElementById('execution-result').innerHTML = 
                    `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            async function getExecutionStats() {
                const result = await makeRequest('/api/executions/stats');
                document.getElementById('execution-result').innerHTML = 
                    `<pre>${JSON.stringify(result, null, 2)}</pre>`;
            }
            
            async function getLastExecutionLogs() {
                const history = await makeRequest('/api/executions/history?limit=1');
                if (history.data && history.data.execution_history && history.data.execution_history.length > 0) {
                    const lastExecution = history.data.execution_history[0];
                    const logs = await makeRequest(`/api/executions/${lastExecution.execution_id}/logs`);
                    document.getElementById('execution-result').innerHTML = 
                        `<h4>📋 Last Execution Logs:</h4><pre>${JSON.stringify(logs, null, 2)}</pre>`;
                } else {
                    document.getElementById('execution-result').innerHTML = 
                        '<p>No execution history found.</p>';
                }
            }
        </script>
    </body>
    </html>
    """)

# Run the server
if __name__ == "__main__":
    print("🚀 Starting Workflow Observer AI Backend Server...")
    print("📍 Dashboard will be available at: http://localhost:8000")
    print("📋 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )