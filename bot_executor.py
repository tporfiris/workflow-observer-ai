"""
Bot Execution Engine for Workflow Observer AI

This module manages the execution of generated automation bots. It handles:
- Running bots in their isolated virtual environments
- Process management and monitoring
- Real-time output streaming and logging
- Status tracking and error handling
- Scheduled execution and automation triggers
- Bot lifecycle management (start, stop, restart)

The execution engine ensures bots run safely in their configured environments
while providing comprehensive monitoring and control capabilities.
"""

import subprocess
import asyncio
import threading
import json
import logging
import signal
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import time
import os
import queue
import sys
from enum import Enum
import uuid


class BotStatus(Enum):
    """Bot execution status enumeration"""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    COMPLETED = "completed"


class BotExecutor:
    """
    Manages execution of automation bots with comprehensive monitoring.
    
    This class provides a complete bot execution environment with process
    management, logging, status tracking, and real-time monitoring capabilities.
    """
    
    def __init__(self, base_dir: str = "generated_bots"):
        """
        Initialize the bot execution engine.
        
        Args:
            base_dir (str): Base directory containing generated bots
        """
        
        self.base_dir = Path(base_dir)
        self.running_bots = {}  # bot_id -> execution info
        self.execution_history = []  # List of past executions
        self.bot_schedules = {}  # bot_id -> schedule info
        
        # Setup logging
        self._setup_logging()
        
        # Output queues for real-time streaming
        self.output_queues = {}  # execution_id -> queue
        
        # Process monitoring
        self.process_monitor_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_processes, daemon=True)
        self.monitor_thread.start()
        
        print("🚀 Bot Execution Engine initialized")
        print(f"📁 Base directory: {self.base_dir}")
    
    def _setup_logging(self):
        """Setup logging for execution tracking."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create bot execution specific logger
        self.logger = logging.getLogger("bot_executor")
        self.logger.setLevel(logging.INFO)
        
        # File handler for execution logs
        file_handler = logging.FileHandler(log_dir / "bot_executions.log")
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers if not already present
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def _get_bot_paths(self, bot_id: str) -> Dict[str, Path]:
        """
        Get important paths for a bot.
        
        Args:
            bot_id (str): Bot identifier
            
        Returns:
            Dict with bot paths
        """
        
        bot_dir = self.base_dir / bot_id
        
        # Look for Python script (try different naming patterns)
        script_candidates = [
            bot_dir / f"{bot_id}.py",
            bot_dir / "main.py",
            bot_dir / "bot.py",
            bot_dir / "automation.py"
        ]
        
        script_file = None
        for candidate in script_candidates:
            if candidate.exists():
                script_file = candidate
                break
        
        # Get virtual environment paths
        venv_dir = bot_dir / "venv"
        
        if os.name == 'nt':  # Windows
            python_exe = venv_dir / "Scripts" / "python.exe"
            activate_script = venv_dir / "Scripts" / "activate.bat"
        else:  # Unix/Linux/macOS - Use more thorough detection
            python_candidates = [
                venv_dir / "bin" / "python",
                venv_dir / "bin" / "python3",
                venv_dir / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}",
                venv_dir / "bin" / f"python{sys.version_info.major}"
            ]
            
            # Find the first existing executable
            python_exe = None
            for candidate in python_candidates:
                if candidate.exists():
                    # Verify it's actually executable
                    try:
                        test_result = subprocess.run([
                            str(candidate), "--version"
                        ], capture_output=True, text=True, timeout=10)
                        if test_result.returncode == 0:
                            python_exe = candidate
                            self.logger.info(f"Found working Python executable: {python_exe}")
                            break
                    except Exception as e:
                        self.logger.warning(f"Python candidate {candidate} failed test: {e}")
                        continue
            
            # Fallback to default if none found
            if python_exe is None:
                python_exe = venv_dir / "bin" / "python"
                self.logger.warning(f"No working Python executable found, using default: {python_exe}")
            
            activate_script = venv_dir / "bin" / "activate"
        
        return {
            "bot_dir": bot_dir,
            "script_file": script_file,
            "venv_dir": venv_dir,
            "python_exe": python_exe,
            "activate_script": activate_script,
            "config_file": bot_dir / "config.json",
            "requirements_file": bot_dir / "requirements.txt",
            "environment_info": bot_dir / "environment_info.json"
        }
    
    def _validate_bot_environment(self, bot_id: str) -> tuple[bool, str]:
        """
        Validate that a bot's environment is ready for execution.
        
        Args:
            bot_id (str): Bot identifier
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        
        paths = self._get_bot_paths(bot_id)
        
        # Check bot directory exists
        if not paths["bot_dir"].exists():
            return False, f"Bot directory not found: {paths['bot_dir']}"
        
        # Check script file exists
        if not paths["script_file"] or not paths["script_file"].exists():
            return False, f"Bot script file not found. Looked for: {paths['script_file']}"
        
        # Check virtual environment exists
        if not paths["venv_dir"].exists():
            return False, f"Virtual environment not found: {paths['venv_dir']}"
        
        # Check Python executable exists - use absolute path
        abs_python_exe = Path(paths["python_exe"]).resolve()
        if not abs_python_exe.exists():
            return False, f"Python executable not found at absolute path: {abs_python_exe}"
        
        # Test Python executable with absolute path
        try:
            result = subprocess.run([
                str(abs_python_exe), "--version"
            ], capture_output=True, text=True, timeout=10, cwd=str(paths["bot_dir"].resolve()))
            
            if result.returncode != 0:
                return False, f"Python executable test failed: {result.stderr}"
            
            self.logger.info(f"Python executable validated: {abs_python_exe} -> {result.stdout.strip()}")
                
        except Exception as e:
            return False, f"Python executable test error: {e}"
        
        return True, f"Environment validation successful - Python: {abs_python_exe}"
    
    def execute_bot(self, bot_id: str, config_overrides: Dict[str, Any] = None, 
                   timeout_minutes: int = 60) -> Dict[str, Any]:
        """
        Execute a bot in its virtual environment.
        
        Args:
            bot_id (str): Bot identifier
            config_overrides (dict): Optional configuration overrides
            timeout_minutes (int): Execution timeout in minutes
            
        Returns:
            Dict with execution results
        """
        
        # Generate unique execution ID
        execution_id = f"{bot_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self.logger.info(f"Starting bot execution: {bot_id} (execution_id: {execution_id})")
        
        # Validate environment
        is_valid, validation_error = self._validate_bot_environment(bot_id)
        if not is_valid:
            error_result = {
                "execution_id": execution_id,
                "bot_id": bot_id,
                "status": BotStatus.FAILED.value,
                "error": f"Environment validation failed: {validation_error}",
                "started_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "duration_seconds": 0
            }
            self.execution_history.append(error_result)
            return error_result
        
        # Get bot paths
        paths = self._get_bot_paths(bot_id)
        
        # Convert to absolute paths to ensure consistency
        abs_paths = {}
        for key, path in paths.items():
            if isinstance(path, Path):
                abs_paths[key] = path.resolve()
            else:
                abs_paths[key] = path
        
        # Create execution info
        execution_info = {
            "execution_id": execution_id,
            "bot_id": bot_id,
            "status": BotStatus.STARTING.value,
            "started_at": datetime.now().isoformat(),
            "timeout_minutes": timeout_minutes,
            "config_overrides": config_overrides or {},
            "paths": {k: str(v) for k, v in abs_paths.items()},
            "process": None,
            "stdout_lines": [],
            "stderr_lines": [],
            "return_code": None
        }
        
        # Add to running bots
        self.running_bots[execution_id] = execution_info
        
        # Create output queue for real-time streaming
        self.output_queues[execution_id] = queue.Queue()
        
        try:
            # Start execution in background thread
            execution_thread = threading.Thread(
                target=self._execute_bot_thread,
                args=(execution_id, abs_paths, config_overrides, timeout_minutes),
                daemon=True
            )
            execution_thread.start()
            
            return {
                "execution_id": execution_id,
                "bot_id": bot_id,
                "status": BotStatus.STARTING.value,
                "message": f"Bot execution started",
                "started_at": execution_info["started_at"],
                "timeout_minutes": timeout_minutes
            }
            
        except Exception as e:
            error_msg = f"Failed to start bot execution: {e}"
            self.logger.error(error_msg)
            
            # Update status
            execution_info["status"] = BotStatus.FAILED.value
            execution_info["error"] = error_msg
            execution_info["completed_at"] = datetime.now().isoformat()
            
            # Move to history
            self.execution_history.append(execution_info)
            del self.running_bots[execution_id]
            
            return {
                "execution_id": execution_id,
                "bot_id": bot_id,
                "status": BotStatus.FAILED.value,
                "error": error_msg,
                "started_at": execution_info["started_at"],
                "completed_at": execution_info["completed_at"]
            }
    
    def _execute_bot_thread(self, execution_id: str, paths: Dict[str, Path], 
                           config_overrides: Dict[str, Any], timeout_minutes: int):
        """
        Execute bot in a separate thread.
        
        Args:
            execution_id (str): Unique execution identifier
            paths (dict): Bot file paths
            config_overrides (dict): Configuration overrides
            timeout_minutes (int): Execution timeout
        """
        
        execution_info = self.running_bots[execution_id]
        bot_id = execution_info["bot_id"]
        
        try:
            # Apply configuration overrides if provided
            if config_overrides:
                self._apply_config_overrides(Path(paths["config_file"]), config_overrides)
            
            # Use the absolute paths that were already resolved
            python_exe = Path(paths["python_exe"])
            script_file = Path(paths["script_file"])
            bot_dir = Path(paths["bot_dir"])
            
            # Verify all paths exist (they should already be absolute)
            if not python_exe.exists():
                raise Exception(f"Python executable not found: {python_exe}")
            
            if not script_file.exists():
                raise Exception(f"Script file not found: {script_file}")
            
            if not bot_dir.exists():
                raise Exception(f"Bot directory not found: {bot_dir}")
            
            # Prepare execution command
            cmd = [str(python_exe), str(script_file)]
            
            self.logger.info(f"Executing command: {' '.join(cmd)}")
            self.logger.info(f"Working directory: {bot_dir}")
            self.logger.info(f"Python executable: {python_exe} (exists: {python_exe.exists()})")
            self.logger.info(f"Script file: {script_file} (exists: {script_file.exists()})")
            
            # Update status
            execution_info["status"] = BotStatus.RUNNING.value
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                cwd=str(bot_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=os.environ.copy()  # Ensure full environment is passed
            )
            
            execution_info["process"] = process
            execution_info["pid"] = process.pid
            
            self.logger.info(f"Bot process started: PID {process.pid}")
            
            # Stream output in real-time
            self._stream_process_output(execution_id, process, timeout_minutes)
            
        except Exception as e:
            error_msg = f"Bot execution error: {e}"
            self.logger.error(error_msg)
            
            execution_info["status"] = BotStatus.FAILED.value
            execution_info["error"] = error_msg
            execution_info["completed_at"] = datetime.now().isoformat()
            
            # Calculate duration
            start_time = datetime.fromisoformat(execution_info["started_at"].replace('Z', '+00:00'))
            duration = (datetime.now() - start_time.replace(tzinfo=None)).total_seconds()
            execution_info["duration_seconds"] = duration
            
            # Move to history
            self.execution_history.append(execution_info.copy())
            del self.running_bots[execution_id]
            
            # Clean up output queue
            if execution_id in self.output_queues:
                del self.output_queues[execution_id]
    
    def _stream_process_output(self, execution_id: str, process: subprocess.Popen, timeout_minutes: int):
        """
        Stream process output and handle completion.
        
        Args:
            execution_id (str): Execution identifier
            process (subprocess.Popen): Running process
            timeout_minutes (int): Timeout in minutes
        """
        
        execution_info = self.running_bots[execution_id]
        output_queue = self.output_queues.get(execution_id)
        
        try:
            # Wait for process with timeout
            timeout_seconds = timeout_minutes * 60
            start_time = time.time()
            
            # Read output in real-time
            while process.poll() is None:
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    self.logger.warning(f"Bot execution timeout reached: {execution_id}")
                    process.terminate()
                    time.sleep(5)  # Give it time to terminate gracefully
                    if process.poll() is None:
                        process.kill()  # Force kill if still running
                    
                    execution_info["status"] = BotStatus.FAILED.value
                    execution_info["error"] = f"Execution timed out after {timeout_minutes} minutes"
                    break
                
                # Read stdout
                if process.stdout:
                    try:
                        line = process.stdout.readline()
                        if line:
                            line = line.rstrip()
                            execution_info["stdout_lines"].append(line)
                            if output_queue:
                                output_queue.put({"type": "stdout", "line": line, "timestamp": datetime.now().isoformat()})
                            self.logger.info(f"[{execution_id}] STDOUT: {line}")
                    except:
                        pass
                
                # Read stderr
                if process.stderr:
                    try:
                        line = process.stderr.readline()
                        if line:
                            line = line.rstrip()
                            execution_info["stderr_lines"].append(line)
                            if output_queue:
                                output_queue.put({"type": "stderr", "line": line, "timestamp": datetime.now().isoformat()})
                            self.logger.warning(f"[{execution_id}] STDERR: {line}")
                    except:
                        pass
                
                time.sleep(0.1)  # Small delay to prevent CPU spinning
            
            # Get final return code
            return_code = process.returncode
            execution_info["return_code"] = return_code
            
            # Determine final status
            if execution_info["status"] != BotStatus.FAILED.value:  # Not already failed due to timeout
                if return_code == 0:
                    execution_info["status"] = BotStatus.COMPLETED.value
                    self.logger.info(f"Bot execution completed successfully: {execution_id}")
                else:
                    execution_info["status"] = BotStatus.FAILED.value
                    execution_info["error"] = f"Bot exited with code {return_code}"
                    self.logger.error(f"Bot execution failed with exit code {return_code}: {execution_id}")
            
            # Set completion time
            execution_info["completed_at"] = datetime.now().isoformat()
            
            # Calculate duration
            start_time = datetime.fromisoformat(execution_info["started_at"].replace('Z', '+00:00'))
            duration = (datetime.now() - start_time.replace(tzinfo=None)).total_seconds()
            execution_info["duration_seconds"] = duration
            
            # Read any remaining output
            try:
                if process.stdout:
                    remaining_stdout = process.stdout.read()
                    if remaining_stdout:
                        for line in remaining_stdout.splitlines():
                            execution_info["stdout_lines"].append(line)
                
                if process.stderr:
                    remaining_stderr = process.stderr.read()
                    if remaining_stderr:
                        for line in remaining_stderr.splitlines():
                            execution_info["stderr_lines"].append(line)
            except:
                pass
            
        except Exception as e:
            self.logger.error(f"Error streaming process output: {e}")
            execution_info["status"] = BotStatus.FAILED.value
            execution_info["error"] = f"Output streaming error: {e}"
            execution_info["completed_at"] = datetime.now().isoformat()
        
        finally:
            # Clean up
            try:
                if process and process.poll() is None:
                    process.terminate()
            except:
                pass
            
            # Move to history
            self.execution_history.append(execution_info.copy())
            if execution_id in self.running_bots:
                del self.running_bots[execution_id]
            
            # Clean up output queue
            if execution_id in self.output_queues:
                del self.output_queues[execution_id]
    
    def _apply_config_overrides(self, config_file: Path, overrides: Dict[str, Any]):
        """
        Apply configuration overrides to bot config file.
        
        Args:
            config_file (Path): Path to config.json
            overrides (dict): Configuration overrides
        """
        
        try:
            # Load existing config
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Apply overrides
            def update_nested_dict(d, overrides):
                for key, value in overrides.items():
                    if isinstance(value, dict) and key in d and isinstance(d[key], dict):
                        update_nested_dict(d[key], value)
                    else:
                        d[key] = value
            
            update_nested_dict(config, overrides)
            
            # Save updated config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Applied config overrides: {overrides}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply config overrides: {e}")
    
    def stop_bot(self, execution_id: str) -> Dict[str, Any]:
        """
        Stop a running bot execution.
        
        Args:
            execution_id (str): Execution identifier
            
        Returns:
            Dict with stop results
        """
        
        if execution_id not in self.running_bots:
            return {
                "execution_id": execution_id,
                "status": "not_found",
                "error": "Execution not found or already completed"
            }
        
        execution_info = self.running_bots[execution_id]
        process = execution_info.get("process")
        
        if not process:
            return {
                "execution_id": execution_id,
                "status": "no_process",
                "error": "No process found for this execution"
            }
        
        try:
            self.logger.info(f"Stopping bot execution: {execution_id}")
            
            # Update status
            execution_info["status"] = BotStatus.STOPPING.value
            
            # Try graceful termination first
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if still running
                self.logger.warning(f"Force killing bot process: {execution_id}")
                process.kill()
                process.wait()
            
            # Update final status
            execution_info["status"] = BotStatus.STOPPED.value
            execution_info["completed_at"] = datetime.now().isoformat()
            execution_info["return_code"] = process.returncode
            
            # Calculate duration
            start_time = datetime.fromisoformat(execution_info["started_at"].replace('Z', '+00:00'))
            duration = (datetime.now() - start_time.replace(tzinfo=None)).total_seconds()
            execution_info["duration_seconds"] = duration
            
            # Move to history
            self.execution_history.append(execution_info.copy())
            del self.running_bots[execution_id]
            
            # Clean up output queue
            if execution_id in self.output_queues:
                del self.output_queues[execution_id]
            
            return {
                "execution_id": execution_id,
                "status": "stopped",
                "message": "Bot execution stopped successfully",
                "stopped_at": execution_info["completed_at"]
            }
            
        except Exception as e:
            error_msg = f"Failed to stop bot execution: {e}"
            self.logger.error(error_msg)
            
            return {
                "execution_id": execution_id,
                "status": "error",
                "error": error_msg
            }
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get status of a specific execution.
        
        Args:
            execution_id (str): Execution identifier
            
        Returns:
            Dict with execution status
        """
        
        # Check running bots first
        if execution_id in self.running_bots:
            execution_info = self.running_bots[execution_id].copy()
            # Remove process object and other non-serializable objects for JSON serialization
            non_serializable_keys = ["process"]
            for key in non_serializable_keys:
                if key in execution_info:
                    del execution_info[key]
            return execution_info
        
        # Check execution history
        for execution in self.execution_history:
            if execution["execution_id"] == execution_id:
                result = execution.copy()
                # Clean any non-serializable objects
                for key in ["process"]:
                    if key in result:
                        del result[key]
                return result
        
        return {
            "execution_id": execution_id,
            "status": "not_found",
            "error": "Execution not found"
        }
    
    def get_running_bots(self) -> List[Dict[str, Any]]:
        """Get list of currently running bots."""
        running_list = []
        
        for execution_id, execution_info in self.running_bots.items():
            bot_info = execution_info.copy()
            # Remove process object for JSON serialization
            if "process" in bot_info:
                del bot_info["process"]
            running_list.append(bot_info)
        
        return running_list
    
    def get_execution_history(self, bot_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get execution history.
        
        Args:
            bot_id (str): Optional filter by bot ID
            limit (int): Maximum number of results
            
        Returns:
            List of execution records
        """
        
        history = self.execution_history.copy()
        
        # Filter by bot_id if specified
        if bot_id:
            history = [ex for ex in history if ex.get("bot_id") == bot_id]
        
        # Sort by start time (most recent first)
        history.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        
        # Limit results
        return history[:limit]
    
    def get_bot_logs(self, execution_id: str) -> Dict[str, Any]:
        """
        Get logs for a specific execution.
        
        Args:
            execution_id (str): Execution identifier
            
        Returns:
            Dict with log data
        """
        
        execution_info = self.get_execution_status(execution_id)
        
        if execution_info.get("status") == "not_found":
            return execution_info
        
        return {
            "execution_id": execution_id,
            "bot_id": execution_info.get("bot_id"),
            "status": execution_info.get("status"),
            "stdout_lines": execution_info.get("stdout_lines", []),
            "stderr_lines": execution_info.get("stderr_lines", []),
            "return_code": execution_info.get("return_code"),
            "started_at": execution_info.get("started_at"),
            "completed_at": execution_info.get("completed_at"),
            "duration_seconds": execution_info.get("duration_seconds")
        }
    
    def _monitor_processes(self):
        """Monitor running processes and clean up orphaned executions."""
        while self.process_monitor_active:
            try:
                # Check for orphaned executions
                to_remove = []
                
                for execution_id, execution_info in self.running_bots.items():
                    process = execution_info.get("process")
                    
                    if process and process.poll() is not None:
                        # Process has finished
                        self.logger.info(f"Detected finished process: {execution_id}")
                        to_remove.append(execution_id)
                
                # Clean up finished executions
                for execution_id in to_remove:
                    if execution_id in self.running_bots:
                        execution_info = self.running_bots[execution_id]
                        
                        # Update completion info if not already set
                        if "completed_at" not in execution_info:
                            execution_info["completed_at"] = datetime.now().isoformat()
                            
                            # Calculate duration
                            start_time = datetime.fromisoformat(execution_info["started_at"].replace('Z', '+00:00'))
                            duration = (datetime.now() - start_time.replace(tzinfo=None)).total_seconds()
                            execution_info["duration_seconds"] = duration
                        
                        # Move to history
                        self.execution_history.append(execution_info.copy())
                        del self.running_bots[execution_id]
                        
                        # Clean up output queue
                        if execution_id in self.output_queues:
                            del self.output_queues[execution_id]
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Process monitor error: {e}")
                time.sleep(10)  # Wait longer on error
    
    def shutdown(self):
        """Shutdown the execution engine."""
        self.logger.info("Shutting down Bot Execution Engine...")
        
        # Stop process monitoring
        self.process_monitor_active = False
        
        # Stop all running bots
        running_executions = list(self.running_bots.keys())
        for execution_id in running_executions:
            try:
                self.stop_bot(execution_id)
            except Exception as e:
                self.logger.error(f"Error stopping bot {execution_id}: {e}")
        
        self.logger.info("Bot Execution Engine shutdown complete")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total_executions = len(self.execution_history)
        running_count = len(self.running_bots)
        
        if total_executions > 0:
            completed_count = len([ex for ex in self.execution_history if ex.get("status") == BotStatus.COMPLETED.value])
            failed_count = len([ex for ex in self.execution_history if ex.get("status") == BotStatus.FAILED.value])
            success_rate = (completed_count / total_executions) * 100
        else:
            completed_count = failed_count = 0
            success_rate = 0
        
        return {
            "total_executions": total_executions,
            "running_count": running_count,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "success_rate": round(success_rate, 2),
            "last_updated": datetime.now().isoformat()
        }


def main():
    """Test function for the bot executor."""
    
    print("🚀 Bot Execution Engine Test")
    print("=" * 60)
    
    # Initialize executor
    executor = BotExecutor()
    
    # Check for bot directories
    bots_dir = Path("generated_bots")
    if not bots_dir.exists():
        print("❌ No generated_bots directory found")
        return
    
    bot_dirs = [d for d in bots_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not bot_dirs:
        print("❌ No bot directories found")
        return
    
    print(f"🤖 Found {len(bot_dirs)} bot directories:")
    for i, bot_dir in enumerate(bot_dirs, 1):
        # Check if environment is ready
        is_valid, validation_msg = executor._validate_bot_environment(bot_dir.name)
        status = "✅" if is_valid else "❌"
        print(f"  {i}. {bot_dir.name} {status}")
        if not is_valid:
            print(f"      Error: {validation_msg}")
    
    # Test execution
    valid_bots = [d.name for d in bot_dirs if executor._validate_bot_environment(d.name)[0]]
    
    if valid_bots:
        bot_to_test = valid_bots[0]
        print(f"\n🧪 Testing execution with bot: {bot_to_test}")
        
        # Start execution
        result = executor.execute_bot(bot_to_test, timeout_minutes=5)
        print(f"Execution started: {result}")
        
        # Monitor for a bit
        execution_id = result.get("execution_id")
        if execution_id:
            for i in range(10):
                time.sleep(1)
                status = executor.get_execution_status(execution_id)
                print(f"Status update {i+1}: {status.get('status')}")
                
                if status.get("status") in [BotStatus.COMPLETED.value, BotStatus.FAILED.value]:
                    print(f"Execution finished: {status}")
                    break
    else:
        print("❌ No valid bots found for testing")
        print("💡 Make sure bots have their environments set up first")
    
    # Show stats
    stats = executor.get_execution_stats()
    print(f"\n📊 Execution Statistics:")
    print(f"   Total executions: {stats['total_executions']}")
    print(f"   Currently running: {stats['running_count']}")
    print(f"   Success rate: {stats['success_rate']}%")
    
    # Cleanup
    executor.shutdown()
    print("\n👋 Test complete!")


if __name__ == "__main__":
    main()