"""
Package Installation System for Workflow Observer AI

This module automatically installs Python packages and manages virtual environments
for generated automation bots. It handles:
- Virtual environment creation and management
- Automatic package installation from requirements.txt
- Dependency conflict resolution
- Cross-platform compatibility (Windows, macOS, Linux)
- Installation progress tracking and error handling

The system ensures each bot has its own isolated environment with all
necessary dependencies installed correctly.
"""

import subprocess
import sys
import os
import venv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import platform
import tempfile
import shutil
from datetime import datetime
import threading
import queue
import time


class PackageInstaller:
    """
    Manages package installation and virtual environments for automation bots.
    
    This class handles the complete lifecycle of setting up Python environments
    for generated automation bots, including virtual environment creation,
    package installation, and dependency management.
    """
    
    def __init__(self, base_dir: str = "generated_bots"):
        """
        Initialize the package installer.
        
        Args:
            base_dir (str): Base directory containing generated bots
        """
        
        self.base_dir = Path(base_dir)
        self.system_info = self._get_system_info()
        self.installation_log = []
        self.active_installations = {}
        
        # Setup logging
        self._setup_logging()
        
        print("📦 Package Installation System initialized")
        print(f"🖥️  System: {self.system_info['platform']} {self.system_info['architecture']}")
        print(f"🐍 Python: {self.system_info['python_version']}")
    
    def _setup_logging(self):
        """Setup logging for installation tracking."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "package_installer.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for platform-specific handling."""
        return {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "python_executable": sys.executable
        }
    
    def _get_venv_paths(self, bot_dir: Path) -> Dict[str, Path]:
        """
        Get virtual environment paths for different platforms.
        
        Args:
            bot_dir: Bot directory path
            
        Returns:
            Dict with venv paths
        """
        
        venv_dir = bot_dir / "venv"
        
        if self.system_info["platform"] == "Windows":
            return {
                "venv_dir": venv_dir,
                "python": venv_dir / "Scripts" / "python.exe",
                "pip": venv_dir / "Scripts" / "pip.exe",
                "activate": venv_dir / "Scripts" / "activate.bat"
            }
        else:
            # macOS and Linux
            python_candidates = [
                venv_dir / "bin" / "python",
                venv_dir / "bin" / "python3",
                venv_dir / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}"
            ]
            
            # Use the first existing python executable, or default to "python"
            python_exe = venv_dir / "bin" / "python"
            for candidate in python_candidates:
                if candidate.exists():
                    python_exe = candidate
                    break
            
            return {
                "venv_dir": venv_dir,
                "python": python_exe,
                "pip": venv_dir / "bin" / "pip",
                "activate": venv_dir / "bin" / "activate"
            }
    
    def create_virtual_environment(self, bot_dir: Path) -> Dict[str, Any]:
        """
        Create a virtual environment for a bot.
        
        Args:
            bot_dir: Directory containing the bot
            
        Returns:
            Dict with creation results
        """
        
        self.logger.info(f"Creating virtual environment for {bot_dir.name}")
        
        try:
            venv_paths = self._get_venv_paths(bot_dir)
            venv_dir = venv_paths["venv_dir"]
            
            # Remove existing venv if it exists
            if venv_dir.exists():
                self.logger.info(f"Removing existing virtual environment: {venv_dir}")
                shutil.rmtree(venv_dir)
                # Wait a moment for filesystem cleanup
                time.sleep(1)
            
            # Create new virtual environment
            self.logger.info(f"Creating virtual environment: {venv_dir}")
            venv.create(venv_dir, with_pip=True, clear=True)
            
            # Wait for virtual environment to be fully created
            max_wait = 30  # 30 seconds max wait
            wait_count = 0
            while not venv_paths["python"].exists() and wait_count < max_wait:
                time.sleep(1)
                wait_count += 1
                self.logger.info(f"Waiting for Python executable... ({wait_count}/{max_wait})")
            
            # Verify creation
            if not venv_paths["python"].exists():
                # Try alternative path detection for different platforms
                alt_paths = [
                    venv_dir / "bin" / "python3",
                    venv_dir / "Scripts" / "python3.exe",
                    venv_dir / "bin" / "python",
                    venv_dir / "Scripts" / "python.exe"
                ]
                
                python_exe = None
                for alt_path in alt_paths:
                    if alt_path.exists():
                        python_exe = alt_path
                        self.logger.info(f"Found Python executable at: {python_exe}")
                        # Update the paths dict
                        venv_paths["python"] = python_exe
                        if self.system_info["platform"] == "Windows":
                            venv_paths["pip"] = python_exe.parent / "pip.exe"
                        else:
                            venv_paths["pip"] = python_exe.parent / "pip"
                        break
                
                if python_exe is None:
                    raise Exception(f"Python executable not found after creation. Tried: {[str(p) for p in alt_paths]}")
            
            # Test Python executable
            self.logger.info("Testing Python executable...")
            test_result = subprocess.run([
                str(venv_paths["python"]), "--version"
            ], capture_output=True, text=True, timeout=30)
            
            if test_result.returncode != 0:
                raise Exception(f"Python executable test failed: {test_result.stderr}")
            
            self.logger.info(f"Python version: {test_result.stdout.strip()}")
            
            # Upgrade pip in the new environment
            self.logger.info("Upgrading pip in virtual environment")
            result = subprocess.run([
                str(venv_paths["python"]), "-m", "pip", "install", "--upgrade", "pip"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.logger.warning(f"Pip upgrade warning: {result.stderr}")
            
            return {
                "status": "success",
                "venv_dir": str(venv_dir),
                "python_path": str(venv_paths["python"]),
                "pip_path": str(venv_paths["pip"]),
                "python_version": test_result.stdout.strip(),
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Failed to create virtual environment: {e}"
            self.logger.error(error_msg)
            return {
                "status": "failed",
                "error": error_msg,
                "created_at": datetime.now().isoformat()
            }
    
    def install_packages(self, bot_dir: Path, requirements_file: str = "requirements.txt") -> Dict[str, Any]:
        """
        Install packages from requirements.txt in the bot's virtual environment.
        
        Args:
            bot_dir: Bot directory containing requirements.txt
            requirements_file: Name of requirements file
            
        Returns:
            Dict with installation results
        """
        
        self.logger.info(f"Installing packages for {bot_dir.name}")
        
        try:
            venv_paths = self._get_venv_paths(bot_dir)
            req_file = bot_dir / requirements_file
            
            # Check if virtual environment exists
            if not venv_paths["venv_dir"].exists():
                raise Exception(f"Virtual environment not found: {venv_paths['venv_dir']}")
            
            # Debug: List contents of venv directory
            venv_contents = list(venv_paths["venv_dir"].iterdir())
            self.logger.info(f"Virtual environment contents: {[p.name for p in venv_contents]}")
            
            # For Unix systems, check bin directory
            if self.system_info["platform"] != "Windows":
                bin_dir = venv_paths["venv_dir"] / "bin"
                if bin_dir.exists():
                    bin_contents = list(bin_dir.iterdir())
                    self.logger.info(f"bin/ directory contents: {[p.name for p in bin_contents]}")
                else:
                    self.logger.error(f"bin/ directory not found in virtual environment")
            
            # Find the correct Python executable with more thorough search
            python_exe = None
            python_candidates = [
                venv_paths["python"],
                venv_paths["venv_dir"] / "bin" / "python3",
                venv_paths["venv_dir"] / "bin" / "python",
                venv_paths["venv_dir"] / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}",
                venv_paths["venv_dir"] / "Scripts" / "python.exe",
                venv_paths["venv_dir"] / "Scripts" / "python3.exe"
            ]
            
            self.logger.info(f"Searching for Python executable in candidates: {[str(p) for p in python_candidates]}")
            
            for candidate in python_candidates:
                self.logger.info(f"Checking candidate: {candidate} - Exists: {candidate.exists()}")
                if candidate.exists():
                    # Resolve symlinks to get the actual executable path
                    try:
                        resolved_path = candidate.resolve()
                        self.logger.info(f"Resolved path: {resolved_path} - Exists: {resolved_path.exists()}")
                        
                        # Use the resolved path if it exists, otherwise use original
                        if resolved_path.exists():
                            python_exe = resolved_path
                            self.logger.info(f"Using resolved Python executable: {python_exe}")
                        else:
                            python_exe = candidate
                            self.logger.info(f"Using original Python executable: {python_exe}")
                        break
                    except (OSError, RuntimeError) as e:
                        self.logger.warning(f"Failed to resolve symlink for {candidate}: {e}")
                        python_exe = candidate
                        break
            
            if python_exe is None:
                # Last resort: find any python* executable in bin or Scripts
                search_dirs = [
                    venv_paths["venv_dir"] / "bin",
                    venv_paths["venv_dir"] / "Scripts"
                ]
                
                for search_dir in search_dirs:
                    if search_dir.exists():
                        python_files = list(search_dir.glob("python*"))
                        self.logger.info(f"Found python* files in {search_dir}: {[p.name for p in python_files]}")
                        
                        # Filter for executable files
                        for py_file in python_files:
                            if py_file.is_file() and (py_file.suffix in ['.exe', ''] or py_file.stat().st_mode & 0o111):
                                python_exe = py_file
                                self.logger.info(f"Using found executable: {python_exe}")
                                break
                        
                        if python_exe:
                            break
            
            if python_exe is None:
                raise Exception(f"No Python executable found. Virtual environment may be corrupted. Contents: {[str(p) for p in venv_contents]}")
            
            # Test Python executable
            self.logger.info(f"Testing Python executable: {python_exe}")
            abs_python_exe = Path(python_exe).resolve()
            self.logger.info(f"Absolute Python path: {abs_python_exe}")
            
            test_result = subprocess.run([
                str(abs_python_exe), "--version"
            ], capture_output=True, text=True, timeout=30, env=os.environ.copy())
            
            if test_result.returncode != 0:
                raise Exception(f"Python executable test failed: {test_result.stderr}")
            
            self.logger.info(f"Python test successful: {test_result.stdout.strip()}")
            
            # Update python_exe to use the resolved path
            python_exe = abs_python_exe
            
            # Check if requirements file exists
            if not req_file.exists():
                raise Exception(f"Requirements file not found: {req_file}")
            
            # Read requirements for logging
            with open(req_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            self.logger.info(f"Installing {len(requirements)} packages: {', '.join(requirements)}")
            
            # Install packages with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Convert to absolute path and resolve any symlinks
                    abs_python_exe = Path(python_exe).resolve()
                    abs_req_file = Path(req_file).resolve()
                    abs_bot_dir = Path(bot_dir).resolve()
                    
                    cmd = [
                        str(abs_python_exe), "-m", "pip", "install", 
                        "-r", str(abs_req_file),
                        "--timeout", "300",
                        "--no-cache-dir"  # Avoid cache issues
                    ]
                    
                    self.logger.info(f"Attempt {attempt + 1}/{max_retries} - Running: {' '.join(cmd)}")
                    self.logger.info(f"Working directory: {abs_bot_dir}")
                    self.logger.info(f"Python executable path: {abs_python_exe}")
                    self.logger.info(f"Requirements file path: {abs_req_file}")
                    
                    result = subprocess.run(
                        cmd,
                        cwd=str(abs_bot_dir),
                        capture_output=True,
                        text=True,
                        timeout=600,  # 10 minute timeout
                        env=os.environ.copy()  # Ensure we have the full environment
                    )
                    
                    if result.returncode == 0:
                        # Success! Get installed packages list
                        list_cmd = [str(python_exe), "-m", "pip", "list", "--format=json"]
                        list_result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=30)
                        
                        installed_packages = []
                        if list_result.returncode == 0:
                            try:
                                packages_data = json.loads(list_result.stdout)
                                installed_packages = [{"name": p["name"], "version": p["version"]} for p in packages_data]
                            except json.JSONDecodeError:
                                pass
                        
                        success_result = {
                            "status": "success",
                            "packages_requested": requirements,
                            "packages_installed": installed_packages,
                            "installation_output": result.stdout,
                            "python_executable": str(python_exe),
                            "attempt": attempt + 1,
                            "installed_at": datetime.now().isoformat()
                        }
                        
                        self.logger.info(f"Successfully installed {len(requirements)} packages on attempt {attempt + 1}")
                        return success_result
                    else:
                        # Installation failed, log and potentially retry
                        error_msg = f"Installation attempt {attempt + 1} failed: {result.stderr}"
                        self.logger.warning(error_msg)
                        
                        if attempt == max_retries - 1:  # Last attempt
                            return {
                                "status": "failed",
                                "error": f"All {max_retries} installation attempts failed. Last error: {result.stderr}",
                                "stdout": result.stdout,
                                "stderr": result.stderr,
                                "packages_requested": requirements,
                                "python_executable": str(python_exe),
                                "attempts": max_retries,
                                "installed_at": datetime.now().isoformat()
                            }
                        else:
                            # Wait before retry
                            time.sleep(5)
                            
                except subprocess.TimeoutExpired:
                    if attempt == max_retries - 1:
                        return {
                            "status": "failed",
                            "error": f"Package installation timed out after {max_retries} attempts",
                            "python_executable": str(python_exe),
                            "attempts": max_retries,
                            "installed_at": datetime.now().isoformat()
                        }
                    else:
                        self.logger.warning(f"Installation attempt {attempt + 1} timed out, retrying...")
                        time.sleep(5)
                        
        except Exception as e:
            error_msg = f"Package installation error: {e}"
            self.logger.error(error_msg)
            return {
                "status": "failed",
                "error": error_msg,
                "installed_at": datetime.now().isoformat()
            }
    
    def setup_bot_environment(self, bot_dir: Path) -> Dict[str, Any]:
        """
        Complete environment setup for a bot (venv + packages).
        
        Args:
            bot_dir: Bot directory path
            
        Returns:
            Dict with complete setup results
        """
        
        bot_name = bot_dir.name
        self.logger.info(f"Setting up complete environment for bot: {bot_name}")
        
        # Track this installation
        installation_id = f"{bot_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_installations[installation_id] = {
            "bot_name": bot_name,
            "status": "in_progress",
            "started_at": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Create virtual environment
            print(f"🔨 Creating virtual environment for {bot_name}...")
            venv_result = self.create_virtual_environment(bot_dir)
            
            if venv_result["status"] != "success":
                raise Exception(f"Virtual environment creation failed: {venv_result.get('error')}")
            
            # Step 1.5: Wait a bit and verify the environment exists
            time.sleep(2)  # Give filesystem time to settle
            
            venv_paths = self._get_venv_paths(bot_dir)
            if not venv_paths["venv_dir"].exists():
                raise Exception(f"Virtual environment directory not found after creation: {venv_paths['venv_dir']}")
            
            # Step 2: Install packages
            print(f"📦 Installing packages for {bot_name}...")
            install_result = self.install_packages(bot_dir)
            
            if install_result["status"] != "success":
                raise Exception(f"Package installation failed: {install_result.get('error')}")
            
            # Step 3: Create activation script
            self._create_activation_script(bot_dir)
            
            # Step 4: Save environment info
            env_info = {
                "bot_name": bot_name,
                "setup_completed": True,
                "venv_info": venv_result,
                "packages_info": install_result,
                "system_info": self.system_info,
                "setup_date": datetime.now().isoformat()
            }
            
            env_file = bot_dir / "environment_info.json"
            with open(env_file, 'w') as f:
                json.dump(env_info, f, indent=2)
            
            # Update installation tracking
            self.active_installations[installation_id].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "packages_installed": len(install_result.get("packages_installed", [])),
                "result": "success"
            })
            
            success_result = {
                "status": "success",
                "bot_name": bot_name,
                "installation_id": installation_id,
                "venv_created": True,
                "packages_installed": True,
                "environment_file": str(env_file),
                "venv_path": venv_result.get("venv_dir"),
                "packages_count": len(install_result.get("packages_installed", [])),
                "python_executable": install_result.get("python_executable"),
                "setup_completed_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Environment setup completed successfully for {bot_name}")
            return success_result
            
        except Exception as e:
            error_msg = f"Environment setup failed for {bot_name}: {e}"
            self.logger.error(error_msg)
            
            # Update installation tracking
            self.active_installations[installation_id].update({
                "status": "failed",
                "failed_at": datetime.now().isoformat(),
                "error": str(e),
                "result": "failed"
            })
            
            return {
                "status": "failed",
                "bot_name": bot_name,
                "installation_id": installation_id,
                "error": error_msg,
                "setup_completed_at": datetime.now().isoformat()
            }
    
    def _create_activation_script(self, bot_dir: Path):
        """Create platform-specific activation script for the bot."""
        venv_paths = self._get_venv_paths(bot_dir)
        
        if self.system_info["platform"] == "Windows":
            # Windows batch script
            script_content = f"""@echo off
echo Activating environment for {bot_dir.name}...
call "{venv_paths['activate']}"
echo Environment activated! Python path: {venv_paths['python']}
echo.
echo To run the bot:
echo python {bot_dir.name}.py
echo.
cmd /k
"""
            script_file = bot_dir / "activate_and_run.bat"
        else:
            # Unix shell script
            script_content = f"""#!/bin/bash
echo "Activating environment for {bot_dir.name}..."
source "{venv_paths['activate']}"
echo "Environment activated! Python path: {venv_paths['python']}"
echo ""
echo "To run the bot:"
echo "python {bot_dir.name}.py"
echo ""
exec "$SHELL"
"""
            script_file = bot_dir / "activate_and_run.sh"
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        if self.system_info["platform"] != "Windows":
            os.chmod(script_file, 0o755)
        
        self.logger.info(f"Created activation script: {script_file}")
    
    def setup_multiple_bots(self, bot_names: List[str] = None) -> List[Dict[str, Any]]:
        """
        Setup environments for multiple bots.
        
        Args:
            bot_names: List of bot directory names. If None, setup all bots.
            
        Returns:
            List of setup results
        """
        
        if bot_names is None:
            # Find all bot directories
            bot_dirs = [d for d in self.base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        else:
            bot_dirs = [self.base_dir / name for name in bot_names if (self.base_dir / name).exists()]
        
        if not bot_dirs:
            self.logger.warning("No bot directories found")
            return []
        
        print(f"🚀 Setting up environments for {len(bot_dirs)} bots...")
        
        results = []
        for i, bot_dir in enumerate(bot_dirs, 1):
            print(f"\n📦 Setting up bot {i}/{len(bot_dirs)}: {bot_dir.name}")
            
            try:
                result = self.setup_bot_environment(bot_dir)
                results.append(result)
                
                if result["status"] == "success":
                    print(f"✅ {bot_dir.name} setup completed successfully")
                else:
                    print(f"❌ {bot_dir.name} setup failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                error_result = {
                    "status": "failed",
                    "bot_name": bot_dir.name,
                    "error": str(e),
                    "setup_completed_at": datetime.now().isoformat()
                }
                results.append(error_result)
                print(f"❌ {bot_dir.name} setup failed: {e}")
        
        successful = len([r for r in results if r["status"] == "success"])
        failed = len(results) - successful
        
        print(f"\n🎉 Environment setup complete!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        return results
    
    def get_installation_status(self, installation_id: str = None) -> Dict[str, Any]:
        """Get status of installations."""
        if installation_id:
            return self.active_installations.get(installation_id, {"error": "Installation not found"})
        else:
            return {
                "active_installations": self.active_installations,
                "total_installations": len(self.active_installations)
            }
    
    def verify_bot_environment(self, bot_dir: Path) -> Dict[str, Any]:
        """
        Verify that a bot's environment is properly set up.
        
        Args:
            bot_dir: Bot directory to verify
            
        Returns:
            Dict with verification results
        """
        
        try:
            venv_paths = self._get_venv_paths(bot_dir)
            
            verification = {
                "bot_name": bot_dir.name,
                "venv_exists": venv_paths["venv_dir"].exists(),
                "python_exists": venv_paths["python"].exists(),
                "pip_exists": venv_paths["pip"].exists(),
                "requirements_exists": (bot_dir / "requirements.txt").exists(),
                "environment_info_exists": (bot_dir / "environment_info.json").exists(),
                "verification_time": datetime.now().isoformat()
            }
            
            # Test Python execution
            if verification["python_exists"]:
                try:
                    result = subprocess.run([
                        str(venv_paths["python"]), "--version"
                    ], capture_output=True, text=True, timeout=10)
                    
                    verification["python_working"] = result.returncode == 0
                    verification["python_version"] = result.stdout.strip() if result.returncode == 0 else None
                except:
                    verification["python_working"] = False
                    verification["python_version"] = None
            else:
                verification["python_working"] = False
                verification["python_version"] = None
            
            # Check installed packages
            if verification["python_working"]:
                try:
                    result = subprocess.run([
                        str(venv_paths["python"]), "-m", "pip", "list", "--format=json"
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        packages = json.loads(result.stdout)
                        verification["installed_packages"] = len(packages)
                        verification["package_list"] = [p["name"] for p in packages]
                    else:
                        verification["installed_packages"] = 0
                        verification["package_list"] = []
                except:
                    verification["installed_packages"] = 0
                    verification["package_list"] = []
            
            # Overall status
            verification["environment_ready"] = all([
                verification["venv_exists"],
                verification["python_working"],
                verification["installed_packages"] > 0
            ])
            
            return verification
            
        except Exception as e:
            return {
                "bot_name": bot_dir.name,
                "verification_error": str(e),
                "environment_ready": False,
                "verification_time": datetime.now().isoformat()
            }


def main():
    """Test function for the package installer."""
    
    print("📦 Package Installation System Test")
    print("=" * 60)
    
    # Initialize installer
    installer = PackageInstaller()
    
    # Check for bot directories
    bots_dir = Path("generated_bots")
    if not bots_dir.exists():
        print("❌ No generated_bots directory found")
        print("💡 Generate some bots first using the bot generator")
        return
    
    bot_dirs = [d for d in bots_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not bot_dirs:
        print("❌ No bot directories found")
        print("💡 Generate some bots first using the bot generator")
        return
    
    print(f"🤖 Found {len(bot_dirs)} bot directories:")
    for i, bot_dir in enumerate(bot_dirs, 1):
        print(f"  {i}. {bot_dir.name}")
    
    # Ask which bots to setup
    while True:
        choice = input(f"\nSetup environments for which bots? (1-{len(bot_dirs)}, 'all', or comma-separated): ").strip()
        
        try:
            if choice.lower() == 'all':
                selected_bots = [d.name for d in bot_dirs]
                break
            elif ',' in choice:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected_bots = [bot_dirs[i].name for i in indices if 0 <= i < len(bot_dirs)]
                break
            else:
                index = int(choice) - 1
                if 0 <= index < len(bot_dirs):
                    selected_bots = [bot_dirs[index].name]
                    break
        except (ValueError, IndexError):
            print("❌ Invalid selection. Please try again.")
    
    print(f"\n🚀 Setting up environments for: {', '.join(selected_bots)}")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("👋 Setup cancelled")
        return
    
    # Setup environments
    results = installer.setup_multiple_bots(selected_bots)
    
    # Show results
    print(f"\n📊 Setup Results:")
    for result in results:
        status = "✅" if result["status"] == "success" else "❌"
        print(f"{status} {result['bot_name']}: {result['status']}")
        if result["status"] == "failed":
            print(f"   Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"   Packages: {result.get('packages_count', 0)}")
    
    print(f"\n🎉 Environment setup complete!")
    print(f"💡 You can now run the bots from their respective directories")


if __name__ == "__main__":
    main()