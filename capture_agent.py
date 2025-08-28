# capture_agent.py
"""
Screen Capture Agent for Workflow Observer AI

This agent continuously captures screenshots at regular intervals to build
a timeline of user workflows. It's designed to be lightweight, configurable,
and respectful of system resources.

Key Features:
- Configurable capture intervals
- Automatic file management with rotation
- Performance monitoring
- Graceful shutdown handling
- Detailed logging of operations
"""

import mss
import os
import time
import signal
import threading
from datetime import datetime, timedelta
from PIL import Image
from pathlib import Path
import psutil 


class ScreenCaptureAgent:
    """
    Main class that handles continuous screen capture with proper resource management.
    
    This agent runs in a separate thread to avoid blocking the main program,
    and includes safety mechanisms to prevent excessive disk usage.
    """
    
    def __init__(self, 
                 capture_interval=3,           # Seconds between captures
                 max_files=1000,                # Maximum screenshots to keep
                 output_dir="screenshots",      # Directory to save screenshots
                 image_quality=85):             # JPEG quality (1-100)
        """
        Initialize the capture agent with configuration parameters.
        
        Args:
            capture_interval (int): Seconds to wait between screenshots
            max_files (int): Maximum number of screenshot files to retain
            output_dir (str): Directory where screenshots will be saved
            image_quality (int): JPEG compression quality (85 is good balance)
        """
        
        # Store configuration parameters
        self.capture_interval = capture_interval
        self.max_files = max_files
        self.output_dir = Path(output_dir)
        self.image_quality = image_quality
        
        # Runtime state variables
        self.is_running = False              # Controls the capture loop
        self.capture_thread = None           # Thread that runs the capture loop
        self.total_captures = 0              # Count of screenshots taken
        self.start_time = None               # When the agent started
        self.last_capture_time = None        # Timestamp of last successful capture
        
        # Performance monitoring
        self.capture_times = []              # List to track how long each capture takes
        self.disk_usage_mb = 0               # Track total disk space used
        
        # Initialize the screenshot tool
        self.sct = None                      # Will hold the mss screenshot object
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler)  # Termination signal
        
        print(f"🤖 Screen Capture Agent initialized")
        print(f"   📷 Capture interval: {capture_interval} seconds")
        print(f"   📁 Output directory: {output_dir}")
        print(f"   🗃️  Max files to keep: {max_files}")
        print(f"   🎨 Image quality: {image_quality}%")
    
    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals (Ctrl+C, etc.) gracefully.
        
        This ensures we don't leave the system in a bad state if the user
        interrupts the program.
        """
        print(f"\n⚠️  Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def _setup_output_directory(self):
        """
        Create the output directory if it doesn't exist.
        
        This also handles any permission issues that might arise.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Output directory ready: {self.output_dir.absolute()}")
            return True
        except PermissionError:
            print(f"❌ Permission denied creating directory: {self.output_dir}")
            return False
        except Exception as e:
            print(f"❌ Error creating directory: {e}")
            return False
    
    def _initialize_screenshot_tool(self):
        """
        Initialize the mss (multi-screen screenshot) tool.
        
        This needs to be done in the same thread that will use it,
        which is why it's separate from __init__.
        """
        try:
            self.sct = mss.mss()
            
            # Get monitor information for logging
            monitors = self.sct.monitors
            print(f"🖥️  Detected {len(monitors)} monitors:")
            for i, monitor in enumerate(monitors):
                if i == 0:
                    print(f"     Monitor {i}: All screens combined")
                else:
                    print(f"     Monitor {i}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")
            
            # We'll capture the primary monitor (index 1, since 0 is all combined)
            self.primary_monitor = monitors[1]
            print(f"📷 Will capture primary monitor: {self.primary_monitor['width']}x{self.primary_monitor['height']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize screenshot tool: {e}")
            return False
    
    def _take_screenshot(self):
        """
        Capture a single screenshot and save it to disk.
        
        Returns:
            bool: True if successful, False if there was an error
            str: Filename of saved screenshot (or None if failed)
        """
        
        # Record start time for performance monitoring
        capture_start = time.time()
        
        try:
            # Take the actual screenshot
            screenshot = self.sct.grab(self.primary_monitor)
            
            # Convert to PIL Image for easier handling and compression
            # mss returns images in BGRA format, so we need to convert
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # Generate filename with timestamp
            # Format: screenshot_YYYYMMDD_HHMMSS.jpg
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            filepath = self.output_dir / filename
            
            # Save as JPEG with specified quality to balance size vs. quality
            img.save(filepath, "JPEG", quality=self.image_quality, optimize=True)
            
            # Calculate how long this capture took
            capture_duration = time.time() - capture_start
            self.capture_times.append(capture_duration)
            
            # Keep only last 100 capture times for rolling average
            if len(self.capture_times) > 100:
                self.capture_times.pop(0)
            
            # Update statistics
            self.total_captures += 1
            self.last_capture_time = datetime.now()
            
            # Calculate file size for disk usage tracking
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            self.disk_usage_mb += file_size_mb
            
            # Log progress every 10 captures to avoid spam
            if self.total_captures % 10 == 0:
                avg_capture_time = sum(self.capture_times) / len(self.capture_times)
                print(f"📊 Captured {self.total_captures} screenshots, "
                      f"avg time: {avg_capture_time:.3f}s, "
                      f"disk usage: {self.disk_usage_mb:.1f}MB")
            
            return True, filename
            
        except Exception as e:
            print(f"❌ Failed to capture screenshot: {e}")
            return False, None
    
    def _cleanup_old_files(self):
        """
        Remove old screenshot files to prevent unlimited disk usage.
        
        This keeps only the most recent max_files screenshots,
        deleting older ones based on filename timestamp.
        """
        try:
            # Get all screenshot files in the directory
            screenshot_files = list(self.output_dir.glob("screenshot_*.jpg"))
            
            # If we haven't exceeded the limit, no cleanup needed
            if len(screenshot_files) <= self.max_files:
                return
            
            # Sort by filename (which includes timestamp) to get chronological order
            screenshot_files.sort()
            
            # Calculate how many files to delete
            files_to_delete = len(screenshot_files) - self.max_files
            old_files = screenshot_files[:files_to_delete]
            
            # Delete the oldest files
            deleted_size_mb = 0
            for file_path in old_files:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                file_path.unlink()  # Delete the file
                deleted_size_mb += file_size_mb
            
            # Update disk usage tracking
            self.disk_usage_mb -= deleted_size_mb
            
            print(f"🧹 Cleaned up {len(old_files)} old screenshots, "
                  f"freed {deleted_size_mb:.1f}MB")
            
        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
    
    def _capture_loop(self):
        """
        Main loop that runs continuously to capture screenshots.
        
        This runs in a separate thread and handles:
        - Taking screenshots at regular intervals
        - Managing file cleanup
        - Error recovery
        - Performance monitoring
        """
        
        print(f"🚀 Starting capture loop (every {self.capture_interval} seconds)")
        
        # Initialize screenshot tool in this thread
        if not self._initialize_screenshot_tool():
            print("❌ Failed to initialize screenshot tool, stopping")
            self.is_running = False
            return
        
        consecutive_failures = 0
        max_consecutive_failures = 5  # Stop if we fail 5 times in a row
        
        while self.is_running:
            try:
                # Take a screenshot
                success, filename = self._take_screenshot()
                
                if success:
                    consecutive_failures = 0  # Reset failure counter
                    
                    # Periodically clean up old files (every 50 captures)
                    if self.total_captures % 50 == 0:
                        self._cleanup_old_files()
                        
                else:
                    consecutive_failures += 1
                    print(f"⚠️  Screenshot failed ({consecutive_failures}/{max_consecutive_failures})")
                    
                    # If we've failed too many times, stop the agent
                    if consecutive_failures >= max_consecutive_failures:
                        print("❌ Too many consecutive failures, stopping capture")
                        self.is_running = False
                        break
                
                # Wait for the next capture interval
                # We use a loop with short sleeps so we can respond quickly to shutdown
                sleep_time = 0
                while sleep_time < self.capture_interval and self.is_running:
                    time.sleep(0.1)  # Sleep in 100ms increments
                    sleep_time += 0.1
                
            except Exception as e:
                print(f"❌ Unexpected error in capture loop: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                time.sleep(1)  # Brief pause before retrying
        
        # Cleanup when loop ends
        if self.sct:
            self.sct.close()
        
        print("🛑 Capture loop ended")
    
    def start(self):
        """
        Start the screen capture agent.
        
        This creates the output directory and starts the capture loop
        in a background thread.
        """
        
        if self.is_running:
            print("⚠️  Agent is already running")
            return False
        
        # Setup output directory
        if not self._setup_output_directory():
            return False
        
        # Start the capture process
        self.is_running = True
        self.start_time = datetime.now()
        
        # Create and start the capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        print(f"✅ Screen capture agent started at {self.start_time.strftime('%H:%M:%S')}")
        return True
    
    def stop(self):
        """
        Stop the screen capture agent gracefully.
        
        This signals the capture loop to stop and waits for it to finish.
        """
        
        if not self.is_running:
            print("⚠️  Agent is not running")
            return
        
        print("🛑 Stopping screen capture agent...")
        self.is_running = False
        
        # Wait for the capture thread to finish (with timeout)
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
            
            if self.capture_thread.is_alive():
                print("⚠️  Capture thread didn't stop cleanly")
            else:
                print("✅ Capture thread stopped cleanly")
        
        # Print final statistics
        if self.start_time:
            runtime = datetime.now() - self.start_time
            print(f"📊 Final statistics:")
            print(f"   ⏱️  Runtime: {runtime}")
            print(f"   📷 Total captures: {self.total_captures}")
            print(f"   💾 Disk usage: {self.disk_usage_mb:.1f}MB")
            if self.capture_times:
                avg_time = sum(self.capture_times) / len(self.capture_times)
                print(f"   ⚡ Avg capture time: {avg_time:.3f}s")
    
    def get_status(self):
        """
        Get current status information about the agent.
        
        Returns:
            dict: Status information including runtime, capture count, etc.
        """
        
        status = {
            'is_running': self.is_running,
            'total_captures': self.total_captures,
            'disk_usage_mb': round(self.disk_usage_mb, 2),
            'last_capture': self.last_capture_time.isoformat() if self.last_capture_time else None,
        }
        
        if self.start_time:
            runtime = datetime.now() - self.start_time
            status['runtime_seconds'] = int(runtime.total_seconds())
            status['runtime_formatted'] = str(runtime).split('.')[0]  # Remove microseconds
        
        if self.capture_times:
            status['avg_capture_time'] = round(sum(self.capture_times) / len(self.capture_times), 3)
        
        return status


def main():
    """
    Main function to run the capture agent as a standalone program.
    
    This provides a simple command-line interface for testing the agent.
    """
    
    print("🚀 Workflow Observer AI - Screen Capture Agent")
    print("=" * 60)
    
    # Create and configure the agent
    # You can modify these parameters to test different configurations
    agent = ScreenCaptureAgent(
        capture_interval=2,        # Capture every 10 seconds as requested
        max_files=500,              # Keep up to 500 screenshots (about 1.4 hours at 10s intervals)
        output_dir="screenshots",   # Save to ./screenshots directory
        image_quality=85            # Good balance of quality vs file size
    )
    
    # Start the agent
    if not agent.start():
        print("❌ Failed to start the agent")
        return
    
    try:
        print("\n📝 Agent is running. Commands:")
        print("   - Press Enter to see status")
        print("   - Type 'quit' or 'exit' to stop")
        print("   - Type 'status' for detailed status")
        print("   - Ctrl+C to force quit")
        print()
        
        # Simple command loop for interaction
        while agent.is_running:
            try:
                user_input = input(">>> ").strip().lower()
                
                if user_input in ['quit', 'exit', 'q']:
                    break
                elif user_input in ['status', 's', '']:
                    status = agent.get_status()
                    print(f"\n📊 Agent Status:")
                    print(f"   🟢 Running: {status['is_running']}")
                    print(f"   📷 Captures: {status['total_captures']}")
                    print(f"   💾 Disk usage: {status['disk_usage_mb']}MB")
                    if 'runtime_formatted' in status:
                        print(f"   ⏱️  Runtime: {status['runtime_formatted']}")
                    if 'avg_capture_time' in status:
                        print(f"   ⚡ Avg capture time: {status['avg_capture_time']}s")
                    print()
                else:
                    print(f"❓ Unknown command: {user_input}")
                    
            except EOFError:
                # Handle Ctrl+D
                break
                
    except KeyboardInterrupt:
        # Handle Ctrl+C
        print("\n")
    
    finally:
        # Always stop the agent cleanly
        agent.stop()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()