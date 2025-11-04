"""
Enhanced Bot Code Generator for Workflow Observer AI

This module uses Claude API to generate complete Python automation scripts
with improved cross-platform compatibility and better library management.

Key Enhancements:
- Platform-specific library selection (Windows vs Mac/Linux)
- Python version compatibility checks  
- Better requirements.txt generation with version constraints
- Fallback libraries for cross-platform compatibility
"""

import anthropic
import os
import json
import re
import sys
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

load_dotenv()


class EnhancedBotCodeGenerator:
    """
    Enhanced bot code generator with cross-platform library support.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the enhanced bot code generator.
        
        Args:
            api_key (str, optional): Anthropic API key
        """
        
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        try:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print("✅ Enhanced Bot Code Generator initialized")
        except Exception as e:
            print(f"❌ Error initializing Claude client: {e}")
            raise
        
        self.generated_bots = {}
        self.total_generations = 0
        
        # Setup output directory
        self.bots_dir = Path("generated_bots")
        self.bots_dir.mkdir(exist_ok=True)
        
        # Cross-platform library mappings
        self.cross_platform_libraries = {
            # Web automation - cross-platform
            "selenium": {
                "package": "selenium>=4.15.0",
                "platforms": ["Windows", "Darwin", "Linux"],
                "python_versions": ["3.8", "3.9", "3.10", "3.11", "3.12"],
                "description": "Web browser automation"
            },
            "requests": {
                "package": "requests>=2.31.0", 
                "platforms": ["Windows", "Darwin", "Linux"],
                "python_versions": ["3.8", "3.9", "3.10", "3.11", "3.12"],
                "description": "HTTP library"
            },
            # GUI automation - platform specific
            "pyautogui": {
                "package": "pyautogui>=0.9.54",
                "platforms": ["Windows", "Darwin", "Linux"],
                "python_versions": ["3.8", "3.9", "3.10", "3.11", "3.12"],
                "description": "Cross-platform GUI automation"
            },
            # Data processing - cross-platform
            "pandas": {
                "package": "pandas>=2.1.0",
                "platforms": ["Windows", "Darwin", "Linux"],
                "python_versions": ["3.8", "3.9", "3.10", "3.11", "3.12"],
                "description": "Data manipulation and analysis"
            },
            "openpyxl": {
                "package": "openpyxl>=3.1.0",
                "platforms": ["Windows", "Darwin", "Linux"],
                "python_versions": ["3.8", "3.9", "3.10", "3.11", "3.12"],
                "description": "Excel file handling"
            },
            # Configuration
            "python-dotenv": {
                "package": "python-dotenv>=1.0.0",
                "platforms": ["Windows", "Darwin", "Linux"],
                "python_versions": ["3.8", "3.9", "3.10", "3.11", "3.12"],
                "description": "Environment variable management"
            },
            "colorama": {
                "package": "colorama>=0.4.6",
                "platforms": ["Windows", "Darwin", "Linux"],
                "python_versions": ["3.8", "3.9", "3.10", "3.11", "3.12"],
                "description": "Colored terminal output"
            }
        }
        
        print("🤖 Enhanced Bot Code Generator ready!")
    
    def get_compatible_libraries(self, pattern: Dict[str, Any]) -> List[str]:
        """
        Get list of compatible libraries based on pattern requirements and current platform.
        
        Args:
            pattern: Workflow pattern information
            
        Returns:
            List of compatible package specifications
        """
        
        current_platform = platform.system()
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Determine required libraries based on pattern
        required_libs = set()
        
        # Base libraries always needed
        required_libs.update([
            "python-dotenv", "requests", "colorama"
        ])
        
        # Application-specific libraries
        applications = pattern.get("applications_involved", [])
        
        for app in applications:
            app_lower = app.lower()
            
            # Web browsers
            if any(browser in app_lower for browser in ["chrome", "firefox", "safari", "edge", "browser"]):
                required_libs.add("selenium")
            
            # Desktop applications
            if any(desktop in app_lower for desktop in ["word", "excel", "powerpoint", "desktop", "application"]):
                required_libs.add("pyautogui")
            
            # Excel/Spreadsheet
            if any(excel in app_lower for excel in ["excel", "spreadsheet", "xlsx", "csv"]):
                required_libs.update(["openpyxl", "pandas"])
        
        # Filter libraries for compatibility
        compatible_packages = []
        
        for lib in required_libs:
            if lib in self.cross_platform_libraries:
                lib_info = self.cross_platform_libraries[lib]
                
                # Check platform compatibility
                if current_platform in lib_info["platforms"]:
                    # Check Python version compatibility
                    if current_python in lib_info["python_versions"]:
                        compatible_packages.append(lib_info["package"])
        
        return sorted(compatible_packages)
    
    def _create_enhanced_generation_prompt(self, pattern: Dict[str, Any]) -> str:
        """
        Create an enhanced prompt with better cross-platform considerations.
        """
        
        current_platform = platform.system()
        current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
        compatible_libs = self.get_compatible_libraries(pattern)
        
        prompt = f'''You are an expert Python automation developer. Generate a complete, production-ready, CROSS-PLATFORM Python automation script based on this detected workflow pattern.

DETECTED WORKFLOW PATTERN:
{json.dumps(pattern, indent=2)}

SYSTEM INFORMATION:
- Target Platform: {current_platform}
- Python Version: {current_python}
- Compatible Libraries Available: {', '.join(compatible_libs)}

CROSS-PLATFORM REQUIREMENTS:
1. Use only the compatible libraries listed above
2. Include proper error handling for platform-specific features
3. Use pathlib for file paths (cross-platform)
4. Handle platform differences in GUI automation
5. Use environment variables for sensitive data
6. Include platform detection where needed

Generate a complete Python automation script that includes:
1. Cross-platform compatibility
2. Proper error handling and logging
3. Configuration management via config.json
4. Platform-specific adaptations
5. Clear documentation and setup instructions

The script should be production-ready and handle the workflow: {pattern.get('pattern_name', 'automation workflow')}

Most importantly, keep things concise and simple. Make sure you accomplish the desired task and that is it.
'''
        
        return prompt
    
    def _create_cross_platform_requirements(self, pattern: Dict[str, Any]) -> str:
        """
        Create cross-platform requirements.txt with proper version constraints.
        
        Args:
            pattern: Workflow pattern information
            
        Returns:
            str: Requirements file content
        """
        
        compatible_libs = self.get_compatible_libraries(pattern)
        
        # Add header comment
        requirements_content = f"""# Cross-Platform Requirements for {pattern.get('pattern_name', 'Automation Bot')}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Platform: {platform.system()}
# Python: {sys.version_info.major}.{sys.version_info.minor}

# Core automation libraries
"""
        
        # Add each compatible library
        for lib in compatible_libs:
            requirements_content += f"{lib}\n"
        
        return requirements_content
    
    def _create_cross_platform_config(self, pattern: Dict[str, Any]) -> str:
        """
        Create a cross-platform configuration file.
        
        Args:
            pattern: Workflow pattern information
            
        Returns:
            str: JSON configuration template
        """
        
        current_platform = platform.system()
        
        config = {
            "bot_info": {
                "name": pattern.get("pattern_name", "Automation Bot"),
                "description": pattern.get("business_value", ""),
                "created": datetime.now().isoformat(),
                "target_platform": current_platform,
                "applications": pattern.get("applications_involved", [])
            },
            "automation_settings": {
                "retry_attempts": 3,
                "timeout_seconds": 30,
                "delay_between_actions": 0.5 if current_platform == "Windows" else 0.8,
                "headless_browser": False,
                "screenshot_on_error": True
            },
            "credentials": {
                "email": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipient": ""
                },
                "web_apps": {
                    "username": "",
                    "password": "",
                    "login_url": ""
                }
            },
            "file_paths": {
                "input_directory": "./input",
                "output_directory": "./output",
                "log_directory": "./logs",
                "backup_directory": "./backup"
            },
            "notifications": {
                "email_enabled": False,
                "desktop_enabled": True,
                "send_on_success": True,
                "send_on_failure": True
            }
        }
        
        return json.dumps(config, indent=2)
    
    def generate_bot(self, pattern: Dict[str, Any], bot_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a complete cross-platform automation bot.
        
        Args:
            pattern: Detected workflow pattern
            bot_name: Optional custom name for the bot
            
        Returns:
            Dict with generation results
        """
        
        if not bot_name:
            bot_name = re.sub(r'[^a-zA-Z0-9_]', '_', pattern.get("pattern_name", "automation_bot")).lower()
        
        print(f"🤖 Generating cross-platform bot: {bot_name}")
        print(f"📋 Pattern: {pattern.get('pattern_name', 'Unknown')}")
        print(f"🖥️ Target Platform: {platform.system()}")
        
        try:
            # Create enhanced generation prompt
            prompt = self._create_enhanced_generation_prompt(pattern)
            
            # Call Claude API
            print("🌐 Calling Claude API for enhanced bot generation...")
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            
            # Extract code components
            python_code, config_json, requirements = self._extract_code_from_response(response_text)
            
            # Create fallback files if extraction failed
            if not config_json:
                config_json = self._create_cross_platform_config(pattern)
            
            if not requirements:
                requirements = self._create_cross_platform_requirements(pattern)
            
            # Create bot directory
            bot_dir = self.bots_dir / bot_name
            bot_dir.mkdir(exist_ok=True)
            
            # Save files
            files_created = {}
            
            # Save Python script
            bot_file = bot_dir / f"{bot_name}.py"
            with open(bot_file, 'w') as f:
                f.write(python_code)
            files_created['bot_script'] = str(bot_file)
            
            # Save config file
            config_file = bot_dir / "config.json"
            with open(config_file, 'w') as f:
                f.write(config_json)
            files_created['config_file'] = str(config_file)
            
            # Save requirements
            req_file = bot_dir / "requirements.txt"
            with open(req_file, 'w') as f:
                f.write(requirements)
            files_created['requirements_file'] = str(req_file)
            
            # Save pattern info
            pattern_file = bot_dir / "pattern_info.json"
            with open(pattern_file, 'w') as f:
                json.dump(pattern, f, indent=2)
            files_created['pattern_file'] = str(pattern_file)
            
            # Save platform info
            platform_info = {
                "target_platform": platform.system(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "compatible_libraries": self.get_compatible_libraries(pattern),
                "generation_date": datetime.now().isoformat()
            }
            
            platform_file = bot_dir / "platform_info.json"
            with open(platform_file, 'w') as f:
                json.dump(platform_info, f, indent=2)
            files_created['platform_file'] = str(platform_file)
            
            # Create enhanced README
            readme_content = self._create_enhanced_readme(pattern, bot_name, platform_info)
            readme_file = bot_dir / "README.md"
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            files_created['readme_file'] = str(readme_file)
            
            # Store bot info
            bot_info = {
                "id": bot_name,
                "pattern_name": pattern.get("pattern_name", "Unknown"),
                "pattern_type": pattern.get("pattern_type", "unknown"),
                "automation_potential": pattern.get("automation_potential", 0),
                "time_savings": pattern.get("time_savings_estimate", "Unknown"),
                "applications": pattern.get("applications_involved", []),
                "created_at": datetime.now().isoformat(),
                "directory": str(bot_dir),
                "files": files_created,
                "status": "generated",
                "implementation_approach": pattern.get("implementation_approach", ""),
                "business_value": pattern.get("business_value", ""),
                "platform_info": platform_info,
                "cross_platform": True
            }
            
            self.generated_bots[bot_name] = bot_info
            self.total_generations += 1
            
            print(f"✅ Cross-platform bot generated successfully!")
            print(f"📁 Location: {bot_dir}")
            print(f"🔧 Files created: {len(files_created)}")
            print(f"🖥️ Platform optimized for: {platform.system()}")
            
            return bot_info
            
        except Exception as e:
            error_info = {
                "id": bot_name,
                "pattern_name": pattern.get("pattern_name", "Unknown"),
                "status": "failed",
                "error": str(e),
                "created_at": datetime.now().isoformat(),
                "platform": platform.system()
            }
            
            print(f"❌ Bot generation failed: {e}")
            return error_info
    
    def _extract_code_from_response(self, response: str) -> tuple[str, str, str]:
        """Extract Python code, config, and requirements from Claude's response."""
        
        # Extract Python code
        python_pattern = r'```python\n(.*?)\n```'
        python_matches = re.findall(python_pattern, response, re.DOTALL)
        python_code = python_matches[0] if python_matches else ""
        
        # Extract JSON config
        json_pattern = r'```json\n(.*?)\n```'
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        config_json = json_matches[0] if json_matches else ""
        
        # Extract requirements
        req_pattern = r'```(?:txt|requirements)\n(.*?)\n```'
        req_matches = re.findall(req_pattern, response, re.DOTALL)
        requirements = req_matches[0] if req_matches else ""
        
        # If no code blocks found, try to extract from the full response
        if not python_code:
            # Look for class definitions as a fallback
            class_pattern = r'(class \w+:.*)'
            class_matches = re.findall(class_pattern, response, re.DOTALL)
            if class_matches:
                python_code = class_matches[0]
        
        return python_code, config_json, requirements
    
    def _create_enhanced_readme(self, pattern: Dict[str, Any], bot_name: str, platform_info: Dict[str, Any]) -> str:
        """Create an enhanced README with cross-platform instructions."""
        
        readme_content = f"""# {pattern.get('pattern_name', 'Automation Bot')}

**🤖 Cross-Platform Automation Bot**  
**Generated by Workflow Observer AI**  
**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Target Platform:** {platform_info['target_platform']}  
**Python Version:** {platform_info['python_version']}

## 📋 Overview
{pattern.get('business_value', 'Automates detected workflow pattern')}

## 🔄 Workflow Steps
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(pattern.get('workflow_steps', [])))}

## 📱 Applications Involved
{', '.join(pattern.get('applications_involved', []))}

## ⏱️ Estimated Time Savings
{pattern.get('time_savings_estimate', 'Unknown')}

## 🚀 Setup Instructions

### 1. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Settings:
Edit `config.json` with your credentials and settings

### 3. Run the Bot:
```bash
python {bot_name}.py
```

## 🔍 Automation Approach

**Automation Potential:** {pattern.get('automation_potential', 'N/A')}/10  
**Implementation:** Cross-platform automation script

## 📝 Notes

- Test the bot thoroughly before production use
- Monitor the first few runs to ensure accuracy
- Keep credentials secure and use environment variables
- The bot adapts its behavior based on your platform
"""
        
        return readme_content
    
    def generate_multiple_bots(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate automation bots for multiple patterns with cross-platform support."""
        
        print(f"🤖 Generating {len(patterns)} cross-platform automation bots...")
        print(f"🖥️ Target Platform: {platform.system()}")
        
        results = []
        for i, pattern in enumerate(patterns, 1):
            print(f"\n🔄 Generating bot {i}/{len(patterns)}")
            
            try:
                result = self.generate_bot(pattern)
                results.append(result)
                
                # Brief pause between generations
                import time
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Failed to generate bot {i}: {e}")
                continue
        
        successful = len([r for r in results if r.get('status') == 'generated'])
        
        print(f"\n✅ Cross-platform bot generation complete!")
        print(f"📊 Successfully generated: {successful}/{len(patterns)} bots")
        print(f"🖥️ All bots optimized for: {platform.system()}")
        
        return results
    
    def get_generated_bots(self) -> Dict[str, Any]:
        """Get information about all generated bots."""
        return {
            "total_bots": len(self.generated_bots),
            "bots": self.generated_bots,
            "generation_stats": {
                "total_generations": self.total_generations,
                "successful": len([b for b in self.generated_bots.values() if b.get("status") == "generated"]),
                "failed": len([b for b in self.generated_bots.values() if b.get("status") == "failed"]),
                "target_platform": platform.system(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
        }
    
    def save_generation_summary(self, output_file: str = "bot_generation_summary.json"):
        """Save a summary of all generated bots."""
        summary = self.get_generated_bots()
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Generation summary saved to: {output_file}")


def main():
    """Test function for the enhanced bot generator."""
    
    print("🤖 Enhanced Cross-Platform Bot Generator Test")
    print("=" * 60)
    print(f"🖥️ Current Platform: {platform.system()}")
    print(f"🐍 Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Initialize enhanced generator
    try:
        generator = EnhancedBotCodeGenerator()
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        return
    
    # Load detected patterns if available
    patterns_file = "detected_patterns.json"
    if Path(patterns_file).exists():
        print(f"📋 Loading patterns from {patterns_file}")
        
        with open(patterns_file, 'r') as f:
            patterns = json.load(f)
        
        if patterns and len(patterns) > 0:
            print(f"🔍 Found {len(patterns)} detected patterns")
            
            # Show platform compatibility for each pattern
            print("\nPlatform Compatibility Analysis:")
            for i, pattern in enumerate(patterns, 1):
                name = pattern.get('pattern_name', f'Pattern {i}')
                compatible_libs = generator.get_compatible_libraries(pattern)
                print(f"  {i}. {name}")
                print(f"     Compatible libraries: {len(compatible_libs)}")
                print(f"     Automation potential: {pattern.get('automation_potential', 'N/A')}")
            
            # Ask which patterns to generate bots for
            choice = input(f"\nGenerate cross-platform bots for which patterns? (1-{len(patterns)}, 'all', or comma-separated): ").strip()
            
            if choice.lower() == 'all':
                selected_patterns = patterns
            else:
                try:
                    if ',' in choice:
                        indices = [int(x.strip()) - 1 for x in choice.split(',')]
                    else:
                        indices = [int(choice) - 1]
                    
                    selected_patterns = [patterns[i] for i in indices if 0 <= i < len(patterns)]
                except (ValueError, IndexError):
                    print("❌ Invalid selection")
                    return
            
            # Generate bots
            print(f"\n🚀 Generating cross-platform bots for {len(selected_patterns)} patterns...")
            results = generator.generate_multiple_bots(selected_patterns)
            
            # Save summary
            generator.save_generation_summary()
            
            # Show results summary
            successful = len([r for r in results if r.get('status') == 'generated'])
            failed = len(results) - successful
            
            print(f"\n🎉 Enhanced bot generation complete!")
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {failed}")
            print(f"🖥️ Platform: {platform.system()}")
            print(f"📁 Check the 'generated_bots/' directory for your automation scripts")
            
        else:
            print("❌ No patterns found in the file")
    else:
        print(f"❌ Patterns file not found: {patterns_file}")
        print("💡 Run pattern analysis first to detect workflow patterns")


if __name__ == "__main__":
    main()