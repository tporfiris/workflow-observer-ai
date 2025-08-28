# cloud_vlm_analyzer.py
"""
Optimized VLM Analyzer for Workflow Observer AI

This module provides a streamlined VLM analysis focused specifically on the data
needed for pattern recognition: application identification and detailed activity breakdown.

The output is optimized for detecting workflow patterns by focusing on:
1. Precise application/website identification
2. Detailed breakdown of user actions within that application
3. Structured data that enables sequence analysis

Key Changes from Previous Version:
- Simplified output focused on application + detailed activity
- More granular action detection (clicking, typing, selecting, etc.)
- Better data structure for pattern recognition
- Reduced unnecessary metadata

Sample output:
    "screenshot_20250725_165737.jpg": {
      "application": "Microsoft Excel for Mac",
      "detailed_activity": "Creating a new spreadsheet with a SUM formula in cell B2 that references cells A2:A5. Column A contains numbers 1-4. The user has a software update notification at the top of the window stating 'We've made some fixes and improvements. To complete the process, the app needs to restart.' The Excel interface shows the user is using the Aptos Narrow font and has various ribbon menus (Home, Insert, Draw, etc.) visible. The spreadsheet is named 'Book1' and appears to be a new or mostly empty worksheet.",
      "image_path": "screenshots/screenshot_20250725_165737.jpg",
      "analysis_timestamp": "2025-08-02T22:28:01.410771",
      "filename": "screenshot_20250725_165737.jpg"
    }
"""

import anthropic
import base64
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import os
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()


class OptimizedVLMAnalyzer:
    """
    Streamlined VLM analyzer focused on application identification and detailed activity analysis.
    
    This analyzer extracts exactly the data needed for pattern recognition:
    - What application/website is being used
    - What specific action the user is performing within that application
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the optimized VLM analyzer.
        
        Args:
            api_key (str, optional): Anthropic API key
        """
        
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        
        if not self.api_key:
            print("❌ Anthropic API key not found!")
            print("💡 Set environment variable: export ANTHROPIC_API_KEY='your-api-key'")
            raise ValueError("Anthropic API key is required")
        
        try:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print("✅ Optimized Claude Vision API client initialized")
        except Exception as e:
            print(f"❌ Error initializing Anthropic client: {e}")
            raise
        
        self.analysis_cache = {}
        self.total_api_calls = 0
        self.estimated_cost = 0.0
        
        self._setup_optimized_prompts()
        
        print("🎯 Optimized VLM Analyzer ready!")
        print("📊 Focus: Application identification + detailed activity analysis")
    
    def _setup_optimized_prompts(self):
        """
        Setup the optimized prompt template focused on application and activity detection.
        """
        
        self.analysis_prompt = """You are an expert at analyzing computer screenshots to identify applications and user activities for workflow automation.

Analyze this screenshot and provide ONLY the following information:

1. **APPLICATION**: What specific application or website is being used? Be as precise as possible.
   Examples: "Microsoft Excel", "QuickBooks Desktop", "Chrome - Gmail", "Outlook", "Salesforce", "SAP", "Chrome - company.bamboohr.com", etc.

2. **DETAILED_ACTIVITY**: Describe in detail what the user is doing within this application. Focus on the specific action/task.
   
   Be very specific about:
   - What type of screen/interface they're looking at
   - What data they're viewing, entering, or manipulating
   - What specific actions they're performing (clicking, typing, selecting, copying, etc.)
   - What fields, buttons, or UI elements they're interacting with

   Examples:
   - "Viewing an invoice PDF attachment in email, looking at vendor name and amount details"
   - "Entering vendor information into QuickBooks expense entry form, filling out vendor name field"
   - "Copying invoice total amount from PDF document"
   - "Clicking 'Save' button to submit expense entry in accounting software"
   - "Opening email attachment labeled 'Invoice_March2024.pdf'"
   - "Selecting vendor 'ABC Corp' from dropdown menu in expense form"
   - "Typing '1,250.00' into amount field of expense entry"

The goal is to capture enough detail that we can later identify when the same type of activity is repeated across multiple screenshots.

Respond ONLY with valid JSON in this exact format:
{
    "application": "specific application name",
    "detailed_activity": "very detailed description of what user is doing"
}"""
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64 for API transmission.
        """
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize for API efficiency
                max_size = 1568
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                image_bytes = buffer.getvalue()
                
                return base64.b64encode(image_bytes).decode('utf-8')
                
        except Exception as e:
            print(f"❌ Error encoding image {image_path}: {e}")
            raise
    
    def _make_api_call(self, image_base64: str, prompt: str) -> str:
        """
        Make an API call to Claude Vision.
        """
        try:
            print("🌐 Analyzing with Claude Vision...")
            start_time = time.time()
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,  # Reduced since we need less output
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            
            api_time = time.time() - start_time
            print(f"⚡ Response received in {api_time:.2f} seconds")
            
            self.total_api_calls += 1
            self.estimated_cost += 0.015
            
            return response.content[0].text
            
        except Exception as e:
            print(f"❌ API call failed: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """
        Parse JSON response from Claude with optimized error handling.
        """
        try:
            response = response.strip()
            
            if response.startswith('```json'):
                response = response.replace('```json', '').replace('```', '').strip()
            
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Validate required fields
                if 'application' not in parsed or 'detailed_activity' not in parsed:
                    raise ValueError("Missing required fields")
                
                return parsed
            else:
                return json.loads(response)
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  JSON parsing error: {e}")
            print(f"📝 Raw response: {response[:200]}...")
            
            # Create fallback response
            return {
                "application": "Parse Error",
                "detailed_activity": "Failed to parse Claude's response",
                "parsing_error": True,
                "raw_response": response
            }
    
    def analyze_screenshot(self, image_path: str, use_cache: bool = True) -> Dict:
        """
        Analyze a single screenshot with optimized output format.
        
        Args:
            image_path (str): Path to the screenshot file
            use_cache (bool): Whether to use cached results
            
        Returns:
            dict: Optimized analysis results
        """
        
        image_path = str(image_path)
        
        if use_cache and image_path in self.analysis_cache:
            print(f"📋 Using cached result for {Path(image_path).name}")
            return self.analysis_cache[image_path]
        
        print(f"🎯 Analyzing screenshot: {Path(image_path).name}")
        
        try:
            # Encode and analyze
            image_base64 = self._encode_image(image_path)
            response = self._make_api_call(image_base64, self.analysis_prompt)
            analysis = self._parse_json_response(response)
            
            if analysis:
                # Add minimal metadata (only what's needed)
                analysis.update({
                    "image_path": image_path,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "filename": Path(image_path).name
                })
                
                # Cache the result
                self.analysis_cache[image_path] = analysis
                
                print(f"✅ Analysis complete:")
                print(f"   📱 Application: {analysis.get('application', 'Unknown')}")
                print(f"   🔍 Activity: {analysis.get('detailed_activity', 'Unknown')[:80]}...")
                
                return analysis
            else:
                raise Exception("Failed to parse Claude's response")
                
        except Exception as e:
            print(f"❌ Error analyzing {image_path}: {e}")
            
            return {
                "image_path": image_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "filename": Path(image_path).name,
                "error": str(e),
                "application": "Analysis Failed",
                "detailed_activity": f"Error during analysis: {e}"
            }
    
    def analyze_batch(self, image_paths: List[str], max_images: int = None, delay_between_calls: float = 1.0) -> List[Dict]:
        """
        Analyze multiple screenshots in batch.
        """
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"📊 Starting optimized batch analysis of {len(image_paths)} screenshots")
        print(f"💰 Estimated cost: ${len(image_paths) * 0.015:.2f}")
        print("=" * 60)
        
        results = []
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n🔍 Analyzing {i}/{len(image_paths)}: {Path(image_path).name}")
            
            try:
                result = self.analyze_screenshot(image_path)
                results.append(result)
                
                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(image_paths) - i) * (avg_time + delay_between_calls)
                
                print(f"⏱️  Progress: {i}/{len(image_paths)} ({i/len(image_paths)*100:.1f}%)")
                print(f"⏱️  Est. remaining: {remaining/60:.1f} minutes")
                
                # Rate limiting
                if i < len(image_paths):
                    time.sleep(delay_between_calls)
                
            except Exception as e:
                print(f"❌ Failed to analyze {image_path}: {e}")
                continue
        
        total_time = time.time() - start_time
        print(f"\n✅ Optimized batch analysis complete!")
        print(f"📊 Processed: {len(results)} screenshots")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"💰 Total cost: ${self.estimated_cost:.2f}")
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str = "optimized_vlm_analysis.json"):
        """
        Save optimized analysis results.
        """
        try:
            # Create optimized cache format
            cache_data = {
                'analysis_cache': {result['filename']: result for result in results if 'filename' in result},
                'total_analyses': len(results),
                'last_updated': datetime.now().isoformat(),
                'analyzer_version': 'optimized_v1.0',
                'focus': 'application_identification_and_detailed_activity'
            }
            
            with open(output_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            print(f"💾 Optimized results saved to: {output_file}")
            
            # Print sample of what was captured
            print(f"\n📋 Sample analysis results:")
            for i, result in enumerate(results[:3]):
                if 'error' not in result:
                    print(f"   {i+1}. {result['application']}")
                    print(f"      Activity: {result['detailed_activity'][:60]}...")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """
    Main function to test the optimized VLM analyzer.
    """
    
    print("🎯 Optimized VLM Analyzer Test")
    print("=" * 60)
    print("🔍 Focus: Precise application + detailed activity detection")
    print()
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Initialize analyzer
    try:
        analyzer = OptimizedVLMAnalyzer()
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return
    
    # Find screenshots
    screenshots_dir = Path("screenshots")
    if not screenshots_dir.exists():
        print(f"❌ Screenshots directory not found: {screenshots_dir}")
        return
    
    screenshot_files = list(screenshots_dir.glob("screenshot_*.jpg"))
    if not screenshot_files:
        print("❌ No screenshot files found")
        return
    
    print(f"📷 Found {len(screenshot_files)} screenshots")
    
    # Ask how many to analyze
    while True:
        try:
            max_images_input = input(f"\nHow many screenshots to analyze? (1-{len(screenshot_files)}, or 'all'): ").strip()
            if max_images_input.lower() == 'all':
                max_images = None
                estimated_cost = len(screenshot_files) * 0.015
                break
            else:
                max_images = int(max_images_input)
                if 1 <= max_images <= len(screenshot_files):
                    estimated_cost = max_images * 0.015
                    break
        except ValueError:
            print("Please enter a valid number or 'all'")
    
    print(f"💰 Estimated cost: ${estimated_cost:.2f}")
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        return
    
    # Analyze screenshots
    screenshot_files.sort()
    results = analyzer.analyze_batch(
        [str(f) for f in screenshot_files],
        max_images=max_images,
        delay_between_calls=1.0
    )
    
    # Save results
    analyzer.save_results(results)
    
    print(f"\n🎯 Optimized analysis complete!")
    print(f"✅ Generated precise application + activity data")
    print(f"🔄 Ready for pattern recognition analysis")
    print(f"💡 Next: Run pattern_analyzer.py to detect workflow sequences")


if __name__ == "__main__":
    main()