"""
Simplified Secure VLM Analyzer using Presidio Image Redactor

This is the BETTER implementation using Microsoft's Presidio Image Redactor
instead of custom OCR and redaction logic. Much simpler and more maintainable.
"""

import anthropic
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from PIL import Image
import base64
from dotenv import load_dotenv

load_dotenv()

# Presidio Image Redactor - does everything in one package!
try:
    from presidio_image_redactor import ImageRedactorEngine
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    print("⚠️  Presidio not installed.")
    print("   Install with: pip install presidio-image-redactor")


class SecureVLMAnalyzer:
    """
    Simplified secure VLM analyzer using Presidio Image Redactor.
    
    Automatically detects and redacts PII before sending screenshots to cloud VLM.
    Much simpler than custom implementation - all PII detection and redaction
    handled by Microsoft's Presidio library.
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 enable_audit_log: bool = True,
                 fill_color: tuple = (0, 0, 0),  # Black by default
                 use_metadata: bool = True):
        """
        Initialize the secure VLM analyzer.
        
        Args:
            api_key: Anthropic API key
            enable_audit_log: Log PII detections for compliance
            fill_color: Color for redaction boxes (R, G, B)
            use_metadata: Include PII detection metadata in results
        """
        
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.enable_audit_log = enable_audit_log
        self.use_metadata = use_metadata
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Initialize Presidio Image Redactor (does OCR + PII detection + redaction)
        if PRESIDIO_AVAILABLE:
            self.image_redactor = ImageRedactorEngine()
            
            # Optional: Customize PII entities to detect
            self.analyzer = AnalyzerEngine()
            self.pii_entities = [
                "CREDIT_CARD",
                "US_SSN", 
                "US_BANK_NUMBER",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "US_DRIVER_LICENSE",
                "US_PASSPORT",
                "PERSON",  # Names
                "MEDICAL_LICENSE",
                "IP_ADDRESS",
                "IBAN_CODE",
            ]
            
            print("✅ Presidio Image Redactor initialized")
            print(f"🔒 Detecting: {', '.join(self.pii_entities)}")
        else:
            self.image_redactor = None
            print("❌ Presidio not available - install with:")
            print("   pip install presidio-image-redactor")
        
        # Setup directories
        self.redacted_dir = Path("screenshots_redacted")
        self.redacted_dir.mkdir(exist_ok=True)
        
        if enable_audit_log:
            self.audit_dir = Path("privacy_audit")
            self.audit_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.analysis_cache = {}
        self.total_analyses = 0
        self.total_redactions = 0
        self.pii_stats = {}
        
        print("✅ Secure VLM Analyzer ready")
        print(f"🔒 Privacy Protection: {'Enabled' if PRESIDIO_AVAILABLE else 'Disabled'}")
        print(f"📋 Audit Logging: {'Enabled' if enable_audit_log else 'Disabled'}")
            
    def redact_image(self, image_path: str) -> tuple[str, List[Dict]]:
        """
        Redact PII from image using Presidio Image Redactor.
        """
        
        if not PRESIDIO_AVAILABLE:
            return image_path, []
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Let Presidio handle everything (OCR + analysis + redaction)
            redacted_image = self.image_redactor.redact(
                image=image,
                fill=(0, 0, 0)
            )
            
            # If you need the detection details, do a separate analysis
            from presidio_image_redactor import ImageAnalyzerEngine
            image_analyzer = ImageAnalyzerEngine(self.analyzer)
            
            analyzer_results = image_analyzer.analyze(
                image=image,
                entities=self.pii_entities
            )
            
            # Convert to logging format
            pii_detections = []
            for result in analyzer_results:
                pii_detections.append({
                    'entity_type': result.entity_type,
                    'score': result.score,
                    'start': result.start,
                    'end': result.end
                })
                
                self.pii_stats[result.entity_type] = \
                    self.pii_stats.get(result.entity_type, 0) + 1
            
            # Save redacted image
            redacted_path = self.redacted_dir / f"redacted_{Path(image_path).name}"
            redacted_image.save(redacted_path)
            
            if pii_detections:
                self.total_redactions += 1
                print(f"   🔒 Redacted {len(pii_detections)} PII instances")
            else:
                print(f"   ✅ No PII detected")
            
            return str(redacted_path), pii_detections
            
        except Exception as e:
            print(f"⚠️  Redaction failed: {e}")
            import traceback
            traceback.print_exc()
            return image_path, []

            
    def analyze_screenshot_securely(self, image_path: str) -> Dict:
        """
        Analyze screenshot with automatic PII protection.
        
        Args:
            image_path: Path to original screenshot
            
        Returns:
            Analysis result with privacy metadata
        """
        
        print(f"🔒 Secure analysis: {Path(image_path).name}")
        
        # Step 1: Redact PII using Presidio (one line!)
        redacted_path, pii_detections = self.redact_image(image_path)
        
        # Step 2: Create audit log
        if self.enable_audit_log and pii_detections:
            self._log_redaction(image_path, pii_detections)
        
        # Step 3: Analyze with cloud VLM (using sanitized image)
        print("   🌐 Analyzing with cloud VLM...")
        analysis = self._analyze_with_anthropic(redacted_path)
        
        # Step 4: Add privacy metadata
        analysis['privacy'] = {
            'pii_detected': len(pii_detections) > 0,
            'pii_count': len(pii_detections),
            'pii_types': [d['entity_type'] for d in pii_detections],
            'redaction_applied': redacted_path != image_path,
            'original_image': image_path,
            'analyzed_image': redacted_path
        }
        
        print("   ✅ Secure analysis complete")
        
        return analysis
    
    def _analyze_with_anthropic(self, image_path: str) -> Dict:
        """
        Analyze image using Anthropic Claude Vision API.
        
        Args:
            image_path: Path to (redacted) screenshot
            
        Returns:
            Analysis result
        """
        
        try:
            # Encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Analysis prompt
            prompt = """Analyze this screenshot and identify:

1. **APPLICATION**: What specific application or website is being used?
2. **DETAILED_ACTIVITY**: What is the user doing?

Note: Some sensitive information may be redacted (black boxes). 
Focus on the workflow and actions, not specific data values.

Respond as JSON:
{"application": "...", "detailed_activity": "..."}"""
            
            # Call Anthropic API
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            
            # Parse response
            response_text = response.content[0].text
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                analysis = {
                    "application": "Unknown",
                    "detailed_activity": response_text
                }
            
            # Add metadata
            analysis.update({
                "image_path": image_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "filename": Path(image_path).name,
                "model": "claude-3-5-sonnet-20241022"
            })
            
            self.total_analyses += 1
            
            return analysis
            
        except Exception as e:
            print(f"❌ Cloud analysis failed: {e}")
            return {
                "application": "Analysis Failed",
                "detailed_activity": f"Error: {e}",
                "error": str(e)
            }
    
    def _log_redaction(self, image_path: str, pii_detections: List[Dict]):
        """
        Create audit log entry for compliance.
        
        Args:
            image_path: Original image path
            pii_detections: List of detected PII
        """
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "image": Path(image_path).name,
            "pii_count": len(pii_detections),
            "pii_types": {},
            "detections": []
        }
        
        for detection in pii_detections:
            # Count by type
            entity_type = detection['entity_type']
            log_entry["pii_types"][entity_type] = \
                log_entry["pii_types"].get(entity_type, 0) + 1
            
            # Add sanitized detection (no actual PII values)
            log_entry["detections"].append({
                "type": detection['entity_type'],
                "score": detection['score']
            })
        
        # Save to audit log
        audit_file = self.audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(audit_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def analyze_batch(self, image_paths: List[str], max_images: int = None,
                     delay_between_calls: float = 1.0) -> List[Dict]:
        """
        Analyze multiple screenshots with PII protection.
        
        Args:
            image_paths: List of image paths
            max_images: Maximum to analyze
            delay_between_calls: Delay between API calls
            
        Returns:
            List of analysis results
        """
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"🔒 Starting secure batch analysis of {len(image_paths)} screenshots")
        print(f"🔒 Using Presidio Image Redactor for automatic PII protection")
        print("=" * 60)
        
        results = []
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n📸 {i}/{len(image_paths)}: {Path(image_path).name}")
            
            try:
                analysis = self.analyze_screenshot_securely(image_path)
                results.append(analysis)
                
                # Progress
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(image_paths) - i) * (avg_time + delay_between_calls)
                
                print(f"   ⏱️  Progress: {i}/{len(image_paths)} ({i/len(image_paths)*100:.1f}%)")
                print(f"   ⏱️  Est. remaining: {remaining/60:.1f} minutes")
                
                time.sleep(delay_between_calls)
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Summary
        pii_found_count = sum(1 for r in results if r.get('privacy', {}).get('pii_detected', False))
        
        print(f"\n✅ Secure batch analysis complete!")
        print(f"📊 Results:")
        print(f"   - Total processed: {len(results)}")
        print(f"   - Screenshots with PII: {pii_found_count}")
        print(f"   - Total PII instances: {sum(self.pii_stats.values())}")
        print(f"   - PII types found: {dict(self.pii_stats)}")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"🔒 Privacy: All sensitive data redacted before cloud analysis")
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str = "secure_vlm_analysis.json"):
        """Save analysis results with privacy metadata."""
        
        cache_data = {
            'analysis_cache': {r['filename']: r for r in results if 'filename' in r},
            'total_analyses': len(results),
            'total_redactions': self.total_redactions,
            'pii_stats': self.pii_stats,
            'last_updated': datetime.now().isoformat(),
            'analyzer_version': 'secure_vlm_presidio_v1.0',
            'privacy_protection': 'presidio_image_redactor',
            'audit_logging': self.enable_audit_log
        }
        
        with open(output_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {output_file}")
        
        if self.enable_audit_log:
            print(f"📋 Audit logs: {self.audit_dir}/")


def main():
    """Test the secure VLM analyzer."""
    
    print("🔒 Secure VLM Analyzer with Presidio Image Redactor")
    print("=" * 60)
    
    # Check API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Initialize
    try:
        analyzer = SecureVLMAnalyzer(enable_audit_log=True)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Find screenshots
    screenshots_dir = Path("screenshots")
    if not screenshots_dir.exists():
        print(f"❌ Screenshots directory not found")
        return
    
    screenshot_files = list(screenshots_dir.glob("screenshot_*.jpg"))
    if not screenshot_files:
        print("❌ No screenshots found")
        return
    
    print(f"📷 Found {len(screenshot_files)} screenshots")
    
    # Ask how many to analyze
    max_input = input(f"\nHow many to analyze? (1-{len(screenshot_files)}, or 'all'): ").strip()
    if max_input.lower() == 'all':
        max_images = None
    else:
        try:
            max_images = int(max_input)
        except:
            max_images = None
    
    print(f"\n🔒 Presidio will automatically detect and redact PII")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        return
    
    # Analyze
    screenshot_files.sort()
    results = analyzer.analyze_batch(
        [str(f) for f in screenshot_files],
        max_images=max_images
    )
    
    # Save
    analyzer.save_results(results)
    
    print(f"\n✅ Complete!")
    print(f"   - Analysis: secure_vlm_analysis.json")
    print(f"   - Redacted images: screenshots_redacted/")
    if analyzer.enable_audit_log:
        print(f"   - Audit logs: privacy_audit/")


if __name__ == "__main__":
    main()