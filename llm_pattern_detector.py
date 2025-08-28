# llm_pattern_detector.py
"""
Simple workflow analyzer using Claude API
Loads JSON data and sends it to Claude for analysis

Sample output:

[
  {
    "pattern_name": "Weekly Timesheet Review and Approval",
    "pattern_type": "time_tracking",
    "frequency": "multiple times per week",
    "applications_involved": ["Chrome - Workday", "Microsoft Outlook"],
    "workflow_steps": [
      "Step 1: Access Workday timesheet approval queue",
      "Step 2: Review individual timesheet entries",
      "Step 3: Verify special cases (overtime, vacation, sick leave)",
      "Step 4: Approve timesheet",
      "Step 5: Move to next employee"
    ],
    "automation_potential": 8,
    "automation_reasoning": "Highly structured process with clear decision points and digital data",
    "business_value": "Time savings for HR staff, faster approvals, reduced errors",
    "implementation_approach": "RPA bot with business rules for standard cases, exception routing for special cases",
    "time_savings_estimate": "2-3 hours per week",
    "sample_occurrences": [
      {
        "start_time": "2024-12-17T09:11:30",
        "end_time": "2024-12-17T09:14:30",
        "duration_minutes": 3,
        "activities": ["Review and approve 3 employee timesheets"]
      }
    ]
  },
  {
    "pattern_name": "New Employee Benefits Enrollment",
    "pattern_type": "onboarding",
    "frequency": "weekly",
    "applications_involved": ["Microsoft Outlook", "Chrome - Benefits Portal", "Chrome - BambooHR"],
    "workflow_steps": [
      "Step 1: Receive benefits enrollment email",
      "Step 2: Access benefits portal",
      "Step 3: Enter employee health/dental selections",
      "Step 4: Calculate deductions",
      "Step 5: Submit enrollment",
      "Step 6: Send confirmation email"
    ],
    "automation_potential": 7,
    "automation_reasoning": "Structured data entry with clear workflow, but may require some human verification",
    "business_value": "Reduced processing time, fewer data entry errors",
    "implementation_approach": "Combination of RPA and API integration with benefits systems",
    "time_savings_estimate": "45 minutes per enrollment",
    "sample_occurrences": [
      {
        "start_time": "2024-12-17T10:08:15",
        "end_time": "2024-12-17T10:12:45",
        "duration_minutes": 30,
        "activities": ["Process new hire benefits enrollment"]
      }
    ]
  },
  {
    "pattern_name": "Leave Request Processing",
    "pattern_type": "time_off_management",
    "frequency": "daily",
    "applications_involved": ["Microsoft Outlook", "Chrome - BambooHR"],
    "workflow_steps": [
      "Step 1: Receive leave request email",
      "Step 2: Check leave balance in BambooHR",
      "Step 3: Approve/deny request",
      "Step 4: Send response email",
      "Step 5: Update leave tracking system"
    ],
    "automation_potential": 9,
    "automation_reasoning": "Highly structured process with clear rules and digital verification",
    "business_value": "Faster response times, accurate leave tracking",
    "implementation_approach": "RPA bot with integration to HR system API",
    "time_savings_estimate": "15 minutes per request",
    "sample_occurrences": [
      {
        "start_time": "2024-12-17T11:22:15",
        "end_time": "2024-12-17T11:24:30",
        "duration_minutes": 15,
        "activities": ["Process vacation request"]
      }
    ]
  },
  {
    "pattern_name": "New Employee Profile Creation",
    "pattern_type": "onboarding",
    "frequency": "weekly",
    "applications_involved": ["Microsoft Outlook", "Chrome - BambooHR"],
    "workflow_steps": [
      "Step 1: Receive new hire notification",
      "Step 2: Create employee profile",
      "Step 3: Enter personal information",
      "Step 4: Set up job details and salary",
      "Step 5: Generate employee ID",
      "Step 6: Send welcome email"
    ],
    "automation_potential": 6,
    "automation_reasoning": "Structured but requires accuracy verification and multiple system interactions",
    "business_value": "Standardized onboarding, reduced manual data entry",
    "implementation_approach": "Workflow automation software with human verification steps",
    "time_savings_estimate": "1 hour per new hire",
    "sample_occurrences": [
      {
        "start_time": "2024-12-18T09:00:45",
        "end_time": "2024-12-18T09:05:15",
        "duration_minutes": 30,
        "activities": ["Create new employee profile and send welcome email"]
      }
    ]
  }
]
"""

import json
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

def load_data(file_path):
    """Load the JSON data file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_prompt(data):
    """Create the analysis prompt"""
    prompt = """You are an expert business process analyst. Analyze the attached sequence of user activities to identify workflow patterns, repetitive tasks, and automation opportunities.

Please analyze this workflow sequence and identify:

REPETITIVE PATTERNS: Look for sequences of activities that happen multiple times. Focus on:
   - Application switching patterns (e.g., Email → PDF → Accounting Software)
   - Data entry workflows (viewing data in one app, entering it in another)
   - Document processing flows
   - Multi-step business processes

AUTOMATION OPPORTUNITIES: For each pattern, assess:
   - How repetitive is it?
   - How structured is the data being processed?
   - How much manual work could be automated?
   - What type of automation would work (RPA, API integration, etc.)?

BUSINESS PROCESS IDENTIFICATION: What business processes do you recognize?
   - Invoice processing
   - Employee onboarding  
   - Expense reporting
   - Data entry workflows
   - Document management
   - Communication patterns

Respond with a JSON array of detected patterns. For each pattern, provide:

[
  {{
    "pattern_name": "descriptive name for the pattern",
    "pattern_type": "type of workflow (e.g., 'invoice_processing', 'data_entry', 'communication')",
    "frequency": "how often this pattern occurs (e.g., 'daily', 'multiple times per day', '3 times in dataset')",
    "applications_involved": ["App1", "App2", "App3"],
    "workflow_steps": [
      "Step 1: description",
      "Step 2: description", 
      "Step 3: description"
    ],
    "automation_potential": "score from 1-10 where 10 is highly automatable",
    "automation_reasoning": "why this can/cannot be automated",
    "business_value": "what business value would automation provide",
    "implementation_approach": "suggested approach for automation (RPA, API, custom script, etc.)",
    "time_savings_estimate": "estimated time savings per occurrence",
    "sample_occurrences": [
      {{
        "start_time": "timestamp of first step",
        "end_time": "timestamp of last step", 
        "duration_minutes": "estimated duration",
        "activities": ["activity 1", "activity 2", "activity 3"]
      }}
    ]
  }}
]

Focus on patterns that occur at least twice and involve meaningful business processes. Ignore one-off activities or random browsing.

Here's the workflow data to analyze:

"""
    
    # Convert data to readable text format
    if 'analysis_cache' in data:
        workflow_text = "HR Activity Log:\n\n"
        for filename, details in sorted(data['analysis_cache'].items()):
            timestamp = details.get('analysis_timestamp', '')
            app = details.get('application', '')
            activity = details.get('detailed_activity', '')
            workflow_text += f"{timestamp} | {app} | {activity}\n"
    else:
        workflow_text = json.dumps(data, indent=2)
    
    return prompt + workflow_text

def analyze_with_claude(prompt, api_key):
    """Send prompt to Claude and get response"""
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

def main():
    # Set up API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Load your data file
    data_file = "vlm_analysis_cache.json"
    
    try:
        print(f"Loading {data_file}...")
        data = load_data(data_file)
        
        print("Creating analysis prompt...")
        prompt = create_prompt(data)
        
        print("Sending to Claude API...")
        result = analyze_with_claude(prompt, api_key)
        
        print("\n" + "="*60)
        print("CLAUDE ANALYSIS RESULTS:")
        print("="*60)
        print(result)
        
        # Optionally save results
        with open("analysis_results.txt", "w") as f:
            f.write(result)
        print(f"\nResults saved to analysis_results.txt")
        
    except FileNotFoundError:
        print(f"Error: {data_file} not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()