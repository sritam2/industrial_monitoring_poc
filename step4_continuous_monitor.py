import cv2
import numpy as np
import base64
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import time
import signal
import sys

# Load environment variables
load_dotenv()

# Global variable for graceful shutdown
monitoring_active = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global monitoring_active
    print("\n🛑 Shutdown signal received. Stopping monitoring...")
    monitoring_active = False

# Set up signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

def capture_image():
    """Capture image from camera"""
    camera_index = 1
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return None
    
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Failed to capture frame")
        cap.release()
        return None
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"monitor_{timestamp}.jpg"
    
    # Save image
    success = cv2.imwrite(filename, frame)
    cap.release()
    
    if success:
        return filename
    else:
        print("❌ Failed to save image")
        return None

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_with_openai(image_path):
    """Analyze image with OpenAI"""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY not found")
            return None
        
        client = OpenAI(api_key=api_key)
        base64_image = encode_image(image_path)
        
        system_prompt = """You are an industrial vision monitoring system. Analyze the image to detect serial numbers on blocks and map them to their EXACT PHYSICAL LOCATION in 3 slots.

**CRITICAL SPATIAL POSITIONING**:
The image shows 3 vertical slots separated by black lines. You must determine the EXACT LOCATION of each block:

**SLOT MAPPING (CRITICAL)**:
- **slot_0** = LEFTMOST position (far left of image)
- **slot_1** = MIDDLE position (center of image)
- **slot_2** = RIGHTMOST position (far right of image)

**STEP-BY-STEP PROCESS**:
1. Look at the image from LEFT to RIGHT
2. Identify which slot each block physically occupies
3. If a block is on the LEFT side → it goes in slot_0
4. If a block is in the MIDDLE → it goes in slot_1  
5. If a block is on the RIGHT side → it goes in slot_2

**CRITICAL RULES**:
- A block in the LEFTMOST physical position must be mapped to slot_0
- A block in the MIDDLE physical position must be mapped to slot_1
- A block in the RIGHTMOST physical position must be mapped to slot_2
- Empty slots get null
- NEVER fill slots in order of detection - map by ACTUAL POSITION

Return ONLY this JSON format:
{"slot_0": "value_or_null", "slot_1": "value_or_null", "slot_2": "value_or_null"}"""

        user_prompt = """CRITICAL SPATIAL MAPPING - Follow these exact steps:

**STEP 1**: Divide the image into 3 equal vertical sections:
- LEFT third = slot_0 region
- MIDDLE third = slot_1 region  
- RIGHT third = slot_2 region

**STEP 2**: For each block you detect, determine which THIRD of the image it's in:
- Is the block in the LEFT third? → slot_0
- Is the block in the MIDDLE third? → slot_1
- Is the block in the RIGHT third? → slot_2

**STEP 3**: Build the JSON response based on PHYSICAL LOCATION:

**VISUAL GUIDE**:
```
[LEFT third]  |  [MIDDLE third]  |  [RIGHT third]
   slot_0     |     slot_1       |     slot_2
```

**EXAMPLES**:
- Block "685" in LEFT third → {"slot_0": "685", "slot_1": null, "slot_2": null}
- Block "923" in MIDDLE third → {"slot_0": null, "slot_1": "923", "slot_2": null}
- Block "742" in RIGHT third → {"slot_0": null, "slot_1": null, "slot_2": "742"}
- Blocks in LEFT and MIDDLE → {"slot_0": "XXX", "slot_1": "YYY", "slot_2": null}

Look at each block's position relative to the black dividing lines and map accordingly.

Return ONLY the JSON."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        
        ai_response = response.choices[0].message.content.strip()
        return json.loads(ai_response)
        
    except Exception as e:
        print(f"❌ OpenAI analysis failed: {str(e)}")
        return None

def check_violations(result):
    """Check for violations and return alarm details"""
    expected_serials = {
        "slot_0": "685",
        "slot_1": "923", 
        "slot_2": "742"
    }
    
    violations = []
    
    for slot, detected in result.items():
        expected = expected_serials[slot]
        
        if detected is None:
            # Slot is vacant - this is allowed
            continue
        elif detected == expected:
            # Correct serial number - no violation
            continue
        else:
            # Wrong serial number - violation!
            violations.append({
                "slot": slot,
                "expected": expected,
                "detected": detected,
                "reason": f"{slot} has wrong serial '{detected}' (expected '{expected}')"
            })
    
    return violations

def raise_alarm(violations, timestamp):
    """Raise alarm for violations"""
    print("\n🚨" + "="*60 + "🚨")
    print("🚨                     VIOLATION ALARM                     🚨")
    print("🚨" + "="*60 + "🚨")
    print(f"⏰ Timestamp: {timestamp}")
    print(f"🔴 Number of violations: {len(violations)}")
    print("\n📋 VIOLATION DETAILS:")
    
    for i, violation in enumerate(violations, 1):
        print(f"   {i}. {violation['reason']}")
    
    print("\n🚨 IMMEDIATE ACTION REQUIRED! 🚨")
    print("🚨" + "="*60 + "🚨\n")

def log_status(timestamp, result, violations):
    """Log monitoring status"""
    status = "🔴 VIOLATION" if violations else "🟢 OK"
    print(f"[{timestamp}] {status} - {json.dumps(result)}")
    
    if not violations:
        # Show status of each slot when OK
        slot_status = []
        for slot, value in result.items():
            if value is None:
                slot_status.append(f"{slot}:VACANT")
            else:
                slot_status.append(f"{slot}:{value}")
        print(f"             Slots: {' | '.join(slot_status)}")

def continuous_monitor():
    """Main continuous monitoring loop"""
    print("🎯 Step 4: Continuous Slot Monitoring System")
    print("=" * 60)
    print("📊 Monitoring Configuration:")
    print("   - Frequency: Every 10 seconds")
    print("   - Expected serials: slot_0='685', slot_1='923', slot_2='742'")
    print("   - Camera: Index 1 (Logitech)")
    print("   - Analysis: OpenAI GPT-4o-mini")
    print("=" * 60)
    print("🚀 Starting continuous monitoring...")
    print("⏹️  Press Ctrl+C to stop monitoring")
    print("=" * 60)
    
    monitoring_count = 0
    
    while monitoring_active:
        try:
            monitoring_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n📸 Monitor cycle #{monitoring_count} at {timestamp}")
            
            # Step 1: Capture image
            image_path = capture_image()
            if not image_path:
                print("❌ Image capture failed, skipping cycle")
                time.sleep(10)
                continue
            
            print(f"✅ Image captured: {image_path}")
            
            # Step 2: Analyze with OpenAI
            result = analyze_with_openai(image_path)
            if not result:
                print("❌ OpenAI analysis failed, skipping cycle")
                time.sleep(10)
                continue
            
            # Step 3: Check for violations
            violations = check_violations(result)
            
            # Step 4: Log status and handle alarms
            log_status(timestamp, result, violations)
            
            if violations:
                raise_alarm(violations, timestamp)
            
            # Clean up old image files (keep only last 5)
            cleanup_old_images()
            
            # Wait 10 seconds before next cycle
            if monitoring_active:
                print("⏳ Waiting 10 seconds for next cycle...")
                time.sleep(10)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error in monitoring cycle: {str(e)}")
            print("⏳ Waiting 10 seconds before retry...")
            time.sleep(10)
    
    print("\n🛑 Monitoring stopped.")
    print("📊 Total monitoring cycles completed:", monitoring_count)

def cleanup_old_images():
    """Keep only the last 5 monitoring images"""
    try:
        image_files = [f for f in os.listdir('.') if f.startswith('monitor_') and f.endswith('.jpg')]
        if len(image_files) > 5:
            image_files.sort()
            for old_file in image_files[:-5]:
                os.remove(old_file)
    except Exception as e:
        pass  # Silent cleanup failure

if __name__ == "__main__":
    continuous_monitor() 