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
import threading
import io
from PIL import Image

# Load environment variables
load_dotenv()

# Global variables for graceful shutdown
monitoring_active = True
display_window_name = "Industrial Monitoring System"
stream_window_name = "Live Camera Feed"
camera_stream = None  # Global reference to camera stream

class CameraStream:
    def __init__(self, camera_index=0):
        """Initialize camera stream thread"""
        self.camera_index = camera_index
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None
        
    def start(self):
        """Start the camera stream thread"""
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
        
    def update(self):
        """Update frames from camera"""
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("❌ Failed to open camera in stream thread")
            self.stopped = True
            return
            
        while not self.stopped:
            ret, frame = cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()
                
                # Display frame in stream window
                if not self.stopped:  # Check again before displaying
                    cv2.imshow(stream_window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stopped = True
                        break
            else:
                print("❌ Failed to read frame in stream thread")
                break
                
        cap.release()
        cv2.destroyWindow(stream_window_name)
        
    def read(self):
        """Read the current frame"""
        with self.lock:
            return None if self.frame is None else self.frame.copy()
            
    def stop(self):
        """Stop the camera stream thread"""
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)  # Wait for thread to finish
        cv2.destroyAllWindows()  # Ensure all windows are closed

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global monitoring_active, camera_stream
    print("\n🛑 Shutdown signal received. Stopping monitoring...")
    monitoring_active = False
    
    # Stop the camera stream if it exists
    if camera_stream:
        camera_stream.stop()
    
    # Ensure all windows are closed
    cv2.destroyAllWindows()
    
    # Force exit if cleanup takes too long
    time.sleep(1)
    sys.exit(0)

# Set up signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string"""
    # Convert from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')
    
    # Get base64 string
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image

def capture_frame(camera_stream):
    """Capture frame from camera stream"""
    frame = camera_stream.read()
    
    if frame is None:
        print("❌ Failed to get frame from stream")
        return None, None
    
    # Create a copy for display
    display_frame = frame.copy()
    return frame, display_frame

def analyze_with_openai(frame):
    """Analyze frame with OpenAI"""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY not found")
            return None
        
        client = OpenAI(api_key=api_key)
        base64_image = frame_to_base64(frame)
        
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
            model="gpt-4-vision-preview",
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
        print(f"❌ OpenAI analysis error: {str(e)}")
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

def update_display(frame, result, violations):
    """Update display with analysis results"""
    if frame is None:
        return
    
    display_frame = frame.copy()
    height, width = display_frame.shape[:2]
    slot_width = width // 3
    
    # Draw results for each slot
    for i, (slot, value) in enumerate(result.items()):
        # Calculate position for text
        x = (i * slot_width) + (slot_width // 2) - 100
        y = 70  # Position below timestamp
        
        # Determine status color
        if value is None:
            status = "VACANT"
            color = (255, 255, 0)  # Yellow
        elif any(v["slot"] == slot for v in violations):
            status = f"WRONG: {value}"
            color = (0, 0, 255)    # Red
        else:
            status = f"OK: {value}"
            color = (0, 255, 0)    # Green
            
        # Draw status
        cv2.putText(display_frame, 
                   status,
                   (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.8,
                   color,
                   2)
    
    # Show updated frame
    cv2.imshow(display_window_name, display_frame)
    cv2.waitKey(1)  # Update display

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
    global camera_stream  # Use global reference
    
    print("🎯 Step 5: Continuous Slot Monitoring System with Display")
    print("=" * 60)
    print("📊 Monitoring Configuration:")
    print("   - Frequency: Every 10 seconds")
    print("   - Expected serials: slot_0='685', slot_1='923', slot_2='742'")
    print("   - Camera: Index 1 (Logitech)")
    print("   - Analysis: OpenAI GPT-4-vision-preview")
    print("   - Display: Real-time camera feed with status overlay")
    print("   - Stream: Independent camera feed window")
    print("=" * 60)
    print("🚀 Starting continuous monitoring...")
    print("⏹️  Press Ctrl+C to stop monitoring")
    print("=" * 60)
    
    monitoring_count = 0
    
    try:
        # Start camera stream
        camera_stream = CameraStream(camera_index=0)
        camera_stream.start()
        
        while monitoring_active:
            try:
                monitoring_count += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"\n📸 Monitor cycle #{monitoring_count} at {timestamp}")
                
                # Step 1: Capture frame
                frame, display_frame = capture_frame(camera_stream)
                if frame is None:
                    print("❌ Frame capture failed, skipping cycle")
                    if monitoring_active:  # Check if we should continue waiting
                        time.sleep(10)
                    continue
                
                print("✅ Frame captured")
                
                # Step 2: Analyze with OpenAI
                result = analyze_with_openai(frame)
                if not result:
                    print("❌ OpenAI analysis failed, skipping cycle")
                    if monitoring_active:  # Check if we should continue waiting
                        time.sleep(10)
                    continue
                
                # Step 3: Check for violations
                violations = check_violations(result)
                
                # Step 4: Update display with results
                if monitoring_active:  # Only update display if still running
                    update_display(display_frame, result, violations)
                
                # Step 5: Log status and handle alarms
                log_status(timestamp, result, violations)
                
                if violations:
                    raise_alarm(violations, timestamp)
                
                # Wait 10 seconds before next cycle
                if monitoring_active:
                    print("⏳ Waiting 10 seconds for next cycle...")
                    time.sleep(10)
                    
            except Exception as e:
                print(f"❌ Error in monitoring cycle: {str(e)}")
                if monitoring_active:  # Check if we should continue waiting
                    print("⏳ Waiting 10 seconds before retry...")
                    time.sleep(10)
    
    finally:
        # Ensure cleanup happens even if an error occurs
        if camera_stream:
            camera_stream.stop()
        cv2.destroyAllWindows()
    
    print("\n🛑 Monitoring stopped.")
    print("📊 Total monitoring cycles completed:", monitoring_count)

if __name__ == "__main__":
    continuous_monitor() 
