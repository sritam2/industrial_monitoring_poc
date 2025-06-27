import cv2
import numpy as np
import base64
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def capture_and_store_image():
    """Capture a fresh image from camera and store it to disk"""
    print("📸 Step 3A: Capture and Store Image")
    print("=" * 40)
    
    # Initialize camera with DirectShow backend
    camera_index = 1
    print(f"🎥 Initializing camera {camera_index} with DirectShow backend...")
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return None
    
    # Capture frame from camera
    print("📷 Capturing frame from camera...")
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Failed to capture frame from camera")
        cap.release()
        return None
    
    # Validate captured frame
    mean_pixel = np.mean(frame)
    print(f"✅ Frame captured successfully!")
    print(f"   📐 Frame dimensions: {frame.shape}")
    print(f"   💡 Mean pixel value: {mean_pixel:.2f}")
    
    if mean_pixel < 5:
        print("⚠️  Warning: Image appears very dark/black")
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"slot_monitoring_{timestamp}.jpg"
    
    # Store image to disk
    print(f"💾 Storing image to disk...")
    success = cv2.imwrite(filename, frame)
    
    if success:
        file_size = os.path.getsize(filename)
        print(f"✅ Image successfully stored!")
        print(f"   📄 Filename: {filename}")
        print(f"   📊 File size: {file_size:,} bytes")
    else:
        print(f"❌ Failed to store image to disk")
        cap.release()
        return None
    
    # Clean up camera resource
    cap.release()
    print(f"🔄 Camera resource released")
    
    return filename

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_with_openai(image_path):
    """Load stored image and send to OpenAI for slot analysis"""
    print("\n🤖 Step 3B: OpenAI Analysis of Stored Image")
    print("=" * 40)
    
    print(f"📂 Loading stored image: {image_path}")
    
    # Verify image file exists
    if not os.path.exists(image_path):
        print(f"❌ Stored image file not found: {image_path}")
        return None
    
    print(f"✅ Stored image file found and accessible")
    
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return None
    
    client = OpenAI(api_key=api_key)
    
    # Encode the stored image
    print("🔄 Encoding stored image for OpenAI...")
    base64_image = encode_image(image_path)
    print(f"✅ Image encoded to base64 ({len(base64_image)} characters)")
    
    # Create the industrial monitoring prompt
    system_prompt = """You are an industrial vision monitoring system. You must analyze blocks and map them to their EXACT PHYSICAL LOCATION.

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
    
    print("🚀 Sending stored image to OpenAI GPT-4o-mini...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
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
        
        # Get the response
        ai_response = response.choices[0].message.content.strip()
        print("✅ Received response from OpenAI!")
        print(f"📤 Raw response: {ai_response}")
        
        # Parse and validate JSON
        try:
            parsed_response = json.loads(ai_response)
            print("✅ Response is valid JSON!")
            print(f"📋 Structured response:")
            print(json.dumps(parsed_response, indent=2))
            
            # Validate the response format
            expected_keys = ["slot_0", "slot_1", "slot_2"]
            if all(key in parsed_response for key in expected_keys):
                print("✅ Response format is correct!")
                
                # Analyze the results
                print("\n📊 Stored Image Analysis Results:")
                expected_serials = ["685", "923", "742"]
                
                for i, slot in enumerate(["slot_0", "slot_1", "slot_2"]):
                    value = parsed_response[slot]
                    expected = expected_serials[i]
                    
                    if value is None:
                        status = "🟡 VACANT"
                        state = "VALID"
                    elif value == expected:
                        status = "🟢 CORRECT"
                        state = "VALID"
                    else:
                        status = f"🔴 WRONG (expected {expected})"
                        state = "VIOLATION"
                    
                    print(f"   Slot {i}: {value} - {status} ({state})")
                
                return parsed_response
            else:
                print(f"❌ Invalid response format. Expected keys: {expected_keys}")
                return None
                
        except json.JSONDecodeError:
            print("❌ Response is not valid JSON")
            print("Raw response might contain extra text")
            return None
            
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {str(e)}")
        return None

if __name__ == "__main__":
    print("🎯 Step 3: Capture → Store → OpenAI Analysis")
    print("=" * 50)
    
    # Step 3A: Capture and store image
    stored_image_path = capture_and_store_image()
    
    if stored_image_path:
        print(f"\n✅ Image capture and storage completed!")
        
        # Step 3B: Analyze stored image with OpenAI
        result = analyze_with_openai(stored_image_path)
        
        if result:
            print(f"\n🎉 SUCCESS! Complete workflow executed!")
            print(f"📋 Final JSON Result: {json.dumps(result)}")
            print(f"💾 Analyzed image stored as: {stored_image_path}")
            print("✅ Ready for continuous monitoring system!")
        else:
            print(f"\n❌ OpenAI analysis failed - check the error messages above")
    else:
        print(f"\n❌ Image capture and storage failed - check camera connection") 