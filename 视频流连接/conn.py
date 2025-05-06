import base64
import io
import re
from asyncio import Lock

from PIL import Image
import requests  # Use standard requests library instead of fastapi.requests
from flask import Flask, Response, jsonify  # Correct Flask imports
from openai import OpenAI

# Create Flask application instance
app = Flask(__name__)

ESP32_STREAM_URL = "http://192.168.137.182:81/stream"  # Replace with your ESP32 IP address
latest_frame = None  # Store the latest frame globally
frame_lock = Lock()  # Ensure thread safety

### Route to stream video from ESP32 ###
@app.route('/video_feed')
def video_feed():
    def generate():
        global latest_frame
        try:
            with requests.get(ESP32_STREAM_URL, stream=True, timeout=5) as r:
                buffer = b''  # Buffer to accumulate data
                boundary = b'--frame'  # MJPEG stream delimiter
                for chunk in r.iter_content(1024):
                    buffer += chunk
                    while True:
                        # Find frame start
                        start_idx = buffer.find(boundary)
                        if start_idx == -1:
                            break
                        # Find frame end
                        end_idx = buffer.find(boundary, start_idx + len(boundary))
                        if end_idx == -1:
                            break
                        # Extract complete frame data
                        frame_data = buffer[start_idx:end_idx]
                        buffer = buffer[end_idx:]  # Keep remaining data

                        # Extract JPEG data (assuming header ends with \r\n\r\n)
                        header_end = frame_data.find(b'\r\n\r\n')
                        if header_end != -1:
                            jpeg_data = frame_data[header_end + 4:]
                            # Update latest frame
                            with frame_lock:
                                latest_frame = jpeg_data

                        yield frame_data  # Yield frame to frontend
        except Exception as e:
            print("Stream Error:", e)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

### Route to capture a single frame ###
@app.route('/capture')
def capture():
    with frame_lock:
        if latest_frame is None:
            return Response(status=404)

        # Process image with PIL
        img = Image.open(io.BytesIO(latest_frame))

        # Calculate resize ratio (e.g., set width to 600px)
        base_width = 600
        w_percent = (base_width / float(img.size[0]))
        h_size = int((float(img.size[1]) * float(w_percent)))

        # High-quality resize
        img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)

        # Convert to JPEG
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)

        return Response(img_byte_arr.getvalue(), mimetype='image/jpeg')

### Route to identify species in the captured frame ###
@app.route('/identify', methods=['POST'])
def identify_species():
    with frame_lock:
        if latest_frame is None:
            return jsonify({"error": "No frame available"}), 400

        # Process image
        img = Image.open(io.BytesIO(latest_frame))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=85)
        base64_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # Call AI model (Qwen-VL)
        client = OpenAI(
            api_key='sk-dd08879040994cb196446a5707b1b1cd',
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # First round: Identify species
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "你是一个生物学家助手"}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    },
                    {"type": "text", "text": "识别图中的动植物，直接给出学名以及一到两个别名"}
                ]
            }
        ]

        completion = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=messages,
        )
        species_info = completion.choices[0].message.content

        # Second round: Get species description
        species_name = re.sub(r"[^a-zA-Z\u4e00-\u9fa5]", "", species_info)[:10]
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"对这个动物或者植物做一个30字的简介"}
            ]
        })

        completion = client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=messages,
        )
        description = completion.choices[0].message.content

        return jsonify({
            "species": species_info,
            "description": description
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)