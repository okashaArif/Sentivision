from flask import Flask, Response
import cv2
import torch
from flask import Flask, jsonify, request, render_template, Response
from ultralytics import YOLO
from flask_cors import CORS
import numpy as np
import time
import requests



app = Flask(__name__)
CORS(app, resources={r"/video_feed": {"origins": "*"}})
# Global Variables
coordinates = {"x": 90, "y": 90, "laser": 1}  # Default coordinates
midpoint = {"x": 90, "y": 90}  # Default midpoint
detected_objects = []  # Global list for storing detected objects
current_coordinates = {"x": 0, "y": 0}
ESP32_IP = "192.168.5.222"
ESP32_PORT = 80
ESP8266_IP = "192.168.5.195"
ESP8266_PORT = 80
ROBOT_COMMANDS = {
    "W": "F",
    "A": "L",
    "S": "B",
    "D": "R",
    "X": "S"
}
# Global Variables for the offset values
x_offset = 0  # Default X Offset
y_offset = 0   # Default Y Offset
# Video Feed URL
video_feed_url = "http://192.168.1.9:5000/video_feed"  # External video feed URL

# YOLO Model Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO('C:\\CUDA\\best.pt')  # Load the trained YOLOv8 model
model.to(device)  # Move the model to GPU if available


@app.route('/')
def index():
    return render_template('index2.html')
# Process the frame (apply YOLO and bounding box drawing)
def process_frame(frame):
    results = model(frame)

    if len(results[0].boxes) > 0:  # Check if there are detections
        for box in results[0].boxes:
            x_center, y_center, width, height = box.xywh[0]
            class_id = box.cls[0]
            confidence = box.conf[0]
            class_name = model.names[int(class_id)]

            # Draw detection on the frame
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Encode frame for streaming
    _, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()

    return frame

def process_frame1(frame ):
    global detected_objects, coordinates, midpoint
    print(f"Captured Frame Shape: {frame.shape}") 

    results = model(frame)

    detected_objects.clear()  # Clear previous detections

    if len(results[0].boxes) > 0:  # Check if there are detections
        for box in results[0].boxes:
            x_center, y_center, width, height = box.xywh[0]
            class_id = box.cls[0]
            confidence = box.conf[0]
            class_name = model.names[int(class_id)]

            # Calculate coordinates and update midpoint
            coordinates["x"], coordinates["y"] = scale_coordinates(
                x_center, y_center, frame.shape[1], frame.shape[0])
            coordinates["laser"] = 1  # Laser ON when an object is detected
            midpoint["x"] = (coordinates["x"] + 90) // 2
            midpoint["y"] = (coordinates["y"] + 90) // 2

            detected_objects.append({
                "class": class_name,
                "confidence": float(confidence),
                "coordinates": {"x": coordinates["x"], "y": coordinates["y"]}
            })
            

        # Send updated coordinates to ESP32
        # send_to_esp32(midpoint["x"], midpoint["y"], coordinates["laser"])
        
    return None

@app.route('/detect', methods=['POST'])
def detect():
    # Get the incoming JSON data
    data = request.get_json()

    # Extract the coordinates and laser status
    coordinates = data.get('coordinates', {})
    midpoint = {
            "x": coordinates.get("x", 0),
            "y": coordinates.get("y", 0),
        }
    laser_status = 1  # Adjust as needed

    # Call the send_to_esp32 function with the extracted data
    send_to_esp32(midpoint["x"], midpoint["y"], laser_status)

    # Respond back with a success message
    return jsonify({"status": "success", "message": "Detection and sending data to ESP32 successful"})

@app.route('/video_feed', methods=['GET'])
def video_feed():
    # Capture video feed from the external URL
    cap = cv2.VideoCapture(video_feed_url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      # To count frames for skipping
    def generate_frames_with_yolo():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No frame available")
                break
            
            # Resize the frame to 1280x720
            frame_resized = cv2.resize(frame, (1280, 720))

            # Process the frame with YOLO detection (detection and drawing bounding boxes)
            processed_frame = process_frame(frame_resized)  # Get processed frame with YOLO detections
            # Resize the frame to 1280x720
            frame_resized_640 = cv2.resize(frame, (640, 640))
            # Process the frame and send coordinates to ESP32 (if needed)
            process_frame1(frame_resized_640)  # Assuming this sends coordinates for laser control


            # Encode the frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')

    return Response(generate_frames_with_yolo(),
                    content_type='multipart/x-mixed-replace; boundary=frame')


# Helper Functions
def scale_coordinates(x, y, image_width, image_height):
    
    x_scaled = int((x / image_width) * 180 )
    y_scaled = int((y / image_height) * 180 )
    x_scaled = max(0, min(180, x_scaled))
    y_scaled = max(0, min(180, y_scaled))
    return x_scaled, y_scaled

def send_to_esp32(x, y, laser):
    """Send coordinates to ESP32."""
    try:
        url = f"http://{ESP32_IP}:{ESP32_PORT}/?x={x}&y={y}&laser={laser}"
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Sent to ESP32: x={x}, y={y}, laser={laser}")
        else:
            print(f"Failed to send to ESP32. Status code: {response.status_code}")
            
            
    except requests.exceptions.RequestException as e:
        print(f"Error sending data to ESP32: {e}")


@app.route('/detect_coordinates', methods=['POST'])
def detect_coordinates():
    # Get the current coordinates (could be fetched from an image detection model, for now, we return a placeholder)
    global current_coordinates

    # In a real scenario, coordinates would be dynamically fetched from detection logic
    coordinates = {
        "x": current_coordinates["x"],
        "y": current_coordinates["y"]
    }

    # Apply offsets
    coordinates["x"] += x_offset
    coordinates["y"] += y_offset

    # Send to ESP32 with the current coordinates and laser status
    send_to_esp32(coordinates["x"], coordinates["y"], 1)  # Laser status can be adjusted as needed

    return jsonify({
        "status": "success",
        "coordinates": coordinates,
        "message": "Coordinates sent to ESP32"
    })

@app.route('/update_offsets', methods=['POST'])
def update_offsets():
    """Update the X and Y offset values in real-time."""
    global x_offset, y_offset

    data = request.json
    x_offset = data.get('x_offset', x_offset)  # Default to current if not provided
    y_offset = data.get('y_offset', y_offset)  # Default to current if not provided

    return jsonify({
        "status": "success",
        "message": "Offsets updated successfully",
        "x_offset": x_offset,
        "y_offset": y_offset
    })


@app.route('/latest_updates', methods=['GET'])
def latest_updates():
    return jsonify({"objects": detected_objects})

@app.route('/control_robot', methods=['POST'])
def control_robot():
    try:
        data = request.json
        command = data.get("command", "").upper()
        if command not in ROBOT_COMMANDS:
            return jsonify({"error": "Invalid command"}), 400

        esp_command = ROBOT_COMMANDS[command]
        response = requests.get(f"http://{ESP8266_IP}:{ESP8266_PORT}/?State={esp_command}")

        if response.status_code == 200:
            return jsonify({"message": "Command sent successfully", "command": command}), 200
        else:
            return jsonify({"error": "Failed to communicate with ESP32"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
