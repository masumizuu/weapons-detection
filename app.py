from flask import Flask, render_template, request, jsonify
import time
from ultralytics import YOLO
import cv2
import base64
import numpy as np
import os

app = Flask(__name__)

detection_logs = []

# Load your model
model = YOLO("best.pt")
class_names = model.names

# Open webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("❌ Failed to open camera. Check permissions or device index.")
# else:
#     print("✅ Camera opened successfully.")

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize for better small object detection
        scaled = cv2.resize(frame, (960, 960))
        results = model(scaled, conf=0.1)

        x_scale = frame.shape[1] / 960
        y_scale = frame.shape[0] / 960

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{class_names[cls_id]} {conf:.2f}"

                # Log it
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                detection_logs.append(f"[{timestamp}] {label}")

                # Keep logs short (last 50 only)
                if len(detection_logs) > 50:
                    detection_logs.pop(0)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 * x_scale)
                y1 = int(y1 * y_scale)
                x2 = int(x2 * x_scale)
                y2 = int(y2 * y_scale)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def get_logs():
    return jsonify(detection_logs[::-1])  # reverse to show latest first

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image received'}), 400

    # Decode base64 image
    image_data = data['image'].split(',')[1]  # remove data:image/jpeg;base64,
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLOv11 Inference
    # Resize for consistent detection (YOLO performs better around 640-960)
    scaled = cv2.resize(frame, (960, 960))
    results = model(scaled, conf=0.1)

    x_scale = frame.shape[1] / 960
    y_scale = frame.shape[0] / 960

    logs = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{class_names[cls_id]} {conf:.2f}"

            # Log it
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            detection_logs.append(f"[{timestamp}] {label}")
            if len(detection_logs) > 50:
                detection_logs.pop(0)

            # Draw on frame
            x1, y1, x2, y2 = box.xyxy[0]
            x1 = int(x1 * x_scale)
            y1 = int(y1 * y_scale)
            x2 = int(x2 * x_scale)
            y2 = int(y2 * y_scale)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Encode frame back to base64
    _, buffer = cv2.imencode('.jpg', frame)
    result_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'result': f'data:image/jpeg;base64,{result_base64}'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))