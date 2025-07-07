from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
from collections import deque
from ultralytics import YOLO
import os
from datetime import datetime
import folium

app = Flask(__name__)
processing_done = False

# === CONFIGURATION ===
MODEL_PATH = 'fire_smoke_detection_model.pt'
VIDEO_PATH = 'dataset/clip_001.avi'
OUTPUT_DIR = 'fire_clips'
CONFIDENCE_THRESHOLD = 0.3
PRE_FRAMES = 75
POST_FRAMES = 75

# === SETUP ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buffer = deque(maxlen=PRE_FRAMES)
recording = False
post_counter = 0
clip_count = 0
out = None
frame_idx = 0
dashboard_data = []
video_ready = False

def fire_detected(results):
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = r.names[cls].lower()
            if label in ['smoke', 'fire'] and conf > CONFIDENCE_THRESHOLD:
                return True
    return False
def create_map_with_google_link(lat, lon, output_file="fire_clips/location_map.html"):
    m = folium.Map(location=[lat, lon], zoom_start=15)
    google_maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    popup_html = f'<a href="{google_maps_link}" target="_blank">Open in Google Maps</a>'
    folium.Marker([lat, lon], popup=popup_html, tooltip="Click for Google Maps").add_to(m)
    m.save(output_file)

def save_frame_image(frame, timestamp, suffix):
    filename = os.path.join(OUTPUT_DIR, f"frame_{timestamp}_{suffix}.jpg")
    cv2.imwrite(filename, frame)
    return filename

def generate_frames():
    global recording, post_counter, clip_count, out, frame_idx, dashboard_data, video_ready, processing_done
    detection_frame_counter = 0
    detection_timestamp = None
    saved_images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        video_ready = True
        buffer.append(frame)
        results = model.predict(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls]
                if conf > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if fire_detected(results):
            if not recording:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_date = datetime.now().strftime('%Y-%m-%d')
                log_time = datetime.now().strftime('%H:%M:%S')
                clip_name = os.path.join(OUTPUT_DIR, f"clip_{timestamp}.mp4")
                out = cv2.VideoWriter(clip_name, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
                for bframe in buffer:
                    out.write(bframe)

                with open(os.path.join(OUTPUT_DIR, "detection_log.txt"), "a") as log:
                    log.write(f"Date       : {log_date}\n")
                    log.write(f"Time       : {log_time}\n")
                    log.write(f"Video File : {clip_name}\n")
                    log.write("-" * 40 + "\n")

                create_map_with_google_link(22.7909362, 87.4380579, os.path.join(OUTPUT_DIR, "location_map.html"))

                img1 = save_frame_image(frame, timestamp, "detected")
                saved_images = [img1]
                detection_timestamp = timestamp
                detection_frame_counter = 1

                dashboard_data.append({
                    "date": log_date,
                    "time": log_time,
                    "video": clip_name,
                    "map": "location_map.html",
                    "images": saved_images
                })

                recording = True
                post_counter = POST_FRAMES
                clip_count += 1
            else:
                post_counter = POST_FRAMES

        if detection_frame_counter > 0:
            detection_frame_counter += 1
            if detection_frame_counter == 5 * fps:
                img2 = save_frame_image(frame, detection_timestamp, "after_5s")
                dashboard_data[-1]["images"].append(img2)
            elif detection_frame_counter == 10 * fps:
                img3 = save_frame_image(frame, detection_timestamp, "after_10s")
                dashboard_data[-1]["images"].append(img3)
                detection_frame_counter = 0

        if recording:
            out.write(frame)
            post_counter -= 1
            if post_counter <= 0:
                recording = False
                out.release()

        ret, buffer_img = cv2.imencode('.jpg', frame)
        frame_bytes = buffer_img.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_idx += 1

    cap.release()
    if out:
        out.release()
    processing_done = True
    print("🎬 Video processing completed.")
    return

@app.route('/')
def index():
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dashboard_data.clear()
    return render_template('index.html', dashboard=dashboard_data)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_status')
def video_status():
    return jsonify({"ready": video_ready})

@app.route('/fire_clips/<path:filename>')
def serve_clip(filename):
    return send_from_directory('fire_clips', filename)

@app.route('/results')
def results():
    return render_template('results.html', dashboard=dashboard_data)

@app.route('/processing_status')
def processing_status():
    return jsonify({"done": processing_done})

@app.route('/fire_images')
def fire_images():
    images = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')]
    return render_template('fire_images.html', images=images)


if __name__ == '__main__':
    app.run(debug=True)
