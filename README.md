🔥 -Real-Time-Fire-Smoke-Detection-System-Empowering-Safety-with-AI-
Excited to share a project I've been working on that leverages computer vision and deep learning to enhance fire safety at CCTV-monitored sites!  Using a custom-trained YOLO model, our system detects fire and smoke in real-time from surveillance footage


🔥 Fire & Smoke Detection System
A real-time fire and smoke detection system using YOLOv8, OpenCV, and Flask, designed to monitor CCTV footage, detect hazards, and alert rescue teams with location data and visual evidence.

🚀 Features
Real-Time Detection: Monitors live CCTV footage for fire and smoke using a trained YOLO model.
Alert System: Automatically logs detections and sends alerts with timestamp and location.
Location Mapping: Generates a clickable Google Maps link for quick navigation to the incident site.
Evidence Capture: Saves key frames and video clips before and after detection.
Dashboard Interface: Displays detection logs, images, and maps for rescue teams.
Web-Based UI: Built with Flask for easy deployment and access.
🧠 Tech Stack
YOLOv8 – Object detection model for identifying fire and smoke.
OpenCV – Video processing and frame manipulation.
Flask – Web framework for serving the dashboard and video feed.
Folium – Interactive map generation with Google Maps integration.
HTML/CSS – Frontend templates for dashboard and image viewer.
📂 Project Structure
fire_smoke_detection/
│
├── fire_clips/               # Saved clips and images
├── templates/                # HTML templates
│   ├── index.html
│   ├── results.html
│ 
├── dataset/                  # Input video files
├── fire_smoke_detection_model.pt  # YOLO model file
├── app.py                    # Main Flask application
└── README.md                 # Project documentation

🛠️ Setup Instructions
Clone the repository:
gh repo clone KUNTALinGITHUB/-Real-Time-Fire-Smoke-Detection-System-Empowering-Safety-with-AI-
cd fire-smoke-detection


Install dependencies:
pip install -r requirements.txt


Add your trained YOLO model: Place your fire_smoke_detection_model.pt in the root directory.

Run the application:
python app.py


Access the dashboard: Open your browser and go to http://localhost:5000

📸 Sample Output
Real-time detection feed
Detection logs with timestamps
Saved images before and after detection
Google Maps link to incident location

📍 Location Integration
Each detection is tagged with GPS coordinates and linked to Google Maps for rapid response:

https://www.google.com/maps?q=22.7909362,87.4380579

📬 Contact
For questions, collaborations, or deployment inquiries, feel free to reach out via LinkedIn or open an issue.

https://github.com/user-attachments/assets/4c77106e-c469-454c-baa2-4c3ff1c24f8c

