# 🛡️ Multiple Person Detection Module (Proctoring System)

**Phase:** 3/4 | **Difficulty:** Medium–High

**Objective:** A real-time computer vision API built to detect multiple people in a frame, prevent external assistance during online assessments, and stream structured JSON violation telemetry.

---

## 📖 Project Overview

This module is a core component of an AI Proctoring System. It utilizes **Google's MediaPipe Tasks API** and **FastAPI** to actively monitor a video feed. If more than one face is detected in the frame, the system triggers a structured violation alert. 

The architecture is split into a robust Backend API (`app.py`) and a real-time testing client (`client_webcam.py`) to simulate production-level network traffic.

### 🚀 Core Deliverables Achieved
* **Face Counting Logic:** Accurately counts faces per frame using `blaze_face_short_range`.
* **Threshold Configuration:** Modular confidence levels to filter out noise.
* **API Output:** Standardized JSON payload (`{"person_count": X, "violation": bool}`).
* **Live Integration Testing:** Real-time webcam client to validate API responsiveness.

---

## ⚙️ Edge Case Handling & Constraints

This system was engineered to handle real-world proctoring environments cleanly without throwing false positives:

1. **Background Faces (Posters/Screens):** * *Solution:* **Confidence Thresholding**. Configured `min_detection_confidence=0.65` in `app.py`. Small, blurry, or 2D printed faces on posters fall below this threshold and are ignored.
2. **Reflections & Mirrors:** * *Solution:* The MediaPipe Face Detection model inherently drops low-contrast, distorted structural features commonly found in faint window or mirror reflections.
3. **Rapid Entry/Exit (Flickering):** * *Solution:* **Temporal Frame Buffering**. Handled in `client_webcam.py` via `CONSECUTIVE_FRAME_THRESHOLD = 4`. A second person must be present for sustained frames to trigger a true violation, preventing false alarms from brief camera glitches or someone quickly walking past a doorway. The penalty resets instantly (`violation_frame_counter = 0`) the moment the frame is clear.

---

## 📂 Project Structure

```text
MULTI_PERSON_DETECTION/
│
├── app.py                          # FastAPI Server & MediaPipe Detection Engine
├── client_webcam.py                # Live Video Capture & Buffer Logic Client
├── blaze_face_short_range.tflite   # Pre-trained ML weights file
├── README.md                       # Project Documentation
├── config.json
└── requirements.txt

```
---

## 💻 Installation & Setup

## 1. Clone the Repository and Navigate to the Project Folder

```bash
cd Multiple_person_detection
```

## 2. Create and Activate a Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```


## 3. Install the Required Dependencies

Ensure you have **Python 3.8+** installed, then run:

```bash
pip install -r requirements.txt
```

## 4. Download the Model File

Make sure the file `blaze_face_short_range.tflite` is present in the project root directory alongside `app.py`.

---


# 🏃‍♂️ How to Run the Application

Since this project simulates a real web application, you must run both the **Backend API Server** and the **Live Testing Client** simultaneously in two separate terminal windows.

## Step 1: Start the Backend API Server

Open your first terminal window and run:

```bash
python app.py
```

The server will start listening for incoming image frames at:

```text
http://127.0.0.1:8000/detect
```

---

## Step 2: Start the Live Testing Client

Open a second terminal window and run:

```bash
python client_webcam.py
```

Your webcam will turn on automatically. The terminal will display live JSON responses, while the video window will show real-time face bounding boxes and violation telemetry.

### Controls

* Press **`q`** while the webcam window is focused to safely exit the live stream.

---

# 🌐 API Documentation & Manual Testing

FastAPI automatically generates interactive API documentation for this module.

### Steps

1. Ensure `app.py` is running.
2. Open your browser and navigate to:

```text
http://127.0.0.1:8000/docs
```

3. Expand the **POST `/detect`** endpoint.
4. Click **Try it out**.
5. Upload a static image containing one or more faces.
6. Click **Execute**.
7. Review the standardized JSON response.

---

# 📦 Expected JSON Response

```json
{
  "person_count": 2,
  "violation": true
}
```
