from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI(title="Proctoring Multiple Person Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MultiPersonDetector:
    def __init__(self, model_path="blaze_face_short_range.tflite", min_detection_confidence=0.65):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence  # Higher confidence to block false noise
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def get_face_count(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        return len(detection_result.detections) if detection_result.detections else 0

# Initialized engine with slightly higher strictness
detector = MultiPersonDetector(model_path="blaze_face_short_range.tflite", min_detection_confidence=0.65)

@app.post("/detect")
async def detect_persons(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"error": "Invalid image file"}

    face_count = detector.get_face_count(frame)
    violation = face_count > 1

    return {
        "person_count": face_count,
        "violation": violation
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)