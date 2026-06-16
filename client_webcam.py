import cv2
import requests
import json

API_URL = "http://127.0.0.1:8000/detect"

# The buffer threshold for entry trigger
CONSECUTIVE_FRAME_THRESHOLD = 4  
violation_frame_counter = 0

cap = cv2.VideoCapture(0)
print("Running optimized high-accuracy stream. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

    try:
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            api_data = response.json()
            face_count = api_data["person_count"]
            
            # --- FIXED HIGH-ACCURACY LOGIC ---
            if face_count > 1:
                violation_frame_counter += 1
            else:
                # INSTANT RESET: If the 2nd person leaves, clear the counter immediately!
                violation_frame_counter = 0

            # Determine actual violation state based on the clean counter threshold
            actual_violation = violation_frame_counter >= CONSECUTIVE_FRAME_THRESHOLD

            display_json = {
                "person_count": face_count,
                "violation": actual_violation
            }
            print("Live Data Sync:", json.dumps(display_json))

            # Visual overlay configuration
            status_text = f"Faces: {face_count} | Violation: {actual_violation}"
            
            # Color turns green instantly when face_count drops back to 1
            color = (0, 0, 255) if actual_violation else (0, 255, 0)
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
    except requests.exceptions.ConnectionError:
        print("Error: Connection lost. Is app.py running?")
        break

    cv2.imshow("High-Accuracy Test Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()