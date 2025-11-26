# detect_drone.py
# Simple YOLOv8 demo: webcam/video -> detect -> log to SQLite -> show window
from ultralytics import YOLO
import cv2
import sqlite3
from datetime import datetime
import os

# ------------ CONFIG ------------
VIDEO_SOURCE = 0            # 0 = default webcam, or "C:/full/path/to/video.mp4"
MODEL_NAME = "yolov8n.pt"   # small & fast. Replace with custom weights later.
CONF_THRESHOLD = 0.25
LOG_DB = "detections.db"
SHOW_WINDOW = True          # Set False if running on headless server
# --------------------------------

# Initialize DB
conn = sqlite3.connect(LOG_DB, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS detections (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 label TEXT,
 confidence REAL,
 timestamp TEXT
)
""")
conn.commit()

# Load model
print("Loading model:", MODEL_NAME)
model = YOLO(MODEL_NAME)

# Open video source
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: cannot open video source:", VIDEO_SOURCE)
    exit(1)

print("Starting detection. Press 'q' to quit the window (or Ctrl+C to stop).")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame (end of video or cannot read frame). Exiting.")
            break

        results = model.predict(source=frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
        r = results[0]
        annotated = frame.copy()

        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = r.names[cls_id] if hasattr(r, "names") else str(cls_id)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(y1-10,10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                timestamp = datetime.utcnow().isoformat(timespec='seconds') + "Z"
                c.execute("INSERT INTO detections (label, confidence, timestamp) VALUES (?, ?, ?)",
                          (label, conf, timestamp))
                conn.commit()

        if SHOW_WINDOW:
            cv2.imshow("SkyGuard - Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    conn.close()
    print("Exited cleanly.")
