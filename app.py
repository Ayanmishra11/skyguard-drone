# app.py
# Minimal Flask app to stream annotated frames from webcam using YOLOv8
from flask import Flask, Response, render_template_string
import cv2
from ultralytics import YOLO

app = Flask(__name__)

VIDEO_SOURCE = 0
MODEL = YOLO("yolov8n.pt")
CONF_THRESHOLD = 0.25
CAP = cv2.VideoCapture(VIDEO_SOURCE)

HTML = """
<!doctype html>
<html>
  <head>
    <title>SkyGuard - Live</title>
    <style>
      body { font-family: Arial, sans-serif; text-align:center; margin:20px; }
      img { border: 2px solid #333; }
    </style>
  </head>
  <body>
    <h2>SkyGuard — Live Stream</h2>
    <p>Press Ctrl+C in terminal to stop the server.</p>
    <img src="{{ url_for('video_feed') }}" width="800" />
  </body>
</html>
"""

def gen_frames():
    while True:
        success, frame = CAP.read()
        if not success:
            break

        results = MODEL.predict(source=frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
        r = results[0]
        annotated = frame.copy()
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = r.names[cls_id] if hasattr(r, "names") else str(cls_id)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(y1-10,10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        CAP.release()
