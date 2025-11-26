# detect_drone_tracker.py
# YOLOv8 detection + simple centroid tracker to assign IDs and reduce duplicate logs.
from ultralytics import YOLO
import cv2
import sqlite3
from datetime import datetime
import time
import numpy as np
import collections

# ---------------- CONFIG ----------------
VIDEO_SOURCE = 0            # 0 = webcam or "C:/path/to/video.mp4"
MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.30       # raise slightly to reduce weak detections
LOG_DB = "detections.db"
SHOW_WINDOW = True
MAX_DISAPPEARED = 10        # frames before removing a tracked object
DISTANCE_THRESHOLD = 50     # pixels for centroid matching
LOG_COOLDOWN_SEC = 8        # seconds before logging same track again
# ----------------------------------------

# --- DB setup ---
conn = sqlite3.connect(LOG_DB, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS detections (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 track_id INTEGER,
 label TEXT,
 confidence REAL,
 timestamp TEXT
)
""")
conn.commit()

# --- Simple Centroid Tracker ---
class CentroidTracker:
    def __init__(self, max_disappeared=10, dist_thresh=50):
        self.next_id = 0
        self.objects = dict()  # id -> centroid (x,y)
        self.disappeared = dict()  # id -> frames disappeared
        self.max_disappeared = max_disappeared
        self.dist_thresh = dist_thresh
        # track last logged time for cooldown per id
        self.last_logged = collections.defaultdict(lambda: 0.0)

    def register(self, centroid):
        tid = self.next_id
        self.objects[tid] = centroid
        self.disappeared[tid] = 0
        self.next_id += 1
        return tid

    def deregister(self, tid):
        if tid in self.objects:
            del self.objects[tid]
        if tid in self.disappeared:
            del self.disappeared[tid]
        if tid in self.last_logged:
            del self.last_logged[tid]

    def update(self, rects):
        """
        rects: list of bounding boxes [(x1,y1,x2,y2), ...]
        returns dict of id -> bbox
        """
        if len(rects) == 0:
            # increment disappeared for all
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self.deregister(tid)
            return {}

        # compute input centroids
        input_centroids = []
        for (x1,y1,x2,y2) in rects:
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids.append((cx, cy))

        # if no existing objects, register all
        if len(self.objects) == 0:
            assigned = {}
            for i, cent in enumerate(input_centroids):
                tid = self.register(cent)
                assigned[tid] = rects[i]
            return assigned

        # build distance matrix between existing object centroids and input centroids
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid] for oid in object_ids]
        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(ic))

        # greedy assignment: for each smallest distance pair assign if below threshold
        assigned = {}
        used_rows = set()
        used_cols = set()
        rows = list(range(D.shape[0]))
        cols = list(range(D.shape[1]))
        # repeatedly pick min
        while True:
            if D.size == 0:
                break
            idx = np.unravel_index(np.argmin(D, axis=None), D.shape)
            r, cidx = idx
            if r in used_rows or cidx in used_cols:
                D[r, cidx] = np.inf
                if np.isinf(D).all():
                    break
                continue
            if D[r, cidx] > self.dist_thresh:
                # too far, don't match
                D[r, cidx] = np.inf
                if np.isinf(D).all():
                    break
                continue
            # assign
            oid = object_ids[r]
            self.objects[oid] = input_centroids[cidx]
            self.disappeared[oid] = 0
            assigned[oid] = rects[cidx]
            used_rows.add(r)
            used_cols.add(cidx)
            # mark row/col as used by setting distances to inf
            D[r, :] = np.inf
            D[:, cidx] = np.inf
            if np.isinf(D).all():
                break

        # any unassigned input cols -> register new objects
        for col in range(len(input_centroids)):
            if col not in used_cols:
                tid = self.register(input_centroids[col])
                assigned[tid] = rects[col]

        # any unassigned rows -> increment disappeared
        for row in range(len(object_centroids)):
            if row not in used_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        return assigned

# --- Load model ---
print("Loading model:", MODEL_NAME)
model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Error: cannot open video source:", VIDEO_SOURCE)
    exit(1)

tracker = CentroidTracker(max_disappeared=MAX_DISAPPEARED, dist_thresh=DISTANCE_THRESHOLD)

print("Starting tracked detection. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame. Exiting.")
            break

        results = model.predict(source=frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)
        r = results[0]
        bboxes = []
        labels = []
        confs = []
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = r.names[cls_id] if hasattr(r, "names") else str(cls_id)
                bboxes.append((x1, y1, x2, y2))
                labels.append(label)
                confs.append(conf)

        # update tracker with current bboxes
        assigned = tracker.update(bboxes)  # dict: tid -> bbox

        # build annotated frame and conditional logging
        annotated = frame.copy()
        # assigned is matched and includes newly registered ones
        for tid, bbox in assigned.items():
            # find index in bboxes to get label/conf (match by bbox equality)
            try:
                idx = bboxes.index(bbox)
            except ValueError:
                # fallback: simply use first label
                idx = 0 if labels else None

            label = labels[idx] if idx is not None and idx < len(labels) else "obj"
            conf = confs[idx] if idx is not None and idx < len(confs) else 0.0
            x1, y1, x2, y2 = bbox
            # draw box + ID
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f"ID:{tid} {label} {conf:.2f}", (x1, max(y1-12,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # conditional logging: only log if never logged before or cooldown passed
            now = time.time()
            last = tracker.last_logged.get(tid, 0.0)
            if now - last > LOG_COOLDOWN_SEC:
                tracker.last_logged[tid] = now
                timestamp = datetime.utcnow().isoformat(timespec='seconds') + "Z"
                c.execute("INSERT INTO detections (track_id, label, confidence, timestamp) VALUES (?, ?, ?, ?)",
                          (int(tid), label, conf, timestamp))
                conn.commit()
                print(f"[LOG] {timestamp} - ID {tid} - {label} {conf:.2f}")

        # Show annotated frame
        if SHOW_WINDOW:
            cv2.imshow("SkyGuard - Tracked Detection", annotated)
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
