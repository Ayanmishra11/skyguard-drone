# SkyGuard - Autonomous Surveillance Drone

Real-time AI surveillance using YOLOv8 and OpenCV at 28 FPS with automated threat alerting and FastAPI dashboard streaming.

## Overview

Autonomous drone surveillance system performing real-time object detection on live video feed. Classifies threats, triggers alerts on confidence-threshold breach, and streams events to a monitoring dashboard.

## Performance

| Metric | Value |
|---|---|
| Inference Speed | 28 FPS real-time |
| Detection Model | YOLOv8 |
| Alert Trigger | Confidence threshold breach |
| Backend | FastAPI and WebSockets |

## Tech Stack

- YOLOv8 (Ultralytics): Object detection
- OpenCV: Video capture and frame processing
- FastAPI: Backend API
- WebSockets: Real-time event streaming
- Python 3.10+

## Architecture

Drone Camera Feed -> OpenCV Frame Capture -> YOLOv8 Inference -> Confidence Check -> Alert Trigger -> FastAPI Backend -> Dashboard

## How to Run

git clone https://github.com/Ayanmishra11/skyguard-drone
cd skyguard-drone
pip install -r requirements.txt
python detection/detector.py
uvicorn backend.main:app --reload

## Author

Ayan Mishra, B.Tech CSE (AI), BIT Durg
Email: ayanmishra9820@gmail.com
LinkedIn: https://linkedin.com/in/ayan-mishra-971bab299
