# =======================================================
# IMPORTS
# =======================================================
import sqlite3
import os
import cv2
import time
import numpy as np
import supervision as sv
from ultralytics import YOLO
import easyocr
import threading
import asyncio
import logging
import re
import requests
from collections import deque, defaultdict
from typing import List, Dict

# FastAPI & Starlette
from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

# Local project files
from . import auth
from scripts import config
from scripts import challan_generator
from scripts.database_manager import DatabaseManager

# =======================================================
# GLOBAL SETUP FOR VIDEO PROCESSING
# =======================================================
output_frames = defaultdict(bytes)
lock = threading.Lock()

# =======================================================
# UTILITY FUNCTIONS & CLASSES (from main.py)
# =======================================================
def clean_plate_text(text: str) -> str:
    """Removes non-alphanumeric characters from the plate text."""
    return re.sub(r'[^A-Z0-9]', '', text).upper()

class ViewTransform:
    """Handles perspective transformation."""
    def __init__(self, source_arr: np.ndarray, target_arr: np.ndarray) -> None:
        source = source_arr.astype(np.float32)
        target = target_arr.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0: return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
    
# --- EDITED: Image Preprocessing Function for OCR with Upscaling ---
def preprocess_plate_for_ocr(plate_image: np.ndarray) -> np.ndarray:
    """
    Applies a series of preprocessing steps to a license plate image to improve OCR accuracy.
    """
    # Get original dimensions. If the image is too small, skip processing.
    if plate_image.shape[0] < 10 or plate_image.shape[1] < 10:
        return plate_image # Return original if too small to avoid errors

    # 1. --- NEW: UPSCALE THE IMAGE ---
    # This is the most critical step for low-resolution plates.
    scaling_factor = 4
    width = int(plate_image.shape[1] * scaling_factor)
    height = int(plate_image.shape[0] * scaling_factor)
    upscaled_plate = cv2.resize(plate_image, (width, height), interpolation=cv2.INTER_CUBIC)

    # 2. Convert the upscaled image to grayscale
    gray_plate = cv2.cvtColor(upscaled_plate, cv2.COLOR_BGR2GRAY)

    # 3. Apply contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_plate = clahe.apply(gray_plate)

    # 4. Apply a bilateral filter for noise reduction while keeping edges sharp
    denoised_plate = cv2.bilateralFilter(enhanced_plate, 9, 75, 75)
    
    # 5. Apply adaptive thresholding to get a clean binary image
    binary_plate = cv2.adaptiveThreshold(
        denoised_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return binary_plate

# =======================================================
# BACKGROUND VIDEO PROCESSING THREAD
# =======================================================
# (The rest of the file is unchanged)
def video_processing_thread(cam_id: str, source: str):
    """
    This function contains the full processing logic for a single camera.
    """
    global output_frames, lock
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logging.error(f"Cannot open video source for camera '{cam_id}': {source}")
            return

        vehicle_model = YOLO(config.VEHICLE_MODEL_PATH, task='detect')
        anpr_model = YOLO(config.ANPR_MODEL_PATH, task='detect')
        reader = easyocr.Reader(['en'])
        db_manager = DatabaseManager(config.DATABASE_CONFIG["db_name"])
        
        PLATE_CHAR_ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        stolen_plates_file_path = os.path.join(config.BASE_DIR, "stolen_plates.txt")
        stolen_log_path = os.path.join(config.BASE_DIR, "stolen_vehicle_alerts.log") 
        try:
            with open(stolen_plates_file_path, 'r') as f:
                stolen_plates = {line.strip().upper() for line in f}
        except FileNotFoundError:
            logging.warning(f"Stolen plates file '{stolen_plates_file_path}' not found.")
            stolen_plates = set()
        
        vehicle_names = ['car', 'truck', 'bus', 'motorcycle']
        class_names = vehicle_model.names
        vehicle_class_ids = [k for k, v in class_names.items() if v in vehicle_names]

        camera_data = {
            'fps': cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS,
            'tracker': sv.ByteTrack(),
            'zone': sv.PolygonZone(polygon=config.SOURCE_MATRICES[cam_id]),
            'previous_ids': set(),
            'has_violated': {},
            'overspeed_logged': {},
            'coordinates': defaultdict(lambda: deque(maxlen=int(cap.get(cv2.CAP_PROP_FPS) or 30))),
            'speed_history': defaultdict(lambda: deque(maxlen=5)),
        }
        
        TARGET_WIDTH, TARGET_HEIGHT = 25, 250
        TARGET = np.array([[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]])
        view_transformer = ViewTransform(source_arr=config.SOURCE_MATRICES[cam_id], target_arr=TARGET)
        
        roi_points = config.SOURCE_MATRICES[cam_id]
        bottom_points = sorted(roi_points, key=lambda p: p[1], reverse=True)[:2]
        capture_trigger_y = (bottom_points[0][1] + bottom_points[1][1]) / 2 - 10
        logging.info(f"[{cam_id}] Evidence capture trigger line set at y={capture_trigger_y}")

    except Exception as e:
        logging.error(f"Error initializing resources for camera '{cam_id}': {e}")
        return

    pTime = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"End of video stream for camera '{cam_id}', restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(1)
            continue

        frame_count += 1
        
        if frame_count % config.FRAME_SKIP == 0:
            frame = cv2.resize(frame, (1280, 720))
            
            result = vehicle_model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[np.isin(detections.class_id, vehicle_class_ids)]
            detections = detections[camera_data['zone'].trigger(detections)]
            tracked_detections = camera_data['tracker'].update_with_detections(detections=detections)

            current_ids = set(tracked_detections.tracker_id)
            exited_ids = camera_data['previous_ids'] - current_ids
            for track_id in exited_ids:
                for data_dict in ['has_violated', 'overspeed_logged', 'coordinates', 'speed_history']:
                    camera_data.get(data_dict, {}).pop(track_id, None)
            camera_data['previous_ids'] = current_ids

            for bbox, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
                if track_id is None:
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                
                bottom_center_point = np.array([[int((x1+x2)/2), y2]])
                transformed_point = view_transformer.transform_points(points=bottom_center_point)[0]
                py = transformed_point[1]
                camera_data['coordinates'][track_id].append(py)

                smooth_speed = 0
                fps = camera_data['fps']
                if len(camera_data['coordinates'][track_id]) >= fps * 0.5:
                    y_start = camera_data['coordinates'][track_id][0]
                    y_end = camera_data['coordinates'][track_id][-1]
                    dy_pixels = abs(y_end - y_start)
                    elapsed_time = len(camera_data['coordinates'][track_id]) / fps
                    
                    if elapsed_time > 0:
                        dy_meters = dy_pixels * config.DISTANCE_PER_PIXEL_METERS.get(cam_id, 0.4)
                        speed = (dy_meters / elapsed_time) * 3.6
                        camera_data['speed_history'][track_id].append(speed)
                        smooth_speed = abs(sum(camera_data['speed_history'][track_id]) / len(camera_data['speed_history'][track_id]))

                if smooth_speed > config.SPEED_LIMITS.get(cam_id, 999):
                    camera_data['has_violated'][track_id] = True

                if y2 >= capture_trigger_y and not camera_data['overspeed_logged'].get(track_id):
                    if camera_data['has_violated'].get(track_id):
                        
                        vehicle_crop = frame[y1:y2, x1:x2]
                        plate_text, plate_crop = "UNKNOWN", None
                        if vehicle_crop.size > 0:
                            try:
                                anpr_results = anpr_model(vehicle_crop, verbose=False)[0]
                                anpr_detections = sv.Detections.from_ultralytics(anpr_results)
                                if len(anpr_detections) > 0:
                                    plate_xyxy = anpr_detections.xyxy[0]
                                    px1, py1, px2, py2 = map(int, plate_xyxy)
                                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                                    
                                    if plate_crop.size > 0:
                                        processed_plate = preprocess_plate_for_ocr(plate_crop)
                                        
                                        ocr_result = reader.readtext(
                                            processed_plate, 
                                            allowlist=PLATE_CHAR_ALLOWLIST
                                        )
                                        
                                        if len(ocr_result) > 0:
                                            plate_text = clean_plate_text(ocr_result[0][1])

                                            # =======================================================
                                            # --- ADDED: STOLEN VEHICLE DETECTION LOGIC ---
                                            # =======================================================
                                            if plate_text and plate_text in stolen_plates:
                                                logging.critical(f"🚨 STOLEN VEHICLE DETECTED! Plate: {plate_text}, Camera: {cam_id}")
                                                try:
                                                    with open(stolen_log_path, 'a') as log_file:
                                                        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: STOLEN VEHICLE ALERT! Plate: {plate_text} seen on Camera: {cam_id}\n")
                                                except Exception as e:
                                                    logging.error(f"Failed to write to stolen vehicle log: {e}")
                                            # =======================================================
                                                                                            # 2. Send real-time notification for stolen vehicle
                                                try:
                                                    # We need a timestamp for the image filename
                                                    temp_timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                                                    vehicle_image_filename_for_alert = f"vehicle_{cam_id}_{track_id}_{temp_timestamp}.jpg"
                                                    alert_payload = {
                                                        "type": "stolen_vehicle_alert",
                                                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                                                        "camera_id": cam_id,
                                                        "plate_number": plate_text,
                                                        "vehicle_image_url": f"/files/vehicles/{vehicle_image_filename_for_alert}"
                                                    }
                                                    requests.post("http://127.0.0.1:8000/api/notify-new-event", json=alert_payload, timeout=2)
                                                    logging.info(f"Sent real-time notification for stolen vehicle: {plate_text}")
                                                except requests.exceptions.RequestException as e:
                                                    logging.error(f"Could not send real-time stolen vehicle notification: {e}")
                                            
                            except Exception as e:
                                logging.error(f"ANPR Error during capture for ID {track_id}: {e}")

                        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                        current_smooth_speed = int(smooth_speed)
                        
                        vehicle_image_filename = f"vehicle_{cam_id}_{track_id}_{timestamp}.jpg"
                        plate_image_filename = f"plate_{cam_id}_{plate_text}_{timestamp}.jpg"
                        vehicle_image_path = os.path.join(config.VEHICLE_DIR, vehicle_image_filename)
                        plate_image_path = os.path.join(config.ANPR_DIR, plate_image_filename)

                        snapshot_frame = frame.copy()
                        cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.imwrite(vehicle_image_path, snapshot_frame)

                        if plate_crop is not None and plate_crop.size > 0:
                            cv2.imwrite(plate_image_path, plate_crop)
                        else:
                            plate_image_path = ""

                        challan_generator.generate_challan(
                            output_dir=config.CHALLAN_DIR, timestamp=timestamp, camera_id=cam_id,
                            vehicle_id=str(track_id), speed_kmh=current_smooth_speed,
                            plate_number=plate_text, vehicle_image_path=vehicle_image_path,
                            plate_image_path=plate_image_path
                        )
                        
                        db_manager.log_overspeeding_event(
                            timestamp=timestamp, camera_id=cam_id, vehicle_id=track_id, 
                            speed_kmh=current_smooth_speed, plate_number=plate_text
                        )
                        
                        try:
                            event_payload = {
                                "id": "new", "timestamp": timestamp, "camera_id": cam_id,
                                "vehicle_id": int(track_id), "speed_kmh": int(current_smooth_speed),
                                "plate_number": plate_text,
                                "vehicle_image_url": f"/files/vehicles/{vehicle_image_filename}",
                                "plate_image_url": f"/files/number_plates/{plate_image_filename}",
                                "challan_pdf_url": f"/files/challans/challan_{cam_id}_{plate_text}_{timestamp}.pdf"
                            }
                            requests.post("http://127.0.0.1:8000/api/notify-new-event", json=event_payload, timeout=2)
                        except requests.exceptions.RequestException as e:
                            logging.error(f"Could not send real-time notification: {e}")
                        
                        logging.warning(f"Violation processed for ID {track_id} at {current_smooth_speed} km/h. Plate: {plate_text}")

                    camera_data['overspeed_logged'][track_id] = True

                is_violator = camera_data['has_violated'].get(track_id, False)
                bbox_color = (0, 0, 255) if is_violator else (0, 255, 0)
                text = f"ID:{track_id} Speed:{int(smooth_speed)} km/h"
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)
                    
            cv2.polylines(frame, [config.SOURCE_MATRICES[cam_id]], isClosed=True, color=(0, 255, 0), thickness=2)
            cTime = time.time()
            if pTime > 0:
                FPS = 1 / (cTime - pTime)
                cv2.putText(frame, f"FPS:{int(FPS)}", (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255))
            pTime = cTime
            
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if flag:
                with lock:
                    output_frames[cam_id] = encodedImage.tobytes()

# =======================================================
# FASTAPI APP, MIDDLEWARE, AND ENDPOINTS
# =======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=config.OUTPUTS_DIR), name="files")

class ConnectionManager:
    def __init__(self): self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket): self.active_connections.remove(websocket)
    async def broadcast(self, data: Dict):
        for connection in self.active_connections: await connection.send_json(data)
manager = ConnectionManager()

def get_db_connection():
    conn = sqlite3.connect(config.DATABASE_CONFIG["db_name"])
    conn.row_factory = sqlite3.Row 
    return conn

@app.on_event("startup")
async def startup_event():
    logging.info("Starting video processing threads...")
    for cam_id, source in config.CAMERA_FEEDS.items():
        thread = threading.Thread(target=video_processing_thread, args=(cam_id, source), daemon=True)
        thread.start()
        logging.info(f"Started thread for camera: {cam_id}")

class Token(BaseModel):
    access_token: str
    token_type: str
    user_info: dict

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth.get_user(auth.FAKE_USERS_DB, form_data.username)
    if not user or not auth.verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user["username"], "role": user["role"]})
    user_info = {"username": user["username"], "role": user["role"], "full_name": user.get("full_name")}
    return {"access_token": access_token, "token_type": "bearer", "user_info": user_info}

async def generate_video_stream(camera_id: str):
    while True:
        with lock:
            frame = output_frames.get(camera_id)
        if frame:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        await asyncio.sleep(1/30)

@app.get("/api/video_feed/{camera_id}")
async def video_feed(camera_id: str):
    if camera_id not in config.CAMERA_FEEDS:
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(generate_video_stream(camera_id), media_type='multipart/x-mixed-replace; boundary=frame')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/notify-new-event")
async def notify_new_event(event_data: Dict):
    await manager.broadcast(event_data)
    return {"status": "notification sent"}

@app.get("/api/events")
async def get_overspeeding_events(current_user: dict = Depends(auth.get_current_active_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    if current_user["role"] == "admin":
        cursor.execute("SELECT * FROM overspeeding_log ORDER BY id DESC")
    else:
        cursor.execute("SELECT * FROM overspeed_log WHERE plate_number = ? ORDER BY id DESC", (current_user["username"],))
    rows = cursor.fetchall()
    conn.close()
    
    events = []
    for row in rows:
        ts = row['timestamp']
        cam_id = row['camera_id']
        plate = row['plate_number'] or "UNKNOWN"
        vehicle_id = row['vehicle_id']
        event_data = {
            "id": row['id'], "timestamp": ts, "camera_id": cam_id,
            "vehicle_id": vehicle_id, "speed_kmh": row['speed_kmh'],
            "plate_number": plate,
            "vehicle_image_url": f"/files/vehicles/vehicle_{cam_id}_{vehicle_id}_{ts}.jpg",
            "plate_image_url": f"/files/number_plates/plate_{cam_id}_{plate}_{ts}.jpg",
            "challan_pdf_url": f"/files/challans/challan_{cam_id}_{plate}_{ts}.pdf"
        }
        events.append(event_data)
    return events

@app.get("/api/stats")
async def get_stats(current_user: dict = Depends(auth.get_current_active_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM overspeeding_log")
    total_violations = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT plate_number) FROM overspeeding_log WHERE plate_number != 'UNKNOWN'")
    unique_vehicles = cursor.fetchone()[0]
    cursor.execute("SELECT camera_id, COUNT(*) FROM overspeeding_log GROUP BY camera_id")
    violations_by_cam = dict(cursor.fetchall())
    conn.close()
    return {"total_violations": total_violations, "unique_vehicles": unique_vehicles, "violations_by_camera": violations_by_cam}

@app.get("/api/stolen-alerts")
async def get_stolen_alerts(current_user: dict = Depends(auth.get_current_active_user)):
    alerts = []
    stolen_log_path = os.path.join(config.BASE_DIR, "stolen_vehicle_alerts.log")
    if os.path.exists(stolen_log_path):
        with open(stolen_log_path, 'r') as f:
            alerts = [line.strip() for line in f.readlines()][::-1]
    return {"alerts": alerts}