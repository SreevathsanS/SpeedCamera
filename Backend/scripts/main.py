# main.py
import sqlite3
import cv2
from ultralytics import YOLO
import time
import supervision as sv
import numpy as np
import os
from collections import deque, defaultdict
from pathlib import Path
import easyocr
import logging
import sys
import re  # <-- Added
import requests  # <-- Added
import config as config
from database_manager import DatabaseManager
import multiprocessing
import challan_generator

# ====================================================================
# UTILITY FUNCTIONS & CLASSES
# ====================================================================

def clean_plate_text(text: str) -> str:
    """Removes non-alphanumeric characters from the plate text."""
    return re.sub(r'[^A-Z0-9]', '', text).upper()

class ViewTransform:
    """
    A class to handle perspective transformation for converting
    image coordinates to a bird's-eye view.
    """
    def __init__(self, source_arr: np.ndarray, target_arr: np.ndarray) -> None:
        source = source_arr.astype(np.float32)
        target = target_arr.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms a set of points using the pre-calculated perspective matrix.
        """
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# ====================================================================
# CAMERA PROCESSING FUNCTION
# ====================================================================

def process_camera(cam_id: str, source: str):
    """
    This function contains the full processing logic for a single camera.
    It runs as a separate process to handle real-time monitoring.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source for camera '{cam_id}': {source}")

        vehicle_model = YOLO(config.VEHICLE_MODEL_PATH, task='detect')
        anpr_model = YOLO(config.ANPR_MODEL_PATH, task='detect')
        reader = easyocr.Reader(['en'])
        db_manager = DatabaseManager(config.DATABASE_CONFIG["db_name"])

        stolen_plates_file_path = os.path.join(config.BASE_DIR, "stolen_plates.txt")
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
            'cap': cap,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'tracker': sv.ByteTrack(),
            'zone': sv.PolygonZone(polygon=config.SOURCE_MATRICES[cam_id]),
            'previous_ids': set(),
            'overspeed_detected': {},
            'plates_detected': {},
            'overspeed_logged': {},
            'coordinates': defaultdict(lambda: deque(maxlen=int(cap.get(cv2.CAP_PROP_FPS)))),
            'speed_history': defaultdict(lambda: deque(maxlen=5)),
            'trajectories': defaultdict(list)
        }
        
        TARGET_WIDTH, TARGET_HEIGHT = 25, 250
        TARGET = np.array([[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]])
        view_transformer = ViewTransform(source_arr=config.SOURCE_MATRICES[cam_id], target_arr=TARGET)
        
    except (IOError, KeyError, Exception) as e:
        logging.error(f"Error initializing resources for camera '{cam_id}': {e}")
        return

    pTime = 0
    frame_count = 0
    
    while True:
        ret, frame = camera_data['cap'].read()
        if not ret:
            logging.info(f"End of video stream for camera '{cam_id}'.")
            break

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
                logging.info(f"Vehicle ID {track_id} from {cam_id} has exited the ROI. Cleaning up data.")
                camera_data['overspeed_detected'].pop(track_id, None)
                camera_data['plates_detected'].pop(track_id, None)
                camera_data['overspeed_logged'].pop(track_id, None)
                camera_data['coordinates'].pop(track_id, None)
                camera_data['speed_history'].pop(track_id, None)
                camera_data['trajectories'].pop(track_id, None)
            camera_data['previous_ids'] = current_ids

            for bbox, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
                x1, y1, x2, y2 = map(int, bbox)
                
                bottom_center_point = np.array([[int((x1+x2)/2), y2]])
                transformed_point = view_transformer.transform_points(points=bottom_center_point)[0]
                py = transformed_point[1]
                
                camera_data['coordinates'][track_id].append(py)

                smooth_speed = 0
                if camera_data['fps'] > 0 and len(camera_data['coordinates'][track_id]) >= camera_data['fps'] * 0.5:
                    y_start = camera_data['coordinates'][track_id][0]
                    y_end = camera_data['coordinates'][track_id][-1]
                    dy_pixels = abs(y_end - y_start)
                    elapsed_time = len(camera_data['coordinates'][track_id]) / camera_data['fps']
                    
                    if elapsed_time > 0:
                        dy_meters = dy_pixels * config.DISTANCE_PER_PIXEL_METERS.get(cam_id, 0.4)
                        speed = (dy_meters / elapsed_time) * 3.6
                        camera_data['speed_history'][track_id].append(speed)
                        smooth_speed = abs(sum(camera_data['speed_history'][track_id]) / len(camera_data['speed_history'][track_id]))

                speed_limit_for_cam = config.SPEED_LIMITS.get(cam_id, 170)

                if not camera_data['plates_detected'].get(track_id):
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
                                    ocr_result = reader.readtext(plate_crop)
                                    if len(ocr_result) > 0:
                                        plate_text = clean_plate_text(ocr_result[0][1])
                        except Exception as e:
                            logging.error(f"Error during ANPR for Cam {cam_id}, ID {track_id}: {e}")
                    camera_data['plates_detected'][track_id] = {"text": plate_text, "crop": plate_crop}

                plate_data = camera_data['plates_detected'].get(track_id)
                plate_text = plate_data['text'] if plate_data else "UNKNOWN"
                
                is_stolen = plate_text != "UNKNOWN" and plate_text in stolen_plates
                is_overspeeding = smooth_speed > speed_limit_for_cam
                
                bbox_color = (0, 255, 0)
                text = f"ID:{track_id} Speed:{int(smooth_speed)} km/h"
                if plate_data and plate_text != "UNKNOWN":
                    text += f" Plate:{plate_text}"

                if is_stolen:
                    bbox_color = (0, 0, 255)
                    text = f"STOLEN: {plate_text}"
                    logging.critical(f"STOLEN VEHICLE DETECTED! Plate: {plate_text} at Camera {cam_id}!")
                    
                    if not camera_data['overspeed_logged'].get(track_id):
                        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                        vehicle_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        if vehicle_crop.size > 0:
                            vehicle_path = os.path.join(config.VEHICLE_DIR, f"stolen_vehicle_{cam_id}_{track_id}_{timestamp}.jpg")
                            cv2.imwrite(vehicle_path, vehicle_crop)
                        
                        plate_crop = plate_data['crop'] if plate_data else None
                        if plate_data and plate_crop is not None and plate_crop.size > 0:
                            plate_path = os.path.join(config.ANPR_DIR, f"stolen_plate_{cam_id}_{plate_text}_{timestamp}.jpg")
                            cv2.imwrite(plate_path, plate_crop)

                        stolen_log_path = os.path.join(config.BASE_DIR, "stolen_vehicle_alerts.log")
                        with open(stolen_log_path, "a") as log_file:
                            log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STOLEN VEHICLE ALERT: Plate '{plate_text}' detected by '{cam_id}'\n")

                        camera_data['overspeed_logged'][track_id] = True
                        
                elif is_overspeeding and not camera_data['overspeed_logged'].get(track_id):
                    bbox_color = (0, 0, 255)
                    
                    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                    current_smooth_speed = int(smooth_speed)

                    vehicle_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    vehicle_path = None
                    if vehicle_crop.size > 0:
                        vehicle_path = os.path.join(config.VEHICLE_DIR, f"vehicle_{cam_id}_{track_id}_{timestamp}.jpg")
                        cv2.imwrite(vehicle_path, vehicle_crop)
                        logging.info(f"Saved overspeeding vehicle image for Cam {cam_id}, ID {track_id}")

                    plate_crop = plate_data['crop'] if plate_data else None
                    plate_path = None
                    if plate_data and plate_crop is not None and plate_crop.size > 0:
                        plate_path = os.path.join(config.ANPR_DIR, f"plate_{cam_id}_{plate_text}_{timestamp}.jpg")
                        cv2.imwrite(plate_path, plate_crop)
                    
                    db_manager.log_overspeeding_event(
                        timestamp=timestamp, 
                        camera_id=cam_id, 
                        vehicle_id=track_id, 
                        speed_kmh=current_smooth_speed, 
                        plate_number=plate_text
                    )

                    if vehicle_path and plate_path:
                        challan_generator.generate_challan(
                            output_dir=config.CHALLAN_DIR, timestamp=timestamp, camera_id=cam_id,
                            vehicle_id=track_id, speed_kmh=current_smooth_speed, plate_number=plate_text,
                            vehicle_image_path=vehicle_path, plate_image_path=plate_path
                        )
                        logging.info(f"Generated challan for vehicle {track_id} from {cam_id}")

                    # --- SEND REAL-TIME NOTIFICATION TO FASTAPI SERVER ---
                    try:
                        event_payload = {
                            "id": "new",
                            "timestamp": timestamp,
                            "camera_id": cam_id,
                            "vehicle_id": int(track_id),
                            "speed_kmh": int(current_smooth_speed),
                            "plate_number": plate_text,
                            "vehicle_image_url": f"/files/vehicles/vehicle_{cam_id}_{track_id}_{timestamp}.jpg",
                            "plate_image_url": f"/files/number_plates/plate_{cam_id}_{plate_text}_{timestamp}.jpg",
                            "challan_pdf_url": f"/files/challans/challan_{cam_id}_{plate_text}_{timestamp}.pdf"
                        }
                        requests.post("http://127.0.0.1:8000/api/notify-new-event", json=event_payload, timeout=2)
                        logging.info(f"Sent real-time notification for vehicle {track_id}")
                    except requests.exceptions.RequestException as e:
                        logging.error(f"Could not send real-time notification: {e}")
                    # ---------------------------------------------------------

                    camera_data['overspeed_logged'][track_id] = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)
                    
            cv2.polylines(frame, [config.SOURCE_MATRICES[cam_id]], isClosed=True, color=(0, 255, 0), thickness=2)

            cTime = time.time()
            if pTime > 0:
                FPS = 1 / (cTime - pTime)
                cv2.putText(frame, f"FPS:{int(FPS)}", (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255))
            pTime = cTime
            
            cv2.imshow(f"Video Stream - {cam_id}", frame)
            
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.info("Exiting video stream on user request.")
            break
    
    db_manager.close()
    camera_data['cap'].release()
    cv2.destroyAllWindows()

# ====================================================================
# MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":
    print("DB path at runtime:", config.DATABASE_CONFIG["db_name"])

    processes = []

    try:
        for cam_id, source in config.CAMERA_FEEDS.items():
            print(f"Starting process for camera '{cam_id}'...")
            p = multiprocessing.Process(target=process_camera, args=(cam_id, source))
            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()
            
    except (FileNotFoundError, IOError) as e:
        print(f"An error occurred: {e}")
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()