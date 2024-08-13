import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import geocoder  # for location (latitude and longitude)
from sort.Sort import *  # Assuming you have the SORT implementation
from util import get_car, read_license_plate, write_csv  # Custom utility functions

# Initialize YOLO models
coco_model = YOLO('yolov8n.pt')  # Model for detecting trucks/vehicles
license_plate_detector = YOLO('license_plate_detector.pt')  # Model for detecting license plates

# Initialize SORT tracker
mot_tracker = Sort()

# Define vehicle classes (e.g., car, truck, etc.)
vehicles = [2, 3, 5, 7]

# Initialize variables for tracking
results = {}
prev_position = None
movement_status = "Stationary"

def detect_movement(position):
    global prev_position, movement_status
    if prev_position is None:
        prev_position = position
        return "Stationary"
    if position != prev_position:
        movement_status = "Moving"
    else:
        movement_status = "Stationary"
    prev_position = position
    return movement_status

def get_time_date():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def get_location():
    g = geocoder.ip('me')
    return g.latlng if g else (None, None)

def main():
    st.title("Enhanced Truck Tracking System")

    # Load video
    cap = cv2.VideoCapture('./sample.mp4')

    frame_nmr = -1
    ret = True

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        results[frame_nmr] = {}

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop and process license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    position = ((xcar1 + xcar2) // 2, (ycar1 + ycar2) // 2)
                    movement_status = detect_movement(position)
                    time_date = get_time_date()
                    location = get_location()

                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        },
                        'movement_status': movement_status,
                        'time_date': time_date,
                        'location': location
                    }

                    st.write(f"Frame {frame_nmr}:")
                    st.write(f"Detected Number Plate: {license_plate_text}")
                    st.write(f"Truck Position: {position}")
                    st.write(f"Movement Status: {movement_status}")
                    st.write(f"Time and Date: {time_date}")
                    st.write(f"Location (Lat, Long): {location}")

        # Convert frame for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, channels="RGB")

    cap.release()

    # Write results to CSV
    write_csv(results, './test.csv')

if __name__ == "__main__":
    main()
