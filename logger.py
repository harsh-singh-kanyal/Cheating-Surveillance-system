import cv2
import time
import os
import csv
from datetime import datetime
from eye_movement import process_eye_movement
from head_pose import process_head_pose
from mobile_detection import process_mobile_detection

# Initialize webcam
cap = cv2.VideoCapture("http://192.168.180.244:4747/video")  # update IP if needed

# Create directory for screenshots (if needed later)
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

# CSV logging setup
log_file = "realtime_behavior_log.csv"
fieldnames = ['timestamp', 'head_direction', 'gaze_direction', 'mobile_detected']

# Create CSV file if it doesn't exist
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

# Calibration and counters
calibrated_angles = None
start_time = time.time()
frame_count = 0
max_frames = 10  # limit to 10 logged frames

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Process eye movement
    frame, gaze_direction = process_eye_movement(frame)
    cv2.putText(frame, f"Gaze: {gaze_direction}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Calibration phase (first 5 seconds)
    if time.time() - start_time <= 5:
        cv2.putText(frame, "Calibrating... Keep head straight", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if calibrated_angles is None:
            _, calibrated_angles = process_head_pose(frame, None)
        cv2.imshow("Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # After calibration
    frame, head_direction = process_head_pose(frame, calibrated_angles)
    cv2.putText(frame, f"Head: {head_direction}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    frame, mobile_detected = process_mobile_detection(frame)
    cv2.putText(frame, f"Mobile: {mobile_detected}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save to CSV
    with open(log_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'head_direction': head_direction,
            'gaze_direction': gaze_direction,
            'mobile_detected': mobile_detected
        })

    frame_count += 1

    # Show frame
    cv2.imshow("Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("✅ Data collection complete. 10 frames logged.")
