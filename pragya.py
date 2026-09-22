import cv2
import os
import csv
from datetime import datetime
from eye_movement import process_eye_movement
from head_pose import process_head_pose
from mobile_detection import process_mobile_detection

# Folder containing images
image_folder = "D:\Cheating Surveillance Project\log"  # Put 10 images here
image_files = sorted(os.listdir(image_folder))  # take only 10

# CSV logging
log_file = "realtime_behavior_log.csv"
fieldnames = ['timestamp', 'head_direction', 'gaze_direction', 'mobile_detected']

# Create CSV if not exists
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

calibrated_angles = None
calibration_done = False

for i, img_name in enumerate(image_files):
    image_path = os.path.join(image_folder, img_name)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error reading {image_path}")
        continue

    # Eye detection
    frame, gaze_direction = process_eye_movement(frame)

    # Calibration on first image
    if not calibration_done:
        _, calibrated_angles = process_head_pose(frame, None)
        calibration_done = True

    # Head + mobile
    frame, head_direction = process_head_pose(frame, calibrated_angles)
    frame, mobile_detected = process_mobile_detection(frame)

    # Logging to CSV
    with open(log_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'head_direction': head_direction,
            'gaze_direction': gaze_direction,
            'mobile_detected': mobile_detected
        })

    # Optional: Display frame
    cv2.putText(frame, f"Gaze: {gaze_direction}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Head: {head_direction}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Mobile: {mobile_detected}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Processed Image", frame)
    cv2.waitKey(500)  # wait 0.5 sec between images

cv2.destroyAllWindows()
print("✅ Finished processing 10 local images.")
