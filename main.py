import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))
import yolov5.detect_tomato as detect_tomato
import cv2
import torch
import time
import numpy as np
import gripper
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from Arm_Lib import Arm_Device

# Initialize tracker for the first view
tracker = detect_tomato.TomatoTracker(1)

# Phase 1: First Tracking and Positioning
print("Starting First Tracking Phase...")
while True:
    pred = tracker.run().cpu()
    if pred is not None and len(pred) > 0:
        # Select the tomato with the highest confidence
        max_conf_idx = torch.argmax(pred[:, 4])
        max_pred = pred[max_conf_idx].numpy()
        x1, y1, x2, y2, conf, cls = max_pred
        
        # Track and align the arm to center the tomato
        first_data = tracker.tracking(x1, x2, y1, y2)

        if first_data is not None:
            print(f"First tracking locked. Data: {first_data}")
            break
    else:
        print("Searching for tomato...", end='\r')

    # Press 'q' to quit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tracker.release()

# Phase 2: Second Tracking and Positioning from a different angle
print("\nStarting Second Tracking Phase...")
# Reinitialize tracker with an offset base angle (second view)
tracker = detect_tomato.TomatoTracker(2, first_data[0])

while True:
    pred = tracker.run().cpu()
    if pred is not None and len(pred) > 0:
        # Select the tomato with the highest confidence
        max_conf_idx = torch.argmax(pred[:, 4])
        max_pred = pred[max_conf_idx].numpy()
        x1, y1, x2, y2, conf, cls = max_pred
        
        # Track and align the arm to center the tomato
        second_data = tracker.tracking(x1, x2, y1, y2)

        if second_data is not None:
            print(f"Second tracking locked. Data: {second_data}")
            break
    else:
        print("Searching for tomato...", end='\r')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tracker.release()

# Phase 3: Data Processing and Harvesting
tracking_pose = np.array([first_data, second_data])

# Calculate camera positions and orientations using Forward Kinematics
cam_pose_1, cam_dir_1 = gripper.forward_kinematics(tracking_pose[0])
cam_pose_2, cam_dir_2 = gripper.forward_kinematics(tracking_pose[1])

print("\n--- Calculations ---")
print(f"Camera 1 Position: {cam_pose_1}, Direction: {cam_dir_1}")
print(f"Camera 2 Position: {cam_pose_2}, Direction: {cam_dir_2}")

# Triangulate to find the actual 2D coordinates of the tomato
object_coordinate = gripper.positioning(cam_pose_1, cam_dir_1, cam_pose_2, cam_dir_2)
print(f"Tomato Coordinates (World): {object_coordinate}")

# Estimate the physical width of the tomato using the bounding box and camera distance
tomato_width = gripper.estimate_size(cam_pose_1, cam_pose_2, object_coordinate, first_data[6], second_data[6])
print(f"Estimated Tomato Width: {tomato_width:.2f}")

# Calculate Inverse Kinematics to find required joint angles
# An offset of 0.5 is added to Y coordinate for better gripping clearance
# Grasping angle phi is set to 15 degrees
theta1, theta2, theta3 = gripper.inverse_kinematic(object_coordinate[0], object_coordinate[1] + 0.5, 15)
print(f"Target Servo Angles (theta1, theta2, theta3): {theta1:.2f}, {theta2:.2f}, {theta3:.2f}")

# Execute the harvesting action
print("Executing Gripping Sequence...")
# A small buffer (0.2) is added to the width to avoid crushing the tomato
gripper.gripping(second_data[0], theta1, theta2, theta3, tomato_width + 0.2)
print("Harvesting complete.")
