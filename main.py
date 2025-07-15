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

tracker = detect_tomato.TomatoTracker(1)

# First Tracking
while True:
    pred = tracker.run().cpu()
    if pred is not None and len(pred) > 0:
        max_conf_idx = torch.argmax(pred[:, 4])
        max_pred = pred[max_conf_idx].numpy()
        x1, y1, x2, y2, conf, cls = max_pred
        first_data = tracker.tracking(x1, x2, y1, y2)

        if first_data is not None:
            print(first_data)
            break
    
    else:
        print("Nothing", end='\r')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

tracker.release()

tracker = detect_tomato.TomatoTracker(2, first_data[0])

# Second Tracking
while True:
    pred = tracker.run().cpu()
    if pred is not None and len(pred) > 0:
        max_conf_idx = torch.argmax(pred[:, 4])
        max_pred = pred[max_conf_idx].numpy()
        x1, y1, x2, y2, conf, cls = max_pred
        second_data = tracker.tracking(x1, x2, y1, y2)

        if second_data is not None:
            print(second_data)
            break
    
    else:
        print("Nothing", end='\r')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


tracking_pose = np.array([first_data, second_data])

cam_pose_1, cam_dir_1 = gripper.forward_kinematics(tracking_pose[0])
cam_pose_2, cam_dir_2 = gripper.forward_kinematics(tracking_pose[1])

print("Camera 1:", cam_pose_1, cam_dir_1)
print("Camera 2:", cam_pose_2, cam_dir_2)

object_coordinate = gripper.positioning(cam_pose_1, cam_dir_1, cam_pose_2, cam_dir_2)
print("Tomato:", object_coordinate)

tomato_width = gripper.estimate_size(cam_pose_1, cam_pose_2, object_coordinate, first_data[6], second_data[6])

print("Width of tomato:", tomato_width)
theta1, theta2, theta3 = gripper.inverse_kinematic(object_coordinate[0], object_coordinate[1]+0.5, 15)
print("theta (deg):", theta1, theta2, theta3)
gripper.gripping(second_data[0], theta1, theta2, theta3, tomato_width+0.2)

tracker.release()
