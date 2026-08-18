import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math
from scipy.optimize import fsolve
from Arm_Lib import Arm_Device
import time

# Mechanical arm segment lengths (unit: cm)
len_1 = 8.5
len_2 = 8.5
len_3 = 17

# Camera offset from the end effector
gripper_camera_angle = 0.392699082 # in radians (approx 22.5 degrees)
gripper_camera_len = 11.18 # in cm

def forward_kinematics(pose):
    """
    Calculate the forward kinematics to find the camera's position and orientation.
    
    Args:
        pose: list containing servo angles, where pose[1:4] are theta1, theta2, theta3
    Returns:
        position: [x, y] coordinates of the camera
        direction: camera orientation angle in radians
    """
    theta1, theta2, theta3 = np.radians(pose[1:4])

    x1 = len_1 * np.cos(theta1)
    y1 = len_1 * np.sin(theta1)

    theta12 = theta1 + theta2 - np.radians(90)
    x2 = x1 + len_2 * np.cos(theta12)
    y2 = y1 + len_2 * np.sin(theta12)

    theta123 = theta12 + theta3 - np.radians(90)
    x3 = x2 + len_3 * np.cos(theta123)
    y3 = y2 + len_3 * np.sin(theta123)

    theta_camera = theta123 - gripper_camera_angle
    camera_x = x3 - gripper_camera_len * np.cos(theta_camera)
    camera_y = y3 - gripper_camera_len * np.sin(theta_camera)

    direction = np.array(theta123)
    position = np.array([camera_x, camera_y])

    return position, direction

def positioning(cam_pose_1, cam_dir_1, cam_pose_2, cam_dir_2):
    """
    Estimate the 2D coordinates of the tomato using triangulation from two camera views.
    
    Args:
        cam_pose_1, cam_pose_2: [x, y] coordinates of the camera in view 1 and 2
        cam_dir_1, cam_dir_2: orientation angle of the camera in view 1 and 2
    Returns:
        [x_tomato, y_tomato]: Estimated 2D coordinates of the tomato
    """
    try:
        x_tomato = (cam_pose_2[1] - cam_pose_1[1] + np.tan(cam_dir_1)*cam_pose_1[0] - np.tan(cam_dir_2)*cam_pose_2[0]) / (np.tan(cam_dir_1) - np.tan(cam_dir_2))
        y_tomato = np.tan(cam_dir_1) * (x_tomato - cam_pose_1[0]) + cam_pose_1[1]

        return [x_tomato, y_tomato]
    except np.linalg.LinAlgError:
        print("Two views are parallel with no intersection!")
        return None

def estimate_size(cam_1, cam_2, tomato, width1_px, width2_px):
    """
    Estimate the physical width of the tomato using the pinhole camera model.
    
    Args:
        cam_1, cam_2: [x, y] coordinates of the camera in view 1 and 2
        tomato: [x, y] coordinates of the tomato
        width1_px, width2_px: bounding box width of the tomato in pixels from view 1 and 2
    Returns:
        width_real: Estimated physical width of the tomato (in cm)
    """
    npz_file = 'calibration_matrix.npz'
    data = np.load(npz_file)
    camera_matrix = data['cameraMatrix']
    
    # Focal length in x direction
    fx = camera_matrix[0, 0]

    # Euclidean distance from camera to tomato for both views
    D1 = math.sqrt((cam_1[0] - tomato[0])**2 + (cam_1[1] - tomato[1])**2)
    D2 = math.sqrt((cam_2[0] - tomato[0])**2 + (cam_2[1] - tomato[1])**2)

    # Convert pixel width to real-world width using similar triangles
    width1_real = width1_px * D1 / fx
    width2_real = width2_px * D2 / fx
    
    # Average the estimated width from both views for better accuracy
    width_real =  (width1_real + width2_real) / 2

    return width_real

def inverse_kinematic(tomato_x, tomato_y, phi=20):
    """
    Calculate the required servo angles (Inverse Kinematics) to reach the tomato.
    
    Args:
        tomato_x, tomato_y: Target coordinates of the tomato
        phi: Desired grasping angle of the end effector (default is 20 degrees to avoid fruit occlusion)
    Returns:
        theta1_servo, theta2_servo, theta3_servo: Angles for servos 2, 3, and 4
    """
    phi_rad = np.radians(phi)

    # Target position for the wrist joint (L3 connection)
    x2 = tomato_x - len_3 * np.cos(phi_rad)
    y2 = tomato_y + len_3 * np.sin(phi_rad)

    def equations(deg):
        theta1_deg, theta2_deg = deg
        theta1 = np.radians(theta1_deg)
        theta2 = np.radians(theta2_deg)

        eq1 = len_2 * np.cos(theta1 - theta2) + len_1 * np.cos(theta1) - x2
        eq2 = len_2 * np.sin(theta1 - theta2) + len_1 * np.sin(theta1) - y2
    
        return [eq1, eq2]

    # Solve the non-linear equations using fsolve
    initial_guess = [0, 0]
    theta1_deg, theta2_deg = fsolve(equations, initial_guess)
    
    # Transform solved angles to DOFBOT servo frame (0-180 degrees)
    theta1_servo = theta1_deg
    theta2_servo = 90 - theta2_deg
    theta3_servo = 90 - phi - (theta1_deg - theta2_deg)

    return theta1_servo, theta2_servo, theta3_servo

def gripping(sevo_1_angle, theta1, theta2, theta3, tomato_width):
    """
    Execute the gripping sequence using the calculated angles and tomato width.
    
    Args:
        sevo_1_angle: Angle for the base servo (Servo 1)
        theta1, theta2, theta3: Angles for joints (Servos 2, 3, 4)
        tomato_width: Estimated physical width of the tomato
    """
    # Calibration data: Maps Gripper Opening Width to Servo Motor Angle
    # data[:, 0] -> Gripper opening width (mm or relative unit)
    # data[:, 1] -> Servo motor angle (0-180 degrees, where 180 is fully closed)
    data = np.array([[62, 30], [61, 45], [58, 60], [56, 75], [51, 90],
                     [46, 105], [39, 120], [31, 135], [22, 150], [165, 14], [2, 180]])

    x = data[:, 0].reshape(-1, 1) # Gripper physical width
    y = data[:, 1]                # Corresponding servo angle

    # Fit a 3rd degree polynomial regression model to predict servo angle
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(x)

    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict the required servo angle based on the tomato's width
    x_input = np.array([[tomato_width]])
    x_input_poly = poly.transform(x_input)
    y_pred = model.predict(x_input_poly)

    Arm = Arm_Device()
    
    # Move arm to the target tomato position (with open gripper)
    Arm.Arm_serial_servo_write6(sevo_1_angle, theta1, theta2, theta3, 90, 90, 1000)
    time.sleep(1)
    
    # Close gripper to the predicted angle
    Arm.Arm_serial_servo_write(6, y_pred, 1000)
    time.sleep(1)
    
    # Lift the arm up slightly (harvesting motion)
    Arm.Arm_serial_servo_write(3, theta2 + 10, 1000)
    time.sleep(1)
    
    # Rotate base to drop area
    Arm.Arm_serial_servo_write(1, sevo_1_angle - 30, 1000)
    time.sleep(1)
    
    # Lower the arm
    Arm.Arm_serial_servo_write(3, theta2 - 30, 1000)
    time.sleep(1)
    
    # Open gripper to release tomato
    Arm.Arm_serial_servo_write(6, 90, 1000)
    time.sleep(1)
    
    # Reset arm posture
    Arm.Arm_serial_servo_write6(sevo_1_angle - 30, theta1, theta2, theta3, 90, 90, 1000)