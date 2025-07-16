import torch
import cv2
import numpy as np
from utils.general import non_max_suppression
import time
import os
from Arm_Lib import Arm_Device

# Shrinking with the same ratio + padding
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]  # height, width
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


class TomatoTracker:
    def __init__(self, track_num, servo_1_init = 90):
        self.track_num = track_num
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        WEIGHT_PATH = os.path.join(BASE_DIR, 'weights', 'yolov5n_tomato.torchscript')
        self.model = torch.jit.load(WEIGHT_PATH).eval().to('cuda')
        self.img_tensor_gpu = torch.empty((1, 3, 640, 640), dtype=torch.float32, device='cuda')

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        self.class_names = ['Green', 'Turning', 'Harvest']

        self.Arm = Arm_Device()
        time.sleep(0.5)
        
        self.angles_initial = [[90, 158, 0, 5, 90, 90],[90, 105, 30, 15, 90, 90]]

        if track_num == 1:
            self.servo_angle_1 = self.angles_initial[0][0]
            self.servo_angle_2 = self.angles_initial[0][1]
            self.servo_angle_3 = self.angles_initial[0][2]
            self.servo_angle_4 = self.angles_initial[0][3]
            self.Arm.Arm_serial_servo_write6(90, 158, 0, 5, 90, 90, 1000)

        elif track_num == 2:
            self.servo_angle_1 = servo_1_init
            self.servo_angle_2 = self.angles_initial[1][1]
            self.servo_angle_3 = self.angles_initial[1][2]
            self.servo_angle_4 = self.angles_initial[1][3]
            self.Arm.Arm_serial_servo_write6(servo_1_init, 105, 30, 15, 90, 90, 1000)

        time.sleep(1)

    def run(self):
        start = time.time()
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Preprocessing
        img, ratio, (dw, dh) = letterbox(frame, new_shape=(640, 640))
        img_rgb = img[:, :, ::-1]
        img_tensor = img_rgb.transpose(2, 0, 1)
        img_tensor = np.ascontiguousarray(img_tensor, dtype=np.float32) / 255.0
        img_cpu_tensor = torch.from_numpy(img_tensor)
        self.img_tensor_gpu.copy_(img_cpu_tensor.unsqueeze(0))

        # Inference
        with torch.no_grad():
            pred = self.model(self.img_tensor_gpu)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres=0.9, iou_thres=0.5)[0]

        # Draw boxes
        if pred is not None:
            for *box, conf, cls in pred.cpu().numpy():
                x1, y1, x2, y2 = box
                x1 = (x1 - dw) / ratio
                y1 = (y1 - dh) / ratio
                x2 = (x2 - dw) / ratio
                y2 = (y2 - dh) / ratio
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f'{self.class_names[int(cls)]} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Tomato Detection", frame)

        return pred

    def tracking(self, x1, x2, y1, y2):
        center_x = (x1 + x2) / 2
        center_y = (2 * y1 + y2) / 3
        if self.track_num == 2:
            center_y = (y1 + 2 * y2) / 3

        image_center = 320
        threshold = 20

        offset_x = center_x - image_center
        offset_y = center_y - image_center
        Kp = 0.005
        Kp_2 = 0.005
        max_step = 1

        if abs(offset_x) < threshold and abs(offset_y) < threshold:
            coordinate = self.angles_initial[self.track_num-1] + [(x2-x1)]
            coordinate[0] = int(self.servo_angle_1)
            coordinate[1] = int(self.servo_angle_2)
            coordinate[2] = int(self.servo_angle_3)
            coordinate[3] = int(self.servo_angle_4)

        else:
            coordinate = None

        delta_angle_x = np.clip(offset_x * Kp, -max_step, max_step)
        delta_angle_y = np.clip(offset_y * Kp, -max_step, max_step)
        # delta_angle_y_2 = np.clip(offset_y * Kp_2, -max_step, max_step)

        new_angle_x = np.clip(self.servo_angle_1 - delta_angle_x, 0, 180)
        # new_angle_y_2 = np.clip(self.servo_angle_2 - delta_angle_y_2, 0, 180)
        new_angle_y_4 = np.clip(self.servo_angle_4 - delta_angle_y, 0, 180)
        

        offset_thre = 0.1
        if abs(new_angle_x - self.servo_angle_1) >= offset_thre:
            self.servo_angle_1 = new_angle_x
            self.Arm.Arm_serial_servo_write(1, self.servo_angle_1, 500)
            time.sleep(0.05)
        
        if abs(new_angle_y_4 - self.servo_angle_4) >= offset_thre:
            # self.servo_angle_2 = new_angle_y_2
            self.servo_angle_4 = new_angle_y_4
            # self.Arm.Arm_serial_servo_write(2, self.servo_angle_2, 500)
            self.Arm.Arm_serial_servo_write(4, self.servo_angle_4, 500)
            time.sleep(0.05)
        
        print(self.servo_angle_1, self.servo_angle_2, self.servo_angle_3, self.servo_angle_4, end='\r')
    
        return coordinate
        

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
