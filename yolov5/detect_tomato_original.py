import torch
import cv2
import numpy as np
from utils.general import non_max_suppression
import time

# Shrinking with the same ratio + padding
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]  # The heigh and width of image
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # Divide padding into 2 sides
    dh /= 2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)

# Loading the model
model = torch.jit.load('./weights/yolov5n_tomato.torchscript').eval().to('cuda')
img_tensor_gpu = torch.empty((1, 3, 640, 640), dtype=torch.float32, device='cuda')

# Open the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Reduce the resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Prepogressing
    img, ratio, (dw, dh) = letterbox(frame, new_shape=(640, 640))
    img_rgb = img[:, :, ::-1]  # BGR to RGB
    img_tensor = img_rgb.transpose(2, 0, 1)
    img_tensor = np.ascontiguousarray(img_tensor, dtype=np.float32) / 255.0
    img_cpu_tensor = torch.from_numpy(img_tensor)
    img_tensor_gpu.copy_(img_cpu_tensor.unsqueeze(0))

    # Inference
    with torch.no_grad():
        pred = model(img_tensor_gpu)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres=0.9, iou_thres=0.5)[0]
    class_names = ['Green', 'Turning', 'Harvest']

    # Draw the Bounding box
    if pred is not None:
        for *box, conf, cls in pred.cpu().numpy():
            x1, y1, x2, y2 = box
            # Turn thr bbox coordinate from letterbox to original image size
            x1 = (x1 - dw) / ratio
            y1 = (y1 - dh) / ratio
            x2 = (x2 - dw) / ratio
            y2 = (y2 - dh) / ratio
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f'{class_names[int(cls)]} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Tomato Detection", frame)
    print(f"fps: {round((1/(time.time()-start)),3)} s", end='\r')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
