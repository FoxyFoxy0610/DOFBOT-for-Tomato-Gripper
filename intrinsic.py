import cv2
import numpy as np
import os
import glob
import time

# ----------- 使用者設定 ------------
CHECKERBOARD = (4, 3)       # 棋盤格交點數
SQUARE_SIZE = 0.04         # 每格大小（單位公尺）
CAMERA_ID = 2
IMAGE_DIR = './calibration_images'
RESULT_FILE = 'calibration_result.npz'
# -----------------------------------

os.makedirs(IMAGE_DIR, exist_ok=True)

# 世界座標點（棋盤在 z=0 平面）
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# 開啟攝影機
cap = cv2.VideoCapture(CAMERA_ID)
print("🔵 按下 scd. 拍照並儲存")
print("🟡 按下 c 執行相機內參校正")
print("🔴 按下 q 離開")

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 攝影機無法啟動")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    show = frame.copy()
    if ret_cb:
        cv2.drawChessboardCorners(show, CHECKERBOARD, corners, ret_cb)

    cv2.imshow("Calibration Camera", show)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(IMAGE_DIR, f"calib_{timestamp}.jpg")
        cv2.imwrite(path, frame)
        print(f"📸 已儲存圖片：{path}")
        frame_id += 1

    elif key == ord('c'):
        print("🟡 開始校正中...")

        objpoints = []
        imgpoints = []
        gray_shape = None

        images = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
        if len(images) < 5:
            print("⚠️ 圖片太少（<5），請多拍幾張棋盤格")
            continue

        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_shape = gray.shape
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        if len(objpoints) < 5:
            print("❌ 校正失敗，成功偵測的圖片太少")
            continue

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray_shape[::-1], None, None)
        
        print("✅ 校正完成")
        print("Camera Matrix:\n", mtx)
        print("Distortion Coeffs:\n", dist)

        np.savez(RESULT_FILE, cameraMatrix=mtx, distCoeffs=dist)
        print(f"📁 結果已儲存於 {RESULT_FILE}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
