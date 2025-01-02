import cv2
import time
import numpy as np

# 初始化相機
camera = cv2.VideoCapture(0)  # 使用第一個相機
if not camera.isOpened():
    print("無法開啟相機")
    exit()

# 定義參數
previous_frame = None  # 用於存儲上一張照片
interval = 20  # 設定拍照間隔（秒）
last_time = time.time()  # 上次拍照的時間

# 初始化背景建模器（使用cv2.createBackgroundSubtractorMOG2）
background_subtractor = cv2.createBackgroundSubtractorMOG2()

try:
    while True:
        # 拍攝當前照片
        ret, current_frame = camera.read()
        if not ret:
            print("無法讀取影像")
            break

        # 調整影像大小（可選）
        current_frame = cv2.resize(current_frame, (640, 480))

        # 顯示即時畫面
        cv2.imshow("Live Feed", current_frame)

        # 檢查是否該進行照片比對
        if time.time() - last_time >= interval:
            # 第一個 20 秒僅保存影像
            if previous_frame is None:
                previous_frame = current_frame
                print("第一張照片已拍攝")
            else:
                # 背景建模（去除背景中的光影變化）
                fg_mask = background_subtractor.apply(current_frame)

                # 進行高斯模糊處理，減少雜訊
                blurred = cv2.GaussianBlur(fg_mask, (5, 5), 0)

                # 二值化處理
                _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

                # 去噪處理
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                thresh = cv2.dilate(thresh, kernel, iterations=2)
                thresh = cv2.erode(thresh, kernel, iterations=1)

                # 找輪廓
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 過濾小區域，避免噪點影響
                min_area = 100  # 設定最小區域範圍，根據需要調整
                contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

                # 畫出Bounding Box
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 顯示處理後的結果
            cv2.imshow("Detected Changes", current_frame)

            # 更新上一張照片
            previous_frame = current_frame.copy()
            last_time = time.time()  # 更新拍照時間

        # 按下 'q' 退出循環
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("程序被中斷")

finally:
    # 釋放資源
    camera.release()
    cv2.destroyAllWindows()
