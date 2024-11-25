import cv2
import os
import time
from datetime import datetime

# 設置解析度 (超過會預設相機最大解析度)
resolution_w, resolution_h = 1920, 1080

# 初始化相機
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_h)

# 檢查相機是否成功開啟
if not cap.isOpened():
    print("無法開啟相機")
    exit()

def count_jpg_files(directory):
    if not os.path.isdir(directory):
        print(f"路徑 {directory} 不存在")
        return 0
    jpg_count = len([file for file in os.listdir(directory) if file.lower().endswith('.jpg')])
    print(f"找到 {jpg_count} 個 JPG 檔案在 {directory}")
    return jpg_count

def get_date_string():
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")

def main():
    current_pic_num = count_jpg_files("./saved_img/")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"設定解析度為 {frame_width}x{frame_height}")
    
    # 延遲以讓相機調整曝光
    time.sleep(2)

    # 丟棄前幾幀以減少黑畫面
    for i in range(5):
        cap.read()

    interval = 10  # 每隔幾秒拍一次照
    last_capture_time = time.time()

    while True:
        # 讀取相機畫面
        ret, frame = cap.read()
        if not ret:
            print("無法接收畫面 (相機已關閉)")
            break

        current_time = time.time()
        
        # 判斷是否到達拍照時間間隔
        if current_time - last_capture_time >= interval:
            date_string = get_date_string()
            current_pic_num += 1
            file_name = f'./saved_img/{date_string}_img{current_pic_num}.jpg'
            cv2.imwrite(file_name, frame)
            print(f"照片已保存：'{file_name}'")
            last_capture_time = current_time

        # 顯示畫面
        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 按 Esc 鍵退出
            break

    # 釋放相機並關閉窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
