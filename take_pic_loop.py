# import cv2
# import os
# import time
# from datetime import datetime
# import numpy as np

# # 設置解析度 (超過會預設相機最大解析度)
# resolution_w, resolution_h = 1920, 1080

# # 初始化相機
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_w)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_h)

# # 檢查相機是否成功開啟
# if not cap.isOpened():
#     print("無法開啟相機")
#     exit()

# def adjust_gamma(image, gamma=1.0):
#     # 建立查找表（LUT）
#     inv_gamma = 0.9 / gamma
#     table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    
#     # 套用 LUT 到圖像
#     return cv2.LUT(image, table)

# def count_jpg_files(directory):
#     if not os.path.isdir(directory):
#         print(f"路徑 {directory} 不存在")
#         return 0
#     jpg_count = len([file for file in os.listdir(directory) if file.lower().endswith('.jpg')])
#     print(f"找到 {jpg_count} 個 JPG 檔案在 {directory}")
#     return jpg_count

# def get_date_string():
#     now = datetime.now()
#     return now.strftime("%Y%m%d-%H%M%S")

# def is_in_time_range():
#     current_time = datetime.now()
#     # 設置工作時間範圍 08:00-12:00 和 13:00-17:00
#     if (current_time.hour >= 8 and current_time.hour < 12) or (current_time.hour >= 13 and current_time.hour < 17):
#         return True
#     return False

# def create_folder_for_today():
#     today_date = datetime.now().strftime("%Y%m%d")
#     folder_path = f"./saved_img/{today_date}"
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     return folder_path

# def main():
#     gamma = 0.9
#     current_pic_num = 0
#     folder_path = create_folder_for_today()
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f"設定解析度為 {frame_width}x{frame_height}")
    
#     # 延遲以讓相機調整曝光
#     time.sleep(2)

#     # 丟棄前幾幀以減少黑畫面
#     for i in range(5):
#         cap.read()

#     interval = 60  # 每隔幾秒拍一次照
#     last_capture_time = time.time()

#     while True:
#         # 檢查是否在允許的時間範圍內
#         if not is_in_time_range():
#             # 若不在時間範圍內，暫停一段時間再檢查
#             time.sleep(60)  # 每分鐘檢查一次
#             continue

#         # 讀取相機畫面
#         ret, frame = cap.read()
#         frame = adjust_gamma(frame, gamma)
#         if not ret:
#             print("無法接收畫面 (相機已關閉)")
#             break

#         current_time = time.time()
        
#         # 判斷是否到達拍照時間間隔
#         if current_time - last_capture_time >= interval:
#             date_string = get_date_string()
#             current_pic_num += 1
#             file_name = f'{folder_path}/{date_string}_img{current_pic_num}.jpg'
#             cv2.imwrite(file_name, frame)
#             print(f"照片已保存：'{file_name}'")
#             last_capture_time = current_time

#         # 顯示畫面
#         resized = cv2.resize(frame, (960, 540))
#         cv2.imshow('Camera', resized)

#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:  # 按 Esc 鍵退出
#             break

#     # 釋放相機並關閉窗口
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()
#======================================================================================================
import cv2
import os
import time
from datetime import datetime
import numpy as np

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

def adjust_gamma(image, gamma=1.0):
    # 建立查找表（LUT）
    inv_gamma = 0.9 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    
    # 套用 LUT 到圖像
    return cv2.LUT(image, table)

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

def is_in_time_range(work_time=(8,12,13,17)):
    current_time = datetime.now()
    # 設置工作時間範圍 08:00-12:00 和 13:00-17:00
    if (current_time.hour >= work_time[0] and current_time.hour < work_time[1]) or (current_time.hour >= work_time[2] and current_time.hour < work_time[3]):
        return True
    return False

def create_folder_for_today():
    today_date = datetime.now().strftime("%Y%m%d")
    folder_path = f"./saved_img/{today_date}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def main():
    gamma = 0.9
    current_pic_num = 0
    folder_path = create_folder_for_today()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"設定解析度為 {frame_width}x{frame_height}")
    
    # 延遲以讓相機調整曝光
    time.sleep(2)

    # 丟棄前幾幀以減少黑畫面
    for i in range(5):
        cap.read()

    interval = 60  # 每隔幾秒拍一次照
    last_capture_time = time.time()

    while True:
        # 檢查是否在允許的時間範圍內
        if not is_in_time_range():
            # 若不在時間範圍內，暫停一段時間再檢查
            time.sleep(60)  # 每分鐘檢查一次
            continue

        # 讀取相機畫面
        ret, frame = cap.read()
        frame = adjust_gamma(frame, gamma)
        if not ret:
            print("無法接收畫面 (相機已關閉)")
            break

        current_time = time.time()
        
        # 判斷是否到達拍照時間間隔
        if current_time - last_capture_time >= interval:
            date_string = get_date_string()
            current_pic_num += 1
            file_name = f'{folder_path}/{date_string}_img{current_pic_num}.jpg'
            cv2.imwrite(file_name, frame)
            print(f"照片已保存：'{file_name}'")
            last_capture_time = current_time

        # 顯示畫面
        resized = cv2.resize(frame, (960, 540))
        cv2.imshow('Camera', resized)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 按 Esc 鍵退出
            break

    # 釋放相機並關閉窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
