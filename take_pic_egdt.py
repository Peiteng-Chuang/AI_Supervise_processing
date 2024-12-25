# import cv2
# import torch
# import numpy as np
# import time
# from ultralytics import YOLO
# from datetime import datetime

# ESC_KEY = 27

# # 加载 YOLOv8 模型
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ultralytics_path = "C:/project_file/ultralytics"

# # 加载邊角檢測模型
# egdt_model_ver = "v12"
# egdt_model = YOLO(f'{ultralytics_path}/runs/detect/sp_egdt_{egdt_model_ver}/weights/best.pt', verbose=False)
# print(f"egdt_Model loaded successfully! model version: {egdt_model_ver}")
# egdt_model = egdt_model.to(device)

# # 加载孔洞檢測模型
# det_model_ver = "v12"
# det_model = YOLO(f'{ultralytics_path}/runs/detect/sp_holes_{det_model_ver}/weights/best.pt', verbose=False)

# print(f"det_Model loaded successfully! model version: {det_model_ver}")
# det_model = det_model.to(device)

# cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)  # 使用攝影機
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

# print("webcam 1 has been opened")

# def get_date_string():
#     now = datetime.now()
#     return now.strftime("%Y%m%d-%H%M%S")

# def have_edge(cropped_img):
#     results = egdt_model(cropped_img, verbose=False)
#     egdt_flag=False
#     score_threshold=0.3
#     if results:
#         for result in results:
#             boxes = result.boxes  # 获取检测框
#             for box in boxes:
#                 score = box.conf[0].item()  # 获取置信度分數
#                 if score >= score_threshold:
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
#                     egdt_flag=True
#                     cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,0), 2)
#         return cropped_img,egdt_flag
#     cropped_img = cv2.putText(cropped_img, f"O", (5,5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
#     return cropped_img,egdt_flag

# def enhance_brightness(frame):
#     # 提高圖像亮度
#     return cv2.convertScaleAbs(frame, alpha=1.2, beta=30)  # 增加亮度

# def calculate_mse(frame1, frame2):
#     return np.mean((frame1.astype("float") - frame2.astype("float")) ** 2)

# def main():
#     log_file_path = './saved_img/detect_record/detection_log.log'
#     # 確認相機是否成功開啟
#     if not cap.isOpened():
#         print("無法開啟攝影機")
#         return

#     last_action_time = time.time()  # 記錄最後一次動作的時間

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("無法拍攝照片")
#             break

#         # 計算經過的時間
#         current_time = time.time()
#         elapsed_time = current_time - last_action_time

#         # 每隔30秒執行一次檢測
#         if elapsed_time >= 60:
            
#             items=0
#             unprossed_item=0
#             print("Performing detection task...")
#             last_action_time = current_time  # 更新最後一次執行時間

#             # 對相機畫面進行處理
#             frame = enhance_brightness(frame)
            
#             # 進行物體檢測
#             results = det_model(frame, verbose=False)
#             for result in results:
#                 boxes = result.boxes  # 获取检测框
#                 for box in boxes:
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
#                     green = (0, 255, 0)
#                     red = (0, 0, 255)
                    

#                     # 使用邊框加大來確認是否為邊緣圖像
#                     by = ((y2 - y1) / 10)
#                     bx = ((x2 - x1) / 10)
#                     yb1, yb2 = y1 - by, y2 + by
#                     xb1, xb2 = x1 - bx, x2 + bx

#                     if yb1 < 0 or yb2 > frame.shape[0] or xb1 < 0 or xb2 > frame.shape[1]:
#                         continue  # 跳过出界的检测框
                    
#                     cropped_img = frame[int(yb1):int(yb2), int(xb1):int(xb2)]
#                     cpisz=cropped_img.size
#                     if cpisz > 0:  # 检查裁剪是否成功
#                         items+=1
#                         # 根據裁剪圖像大小選擇最佳縮放方法
#                         if cpisz >= 100 * 100:  # 确保图像尺寸足够大
#                             resized_img = cv2.resize(cropped_img, (100, 100), interpolation=cv2.INTER_AREA)
#                         else:
#                             resized_img = cv2.resize(cropped_img, (100, 100), interpolation=cv2.INTER_CUBIC)

#                         egdt_img, egdt_flag = have_edge(resized_img)

#                         # 画出检测框
#                         if egdt_img is not None:
#                             if egdt_flag == False:
#                                 cv2.putText(frame, f"O", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
#                                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
#                                 unprossed_item+=1
                            
#                             if egdt_flag == True:
#                                 cv2.putText(frame, f"Q", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
#                                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), green, 2)
#             if unprossed_item!=0:
#                 with open(log_file_path, "a", encoding="utf-8") as log_file:
#                     log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Detected items: {items} - Unprocessed items: {unprossed_item}\n")
#             # 顯示即時影像
#             cv2.imshow("Detection Results", frame)
#             date_string = get_date_string()
#             file_name = f'./saved_img/detect_record/{date_string}.jpg'
#             cv2.imwrite(file_name,frame)

#         # 按下 ESC 鍵退出
#         if cv2.waitKey(1) & 0xFF == ESC_KEY:
#             break

#     # 釋放資源
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from datetime import datetime

ESC_KEY = 27

# 載入 YOLOv8 模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ultralytics_path = "C:/project_file/ultralytics"

# 載入邊角檢測模型
egdt_model_ver = "v12"
egdt_model = YOLO(f'{ultralytics_path}/runs/detect/sp_egdt_{egdt_model_ver}/weights/best.pt', verbose=False)
print(f"egdt_Model loaded successfully! model version: {egdt_model_ver}")
egdt_model = egdt_model.to(device)

# 載入孔洞檢測模型
det_model_ver = "v12"
det_model = YOLO(f'{ultralytics_path}/runs/detect/sp_holes_{det_model_ver}/weights/best.pt', verbose=False)

print(f"det_Model loaded successfully! model version: {det_model_ver}")
det_model = det_model.to(device)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 使用擊像頭
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("webcam 1 has been opened")

def get_date_string():
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")

def have_edge(cropped_img):
    results = egdt_model(cropped_img, verbose=False)
    egdt_flag = False
    score_threshold = 0.3
    if results:
        for result in results:
            boxes = result.boxes  # 获取检测框
            for box in boxes:
                score = box.conf[0].item()  # 获取信心度分数
                if score >= score_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    egdt_flag = True
                    cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return cropped_img, egdt_flag
    cropped_img = cv2.putText(cropped_img, f"O", (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return cropped_img, egdt_flag

def enhance_brightness(frame):
    # 提高圖像亮度
    return cv2.convertScaleAbs(frame, alpha=1.2, beta=30)  # 增加亮度

def calculate_mse(frame1, frame2):
    return np.mean((frame1.astype("float") - frame2.astype("float")) ** 2)

def main():
    log_file_path = './saved_img/detect_record/detection_log.log'
    # 確認相機是否成功開啟
    if not cap.isOpened():
        print("無法開啟擊像頭")
        return

    last_action_time = time.time()  # 記錄最後一次動作的時間
    last_frame = None  # 記錄最後一張圖片

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法拍攝照片")
            break

        # 計算經過的時間
        current_time = time.time()
        elapsed_time = current_time - last_action_time

        # 每隔30秒執行一次穩定性检查
        if elapsed_time >= 60:
            if last_frame is not None:
                mse = calculate_mse(last_frame, frame)
                print(f"MSE: {mse}")

                # 如果比較穩定（MSE低於開關）前往拍攝
                if mse < 500:  # 您可以釋述MSE開關
                    print("Performing detection task...")
                    last_action_time = current_time  # 更新最後一次執行時間

                    items = 0
                    unprossed_item = 0

                    # 對相機畫面進行處理
                    frame = enhance_brightness(frame)

                    # 進行物體检测
                    results = det_model(frame, verbose=False)
                    for result in results:
                        boxes = result.boxes  # 获取检测框
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                            green = (0, 255, 0)
                            red = (0, 0, 255)

                            # 使用邊框加大來確認是否為邊緣圖像
                            by = ((y2 - y1) / 10)
                            bx = ((x2 - x1) / 10)
                            yb1, yb2 = y1 - by, y2 + by
                            xb1, xb2 = x1 - bx, x2 + bx

                            if yb1 < 0 or yb2 > frame.shape[0] or xb1 < 0 or xb2 > frame.shape[1]:
                                continue  # 跳過出界的检测框

                            cropped_img = frame[int(yb1):int(yb2), int(xb1):int(xb2)]
                            cpisz = cropped_img.size
                            if cpisz > 0:  # 检查裁剪是否成功
                                items += 1
                                # 根據裁剪圖像大小選擇最佳縮放方法
                                if cpisz >= 100 * 100:  # 确保图像尺寸足够大
                                    resized_img = cv2.resize(cropped_img, (100, 100), interpolation=cv2.INTER_AREA)
                                else:
                                    resized_img = cv2.resize(cropped_img, (100, 100), interpolation=cv2.INTER_CUBIC)

                                egdt_img, egdt_flag = have_edge(resized_img)

                                # 画出检测框
                                if egdt_img is not None:
                                    if not egdt_flag:
                                        cv2.putText(frame, f"O", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
                                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
                                        unprossed_item += 1

                                    if egdt_flag:
                                        cv2.putText(frame, f"Q", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
                                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), green, 2)

                    if unprossed_item != 0:
                        with open(log_file_path, "a", encoding="utf-8") as log_file:
                            log_file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Detected items: {items} - Unprocessed items: {unprossed_item}\n")
                    # 顯示即時影像
                    cv2.imshow("Detection Results", frame)
                    date_string = get_date_string()
                    file_name = f'./saved_img/detect_record/{date_string}.jpg'
                    cv2.imwrite(file_name, frame)

            last_frame = frame.copy()

        # 按下 ESC 鍵退出
        if cv2.waitKey(1) & 0xFF == ESC_KEY:
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
