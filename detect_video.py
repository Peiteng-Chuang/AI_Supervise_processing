import cv2
import torch
import numpy as np
import os  # 用于遍历文件夹
from ultralytics import YOLO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ultralytics_path="C:/project_file/ultralytics"

#加載分類模型
cls_model_ver="v4"
cls_model = YOLO(f'{ultralytics_path}/runs/classify/sp_cls_{cls_model_ver}/weights/best.pt')
print(f"cls_Model loaded successfully! model version: {cls_model_ver}")
cls_model = cls_model.to(device)

#加載偵測模型
det_model_ver="v12"
det_model = YOLO(f'{ultralytics_path}/runs/detect/sp_obj_{det_model_ver}/weights/best.pt')
print(f"det_Model loaded successfully! model version: {det_model_ver}")
det_model = det_model.to(device)

# print(torch.cuda.is_available())  # 检查是否检测到 CUDA
# print(torch.cuda.device_count())   # 检查有多少个 GPU
# print(torch.cuda.get_device_name(0))  # 获取 GPU 的名称
# print(cls_model.device)  # 检查分类模型的设备
# print(det_model.device)  # 检查检测模型的设备


def check_label(cropped_img):
    results = cls_model(cropped_img)
    
    # 從列表中的第一個結果提取 `probs`
    if results:  # 確保結果列表非空
        probs = results[0].probs  # 提取第一個結果的機率對象
        
        predicted_idx = probs.top1  # 使用 top1 來獲取最高機率的類別索引
        
        # 根據類別索引返回對應的名稱
        label_name = results[0].names[predicted_idx]
        return label_name
    else:
        return None


# 打開輸入視頻
# cap = cv2.VideoCapture('m1025_cut.mp4')
cap = cv2.VideoCapture('IMG_5100.mp4')

# fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
# fourcc_str = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
# print(f"FourCC code: {fourcc_str}")

# 取得視頻資訊
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設置保存為 MP4 格式
fourcc = cv2.VideoWriter_fourcc(*'X264')  # 設置保存為 MP4 格式
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# 設置輸出視頻
out = cv2.VideoWriter('test2.mp4', fourcc, fps, (width, height))
frame_count=0
crop_size=(100,100)

while cap.isOpened():
    ret, frame = cap.read()
    frame_count+=1
    if not ret:
        break
    
    # 轉換影像為 RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 用 mediapipe 進行人體姿態偵測
    results = det_model(frame)
    # 如果有偵測到姿態，繪製點與連接線
    last_results = results  # 保存最新结果
            
    # 在每一帧上绘制 YOLO 推理结果（使用上一次的推理结果）
    count=0
    for result in last_results:
        boxes = result.boxes  # YOLOv8 中检测框的结果
        for box in boxes:
            count+=1
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
            conf = box.conf.item()  # 置信度
            cls = box.cls.item()  # 类别索引
            label = det_model.names[int(cls)]  # 获取物件的名称
            #================================================================測量方框是否為正
            # long_side,short_side=max((x2-x1),(y2-y1)),min((x2-x1),(y2-y1))
            # ls_rate=short_side/long_side
            # print(f"ls_rate: {ls_rate:.2f}")
            # if ls_rate<=0.7:                #如果長短比少於0.7，就不是一個完整的圖形(完整的應為方形)
            #     continue
            #===============================================================
            green = (0, 255, 0)
            red = ( 0 , 0 , 255)
            yellow=(0,255,255)

            #===============================================================裁切
            by=(int(y2)-int(y1))//20
            bx=(int(x2)-int(x1))//20
            cropped_img = frame[int(y1)-by:int(y2)+by, int(x1)-bx:int(x2)+bx]
            if cropped_img.size > 0:  # 检查是否裁剪成功
                resized_img = cv2.resize(cropped_img, crop_size)
                # score = image_match_score_with_SIFT(resized_img)        #改這段
                
                label_name = check_label(resized_img)                       #跑模型
                if label_name is not None:
                    lb_loc=(int(x1), int(y1)-10)
                    if label_name=='O_shape':
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), red, 1)
                        cv2.putText(frame, f"O", lb_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 1)
                    elif label_name=='Q_shape':
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), green, 1)
                        cv2.putText(frame, f"Q", lb_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1)
                    else:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), yellow, 1)
                        cv2.putText(frame, f"wtf", lb_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1)
                else:
                    cv2.putText(frame, f"None", lb_loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1)
            #===============================================================

            

    cv2.putText(frame,F'frame : {frame_count}',(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # 將結果寫入輸出視頻
    out.write(frame)
    # 顯示當前幀影像（可選）
    # cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
