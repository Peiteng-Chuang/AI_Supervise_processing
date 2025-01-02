import cv2
from ultralytics import YOLO
import torch


# 設定裝置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
version = "v3"
model = YOLO(f"C:/project_file/ultralytics_v11/runs/detect/sp_obtk_{version}/weights/best.pt")
model = model.to(device)

# 輸入影片檔案
vid_name = "2024-11-26_11-17-30.avi"
video_path = f"C:/Users/Peiteng.Chuang/Desktop/factor/video/{vid_name}"
cap = cv2.VideoCapture(video_path)

# 確認影片是否開啟成功
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# 取得影片資訊
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 輸出影片設定
output_path = f"C:/Users/Peiteng.Chuang/Desktop/{vid_name}"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 選擇編碼格式，例如 'XVID' 或 'MP4V'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# while cap.isOpened():
#     success, frame = cap.read()
#     if success:
#         # 使用模型處理
#         results = model.track(frame, persist=True, conf=0.6, iou=0.3, tracker="bytetrack.yaml")
#         annotated_frame = results[0].plot()

#         # 將處理後的畫面寫入影片
#         out.write(annotated_frame)

#         # 顯示處理後的畫面
#         cv2.imshow("YOLO11 Tracking", annotated_frame)

#         # 按下 Esc 鍵退出
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
#     else:
#         break
##================================================================================================
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 使用模型處理
        results = model.track(frame, persist=True, conf=0.6, iou=0.3, tracker="bytetrack.yaml")

        # 提取類別名稱
        class_names = model.names  # 提取模型的類別名稱字典

        # 過濾掉 "saw" 的檢測結果
        detections = results[0].boxes.data if results[0].boxes is not None else []
        filtered_detections = []
        for detection in detections:
            class_id = int(detection[-1])  # 獲取類別 ID
            if class_names[class_id] != "saw":
                filtered_detections.append(detection)

        # 繪製剩餘的檢測結果
        annotated_frame = frame.copy()
        for detection in filtered_detections:
            bbox = detection[:4].cpu().numpy().astype(int)  # 邊界框
            conf = detection[4].item()  # 信心值
            class_id = int(detection[-1])  # 類別 ID
            cv2.rectangle(
                annotated_frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (255, 0, 0), 2  # 藍色框
            )
            cv2.putText(
                annotated_frame,
                f"{class_names[class_id]} {conf:.2f}",
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1
            )

        # 將處理後的畫面寫入影片
        out.write(annotated_frame)

        # 顯示處理後的畫面
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # 按下 Esc 鍵退出
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break
#================================================================================================
# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at {output_path}")
