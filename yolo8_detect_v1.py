import cv2
import torch
import numpy as np
import time  # 用于控制推理速度
from ultralytics import YOLO
from pyorbbecsdk import Config, Pipeline, OBError, OBSensorType, OBFormat, FrameSet, VideoStreamProfile
from orbbec_utils import frame_to_bgr_image

ESC_KEY = 27

# 加载 YOLOv8 模型
version="v6"
model = YOLO(f'C:/Users/Peiteng.Chuang/Desktop/color_cube/ultralytics/runs/detect/sp_obj_{version}/weights/best.pt')
print("Model loaded successfully! here we go!")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 設定模型推理的頻率 (每秒5次)
MODEL_INFERENCE_INTERVAL = 0.1  # 0.1 秒

def main():
    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return

    pipeline.start(config)

    last_inference_time = 0  # 初始化模型推理的時間
    last_results = []  # 保存上一次的YOLO推理結果

    while True:
        try:
            # 获取当前时间
            current_time = time.time()

            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            # 将帧转换为BGR格式
            frame = frame_to_bgr_image(color_frame)
            if frame is None:
                print("failed to convert frame to image")
                continue

            # 每隔 0.2 秒（5 次/秒）进行一次 YOLO 推理
            if current_time - last_inference_time >= MODEL_INFERENCE_INTERVAL:
                # 使用 YOLOv8 进行物件检测
                results = model(frame, verbose=False)

                # 更新 YOLO 推理結果
                last_results = results  # 保存最新結果

                # 更新最后一次推理的时间
                last_inference_time = current_time

            # 在每一幀上繪製 YOLO 推理結果（使用上一次的推理結果）
            for result in last_results:
                boxes = result.boxes  # YOLOv8 中检测框的结果
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    conf = box.conf.item()  # 置信度
                    cls = box.cls.item()  # 类别索引
                    label = model.names[int(cls)]  # 获取物件的名称

                    color = (0, 0, 255)
                    if label == "processed":
                        color = (0, 255, 0)

                    # 在图像上显示物件名称和中心点坐标
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 即使没有模型推理，也要显示相机画面
            cv2.imshow("Object Tracking with YOLOv8", frame)

            # 按 'q' 键退出
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break

        except KeyboardInterrupt:
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
