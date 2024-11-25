import cv2
import torch
import numpy as np
import os  # 用于遍历文件夹
from ultralytics import YOLO

ESC_KEY = 27

# 加载 YOLOv8 模型
model = YOLO('C:/Users/Peiteng.Chuang/Desktop/color_cube/ultralytics/runs/detect/sp_obj_v5/weights/best.pt')
print("Model loaded successfully! here we go!")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def main():
    # 指定包含图像的目录
    # img_dir = "./product_img"
    img_dir = "./a"

    # 遍历目录中的所有图像文件
    for img_name in os.listdir(img_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
            img_path = os.path.join(img_dir, img_name)
            frame = cv2.imread(img_path)
            # frame = cv2.resize(frame,(480,480))

            results = model(frame)
            # 更新 YOLO 推理结果
            last_results = results  # 保存最新结果
            
            # 在每一帧上绘制 YOLO 推理结果（使用上一次的推理结果）
            for result in last_results:
                boxes = result.boxes  # YOLOv8 中检测框的结果
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    conf = box.conf.item()  # 置信度
                    cls = box.cls.item()  # 类别索引
                    label = model.names[int(cls)]  # 获取物件的名称

                    color = (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 显示图像，窗口标题为图像名
            cv2.imshow(f"Object_{img_name}", frame)

            # 等待按键，按ESC退出
        
    # if cv2.waitKey(0) & 0xFF == ESC_KEY:
    #     break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
