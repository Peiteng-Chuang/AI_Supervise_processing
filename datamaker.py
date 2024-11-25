import cv2
import torch
import numpy as np
import os  # 用于遍历文件夹
from ultralytics import YOLO
import random

ESC_KEY = 27
# 加载 YOLOv8 模型
model_ver="v12"
ultralytics_path="C:/project_file/ultralytics/"
model = YOLO(f'{ultralytics_path}runs/detect/sp_obj_{model_ver}/weights/best.pt')
print("Model loaded successfully! here we go!")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def main():
    # 指定包含图像的目录
    # img_dir = "./product_img"
    img_dir = "./unmake_data"

    # 遍历目录中的所有图像文件
    for img_name in os.listdir(img_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
            img_path = os.path.join(img_dir, img_name)
            
            frame = cv2.imread(img_path)
            height, width, channels = frame.shape
            
            rate=1.5
            frame = cv2.resize(frame,(int(height/rate),int(width/rate)),interpolation=cv2.INTER_AREA)

            results = model(frame)
            # 更新 YOLO 推理结果
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
                    label = model.names[int(cls)]  # 获取物件的名称

                    crop_size=(100, 100)
                    r=3
                    by=int((y2-y1)/3)
                    bx=int((x2-x1)/3)
                    cropped_img = frame[int(y1)-by:int(y2)+by, int(x1)-bx:int(x2)+bx]

                    # random_number = random.randint(1, 4)
                    # r=random_number+3
                    # by=(int(y2)-int(y1))//r*(r-1)
                    # bx=(int(x2)-int(x1))//r*(r-1)
                    # if random_number==1:
                    #     cropped_img = frame[int(y1)-by:int(y2)-by, int(x1)-bx:int(x2)-bx]
                    # elif random_number==2:
                    #     cropped_img = frame[int(y1)-by:int(y2)-by, int(x1)+bx:int(x2)+bx]                        
                    # elif random_number==3:
                    #     cropped_img = frame[int(y1)+by:int(y2)+by, int(x1)-bx:int(x2)-bx]                        
                    # else:
                    #     cropped_img = frame[int(y1)+by:int(y2)+by, int(x1)+bx:int(x2)+bx]                        
                        

                    if cropped_img is not None and cropped_img.size > 0:  # 检查是否裁剪成功
                        base_name = os.path.splitext(img_name)[0]

                        print(f"{img_name}--{count} crooped success x:{ int(x1),int(x2)},y:{int(y1),int(y2)}")
                        resized_img = cv2.resize(cropped_img, crop_size)
                        # cv2.imwrite(f"./data/{base_name}_{count}_r{r}.png", resized_img)
                        cv2.imwrite(f"./data/{base_name}_{count}_r{r}.png", resized_img)

            # 等待按键
            if cv2.waitKey(100) & 0xFF == ESC_KEY:  # 每张显示 100 毫秒
                break


if __name__ == "__main__":
    main()
