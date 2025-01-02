import cv2
import torch
import numpy as np
import os  # 用于遍历文件夹
from ultralytics import YOLO


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ESC_KEY = 27

model_ver="v13f"
ultralytics_path="C:/project_file/ultralytics_v11/"
model = YOLO(f'{ultralytics_path}runs/detect/sp_egdt_{model_ver}/weights/best.pt')
print(f"Model EGDT_{model_ver} loaded successfully! here we go!")

model = model.to(device)

def have_edge(cropped_img):
    # gray=do_otsu(cropped_img)
    # results = egdt_model(gray, verbose=False)
    # cropped_img=gray
    results = model(cropped_img, verbose=False)
    egdt_flag=False
    score_threshold=[0.2,0.4,0.6]
    if results:
          
        for result in results:
            boxes = result.boxes  # 获取检测框
            for box in boxes:
                score = box.conf[0].item()  # 获取置信度分數
                if score >= score_threshold[2]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    egdt_flag=True
                    cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,0), 2)
                elif score >= score_threshold[1]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    # egdt_flag=True
                    cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,255), 2)
                elif score >= score_threshold[0]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    # egdt_flag=True
                    # cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,125,255), 2)
                else:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,0,255), 2)
        return cropped_img,egdt_flag
                
    cropped_img = cv2.putText(cropped_img, f"O", (5,5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    return cropped_img,egdt_flag

def main():
    # 指定包含图像的目录
    # img_dir = "./product_img"
    # img_dir = "C:/Users/Peiteng.Chuang/Desktop/factor/image/"
    img_dir = "C:/project_file/AI_Supervise_processing/data/"
    # img_dir = "./unmake_data"

    # 遍历目录中的所有图像文件
    for img_name in os.listdir(img_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
            img_path = os.path.join(img_dir, img_name)
            
            frame = cv2.imread(img_path)
            height, width, channels = frame.shape
            print("/",end="")
            cropped_img,egdt_flag=have_edge(frame.copy())
            count=0
    
            if egdt_flag==False:  # 查看是否沒有邊角
                base_name = os.path.splitext(img_name)[0]
                print(f"\n{img_name}--{count} detect success , none edge detect.")
                cv2.imwrite(f"./data_filter/{base_name}_{count}_filter_bad.png", frame)

            # 等待按键
            # if cv2.waitKey(1) & 0xFF == ESC_KEY:  # 每张显示 100 毫秒
            #     break


if __name__ == "__main__":
    main()
