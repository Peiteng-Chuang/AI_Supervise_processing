import cv2
import torch
import numpy as np

import os
from ultralytics import YOLO

ESC_KEY = 27

# 加载 YOLOv8 模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ultralytics_path = "C:/project_file/ultralytics"

# 加载邊角檢測模型
egdt_model_ver = "v7"
egdt_model = YOLO(f'{ultralytics_path}/runs/detect/sp_egdt_{egdt_model_ver}/weights/best.pt', verbose=False)
print(f"egdt_Model loaded successfully! model version: {egdt_model_ver}")
egdt_model = egdt_model.to(device)

# 加载孔洞檢測模型
det_model_ver = "v11"
det_model = YOLO(f'{ultralytics_path}/runs/detect/sp_obj_{det_model_ver}/weights/best.pt', verbose=False)

print(f"det_Model loaded successfully! model version: {det_model_ver}")
det_model = det_model.to(device)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # this thing is magic

print("webcam 1 has been open")

def have_edge(cropped_img):
    results = egdt_model(cropped_img, verbose=False)
    egdt_flag=False
    score_threshold=0.3
    if results:
          
        for result in results:
            boxes = result.boxes  # 获取检测框
            for box in boxes:
                score = box.conf[0].item()  # 获取置信度分數
                if score >= score_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    egdt_flag=True
                    cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,0), 2)
        return cropped_img,egdt_flag
                
    cropped_img = cv2.putText(cropped_img, f"O", (5,5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    return cropped_img,egdt_flag

def do_otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bgr_thresh=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    # cv2.imshow("otsu_img",bgr_thresh)#驗證
    return bgr_thresh

def main():
    
    # 檢查攝影機是否已正確開啟
    if not cap.isOpened():
        print("無法開啟攝影機")
        return

    # 設置影像解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 從 webcam 拍攝一張照片
    ret, frame = cap.read()
    if ret:
        # 印出照片的 shape (高度, 寬度, 色彩通道數)
        print("照片的 shape:", frame.shape)
        height, width, channels = frame.shape

        set_rate=1280
        width_set=set_rate
        height_set=int(height/width*set_rate)
        # frame = cv2.resize(frame,(int(width/rate),int(height/rate)),interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame,(width_set,height_set),interpolation=cv2.INTER_AREA)
    
        crop_size = (100, 100)
        collected_images = []
    
        # 进行物体检测
        results = det_model(frame, verbose=False)
        # results = det_model(d2_frame, verbose=False)
    
        for result in results:
            boxes = result.boxes  # 获取检测框
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                green = (0, 255, 0)
                red = (0, 0, 255)
                yellow = (0, 255, 255)
    
                # 使用邊框加大來確認是否為邊緣圖像
                by = ((y2 - y1) / 10)
                bx = ((x2 - x1) / 10)
                yb1, yb2 = y1 - by, y2 + by
                xb1, xb2 = x1 - bx, x2 + bx
    
                if yb1 < 0 or yb2 > height or xb1 < 0 or xb2 > width:
                    continue  # 跳过出界的检测框
                
                cropped_img = frame[int(yb1):int(yb2), int(xb1):int(xb2)]
                # cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
                cpisz=cropped_img.size
                if cpisz > 0:  # 检查裁剪是否成功
                    if cpisz>=crop_size[0]*crop_size[1]:        #大圖縮小最佳演算法
                        resized_img = cv2.resize(cropped_img, crop_size,interpolation=cv2.INTER_AREA)
                    else:                                       #小圖放大最佳演算法
                        resized_img = cv2.resize(cropped_img, crop_size,interpolation=cv2.INTER_CUBIC)
                    
                    egdt_img,egdt_flag = have_edge(resized_img)
    
                    # # 画出检测框
                    
                    if egdt_img is not None:
                        if egdt_flag == False:
                            cv2.putText(frame, f"O", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
                        
                        if egdt_flag == True:
                            cv2.putText(frame, f"Q", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), green, 2)
    
                    else:
                        cv2.putText(frame, "None", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 2)
                    
                    if egdt_img is not None:
                        collected_images.append(resized_img)
    
        max_width = crop_size[0] * 10
        images_per_row = 10
        rows = []
        current_row = []
    
        for i, img in enumerate(collected_images):
            if img.shape[0] == crop_size[0] and img.shape[1] == crop_size[1]:  # 确保图像尺寸正确
                current_row.append(img)
                if len(current_row) == images_per_row or i == len(collected_images) - 1:
                    row_image = cv2.hconcat(current_row)
                    if row_image.shape[1] < max_width:
                        padding = np.zeros((crop_size[1], max_width - row_image.shape[1], 3), dtype=np.uint8)
                        row_image = np.hstack((row_image, padding))
                    rows.append(row_image)
                    current_row = []
    
        if len(rows) > 0:
            final_image = cv2.vconcat(rows)
            cv2.imshow("all_hole_in_this_img", final_image)
            # cv2.imwrite(f"egdt{egdt_model_ver}_holes.jpg", final_image)
    
    
        cv2.imwrite(f"./img/pic_d{det_model_ver}egdt{egdt_model_ver}.jpg", frame)     #存辨識原圖
        print(f"{frame.shape}")
        frame=cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))   #把frame變成比較好顯示的大小
        cv2.imshow("Detection Results", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("無法拍攝照片")

    # 釋放攝影機資源
    cap.release()
    # 指定图像文件路径
    # pth="./img/"
    # f_name='egdt_test.jpg'
    # img_path = pth+f_name  # 替换为你的图像路径

    # # img_path = "./product_img/img (1).png"  # 替换为你的图像路径
    # # img_path = "./product_img/img_no (5).png"  # 替换为你的图像路径
    # frame = cv2.imread(img_path)
    

if __name__ == "__main__":
    main()
