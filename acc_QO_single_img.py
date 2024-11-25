import cv2
import torch
import numpy as np

import os
from ultralytics import YOLO

ESC_KEY = 27

# 加载 YOLOv8 模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ultralytics_path = "C:/project_file/ultralytics"

# 加载分类模型
cls_model_ver = "v9"
cls_model = YOLO(f'{ultralytics_path}/runs/classify/sp_cls_{cls_model_ver}/weights/best.pt', verbose=False)
# cls_model = YOLO(f'{ultralytics_path}/runs/classify/sp_cls_{cls_model_ver}/weights/hole_3_good.pt', verbose=False)
print(f"cls_Model loaded successfully! model version: {cls_model_ver}")
cls_model = cls_model.to(device)

# 加载检测模型
det_model_ver = "v10"
det_model = YOLO(f'{ultralytics_path}/runs/detect/sp_obj_{det_model_ver}/weights/best.pt', verbose=False)
print(f"det_Model loaded successfully! model version: {det_model_ver}")
det_model = det_model.to(device)

def check_label(cropped_img):
    results = cls_model(cropped_img, verbose=False)
    
    if results:  # 确保结果列表非空``
        probs = results[0].probs  # 提取第一项的概率
        predicted_idx = probs.top1  # 获取 top1 类别索引    
        label_name = results[0].names[predicted_idx]  # 根据索引返回名称
        return label_name
    else:
        return None

def do_otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bgr_thresh=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    # cv2.imshow("otsu_img",bgr_thresh)#驗證
    return bgr_thresh

def main():
    # 指定图像文件路径
    img_path = "./img/m10252.jpg"  # 替换为你的图像路径

    # img_path = "./product_img/img (1).png"  # 替换为你的图像路径
    # img_path = "./product_img/img_no (5).png"  # 替换为你的图像路径
    frame = cv2.imread(img_path)
    height, width, channels = frame.shape

    rate=1.5
    frame = cv2.resize(frame,(int(height/rate),int(width/rate)),interpolation=cv2.INTER_AREA)

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

            # 使用邊框加大來確認是否為邊緣圖像，多加大原圖的1/20=105%
            by = ((y2 - y1) / 20)
            bx = ((x2 - x1) / 20)
            yb1, yb2 = y1 - by, y2 + by
            xb1, xb2 = x1 - bx, x2 + bx

            if yb1 < 0 or yb2 > height or xb1 < 0 or xb2 > width:
                continue  # 跳过出界的检测框

            cropped_img = frame[int(yb1):int(yb2), int(xb1):int(xb2)]
            cpisz=cropped_img.size
            if cpisz > 0:  # 检查裁剪是否成功
                if cpisz>=crop_size[0]*crop_size[1]:        #大圖縮小最佳演算法
                    resized_img = cv2.resize(cropped_img, crop_size,interpolation=cv2.INTER_AREA)
                else:                                       #小圖放大最佳演算法
                    resized_img = cv2.resize(cropped_img, crop_size,interpolation=cv2.INTER_CUBIC)

                
                # otsu_img=do_otsu(resized_img)

                # label_name = check_label(otsu_img)
                
                label_name = check_label(resized_img)


                # 画出检测框
                
                if label_name is not None:
                    if label_name == 'O_shape':
                        cv2.putText(frame, f"O", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
                    
                    if label_name == 'Q_shape':
                        cv2.putText(frame, f"Q", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), green, 2)

                else:
                    cv2.putText(frame, "None", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 2)
                
                if label_name is not None:
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
        # cv2.imwrite(f"final_{det_model_ver}det_{cls_model_ver}cls.png", final_image)

    frame = cv2.resize(frame,(width//4,height//4), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"d{det_model_ver}c{cls_model_ver}_img.jpg", frame)
    cv2.imshow("Detection Results", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
