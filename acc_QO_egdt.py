import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from tqdm import tqdm  # 导入tqdm模块

ESC_KEY = 27
# 加载 YOLOv8 模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ultralytics_path = "C:/project_file/ultralytics"
# 加载分类模型
# 加载分类模型
egdt_model_ver = "v3"
egdt_model = YOLO(f'{ultralytics_path}/runs/detect/sp_egdt_{egdt_model_ver}/weights/best.pt', verbose=False)
print(f"egdt_Model loaded successfully! model version: {egdt_model_ver}")
egdt_model = egdt_model.to(device)

# 加载检测模型
det_model_ver = "v10"
det_model = YOLO(f'{ultralytics_path}/runs/detect/sp_obj_{det_model_ver}/weights/best.pt', verbose=False)
print(f"det_Model loaded successfully! model version: {det_model_ver}")
det_model = det_model.to(device)

def have_edge(cropped_img):
    results = egdt_model(cropped_img, verbose=False)
    egdt_flag=False
    if results:
          
        for result in results:
            boxes = result.boxes  # 获取检测框
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                egdt_flag=True
                cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,0), 2)
        return cropped_img,egdt_flag
                
    cropped_img = cv2.putText(cropped_img, f"O", (5,5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    return cropped_img,egdt_flag



def main():
    img_dir = "./product_img"
    # img_dir = "./img"
    crop_size = (100, 100)
    collected_images = []  # 存储处理后的图像

    # 获取所有图像文件名
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(img_files)  # 总文件数量

    # tqdm 进度条包裹循环
    for img_name in tqdm(img_files, desc="Processing Images", unit="file", leave=True):
        img_path = os.path.join(img_dir, img_name)
        frame = cv2.imread(img_path)

        height, width, channels = frame.shape
        # print(f"{img_name}==> height(y): {height}, width(x): {width}, channels: {channels}")  #除錯用
        results = det_model(frame, verbose=False)
        last_results = results

        count = 0
        for result in last_results:
            boxes = result.boxes  # 获取检测框
            for box in boxes:
                count += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                # print(f"x={int(x1)}~{int(x2)}, y={int(y1)}~{int(y2)},\t ",end="")     #除錯用
                green = (0, 255, 0)
                red = (0, 0, 255)
                yellow = (0, 255, 255)

                #使用邊框加大來確認是否為邊緣圖像
                by = ((y2 - y1) / 10)
                bx = ((x2 - x1) / 10)
                yb1,yb2 = y1-by,y2+by
                xb1, xb2 = x1-bx, x2+bx
                # print(f'do y={yb1:.2f}:{yb2:.2f}, x={xb1:.2f}:{xb2:.2f}')         #除錯用
                if yb1<0 or yb2>height or xb1<0 or xb2>width:
                    # print("**WARNING OUT OF RANGE**")
                    continue
                cropped_img = frame[int(yb1):int(yb2), int(xb1):int(xb2)]
                
                cpisz=cropped_img.size
                if cpisz > 0:  # 检查裁剪是否成功
                    if cpisz>=crop_size[0]*crop_size[1]:        #大圖縮小最佳演算法
                        resized_img = cv2.resize(cropped_img, crop_size,interpolation=cv2.INTER_AREA)
                    else:                                       #小圖放大最佳演算法
                        resized_img = cv2.resize(cropped_img, crop_size,interpolation=cv2.INTER_CUBIC)
#===========================================================================
                    dgdt_img,egdt_flag = have_edge(resized_img)
                    
                    if dgdt_img is not None:
                        if egdt_flag == False:
                            # cv2.putText(frame, f"O", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
                            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
                             cv2.putText(resized_img, f"O", (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)

                        if egdt_flag == True:
                            # cv2.putText(frame, f"Q", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
                            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), green, 2)
                             cv2.putText(resized_img, f"Q", (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)

                    else:
                        cv2.putText(frame, "None", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 2)

                    if dgdt_img is not None:
                        collected_images.append(resized_img)

#===========================================================================

        if cv2.waitKey(100) & 0xFF == ESC_KEY:
            break

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
        cv2.imshow("Final Image", final_image)
        # cv2.imwrite(f"{det_model_ver}det_{egdt_model_ver}egdt.png", final_image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
