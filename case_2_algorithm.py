import cv2
import torch
import numpy as np
import os  # 用于遍历文件夹
from ultralytics import YOLO

ESC_KEY = 27
# 加载 YOLOv8 模型
model_ver="v10"
model = YOLO(f'C:/Users/Peiteng.Chuang/Desktop/color_cube/ultralytics/runs/detect/sp_obj_{model_ver}/weights/best.pt')
print("Model loaded successfully! here we go!")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

sift_image = cv2.imread('threadhole_img.png',cv2.IMREAD_GRAYSCALE)
# sift_image = cv2.imread('sift_img.png',cv2.IMREAD_GRAYSCALE)
sift_obj = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=3)
orb = cv2.ORB_create()


def image_match_score_with_SIFT(image,sift_img=sift_image,sift=sift_obj):

    if len(image.shape) == 3:  # 檢查是否是彩色圖像
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    keypoints1, descriptors1 = sift.detectAndCompute(image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(sift_img, None)

    if descriptors1 is None or descriptors2 is None:
        print("未能提取到特徵點")
        return 0  # 回傳相似度為0，表示不相似

    # 使用BFMatcher進行描述子匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    similarity_score = len(matches)
    if len(matches) > 0:
        average_distance = sum([match.distance for match in matches]) / len(matches)
    else:
        average_distance = float('inf')  # 如果沒有匹配，設置為無限大
    
    # 計算匹配率，範圍為0到1
    matching_ratio = len(matches) / min(len(descriptors1), len(descriptors2))
    matching_score=matching_ratio*100
    print(f"*Sim: {similarity_score}, Avg: {average_distance:.2f}, Mhs: {matching_score}pt\n{'='*30}")
    
    return matching_score


def is_this_hole_processed(hole_image):
    # 轉成灰階圖像
    gray = cv2.cvtColor(hole_image, cv2.COLOR_BGR2GRAY)
    
    # 使用 Otsu 自動二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 應用 Canny 邊緣檢測，根據二值化結果進行邊緣檢測
    edges = cv2.Canny(binary, 50, 150)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到圖像的中心點
        img_center = (hole_image.shape[1] // 2, hole_image.shape[0] // 2)
        
        # 計算每個輪廓的中心點，並找出最接近圖像中心的內層輪廓
        min_distance = float('inf')
        selected_contour = None
        
        for contour in contours:
            # 計算輪廓的邊界框並找其質心
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                contour_center = (cx, cy)
                
                # 計算此輪廓中心與圖像中心的距離
                distance = np.sqrt((contour_center[0] - img_center[0]) ** 2 + (contour_center[1] - img_center[1]) ** 2)
                
                # 選取最接近中心點的輪廓
                if distance < min_distance:
                    min_distance = distance
                    selected_contour = contour
        
        # 如果找到內層輪廓，則進行圓形度判斷
        if selected_contour is not None:
            # 計算面積與周長
            area = cv2.contourArea(selected_contour)
            perimeter = cv2.arcLength(selected_contour, True)
            
            # 計算圓形度
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            
            # 計算圖像分數
            score = circularity * 100  # 分數為圓形度的百分比
            
            # 判斷分數範圍
            if score > 87:  # 設定85分以上為圓形，根據需求調整
                return True, score
            else:
                return False, score
    
    # 無法識別時回傳0分
    return False, 0


    

def main():
    # 指定包含图像的目录
    img_dir = "./product_img"
    # img_dir = "./a"

    collected_images = []  # 存储处理后的图像

    # 遍历目录中的所有图像文件
    for img_name in os.listdir(img_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
            img_path = os.path.join(img_dir, img_name)
            frame = cv2.imread(img_path)
            
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

                    green = (0, 255, 0)
                    red = ( 0 , 0 , 255)
                    yellow=(0,255,255)
                    #單獨視窗畫出框框跟標記
                    # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), green, 2)
                    # cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1)

                    #設定裁切大小
                    crop_size=(100, 100)
                    
                    # 裁剪和调整大小
                    # cropped_img = frame[int(y1)-10:int(y2)+10, int(x1)-10:int(x2)+10]
                    by=(int(y2)-int(y1))//20
                    bx=(int(x2)-int(x1))//20
                    cropped_img = frame[int(y1)-by:int(y2)+by, int(x1)-bx:int(x2)+bx]

                    

                    if cropped_img.size > 0:  # 检查是否裁剪成功

                        print(f"{img_name}--{count} crooped success x:{ int(x1),int(x2)},y:{int(y1),int(y2)}")
                        #================================================================
                        # gray=cv2.cvtColor(cropped_img,cv2.COLOR_BGR2GRAY)
                        # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#大津
                        
                        # Sobel 算子                                               #可以用，但是要寫新的算法
                        # sobelx = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
                        # sobely = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
                        # sobel = cv2.magnitude(sobelx, sobely)
                        # sobel = np.uint8(np.absolute(sobel))
                        # binary=sobel.copy()

                        # binary_3c = cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)
                        #================================================================

                        resized_img = cv2.resize(cropped_img, crop_size)
                        # resized_img = cv2.resize(binary_3c, crop_size)

                        #================================================================使用process_hole_image()來校準圓形
                        # resized_img = process_hole_image(cropped_img,crop_size)
                        # if resized_img is None:
                        #     print("process_hole_image_error, continue to next loop...")
                        #     continue
                        #================================================================
                            #原本的算法
                        # check_flag,score=is_this_hole_processed(resized_img)
                        # if check_flag:
                        #     cv2.putText(resized_img, f"T :{int(score)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
                        # elif check_flag == False and score>=70:
                        #     cv2.putText(resized_img, f"F :{int(score)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
                        # else:
                        #     cv2.putText(resized_img, f"N :{int(score)}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 2)
                        #================================================================

                        score = image_match_score_with_SIFT(resized_img)
                        
                        if score>50:
                            cv2.putText(resized_img, f"T :{score:.2f}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
                        else:
                            cv2.putText(resized_img, f"F :{score:.2f}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)

                        # cv2.putText(resized_img, "OK", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1)
                        cv2.putText(resized_img, f"{img_name[:-4]}", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                        if score !=None:
                            collected_images.append(resized_img)  # 保存处理后的图像，除了0分的

            # 显示图像，窗口标题为图像名
            # cv2.imshow(f"Object_{img_name}", frame)

            # 等待按键
            if cv2.waitKey(100) & 0xFF == ESC_KEY:  # 每张显示 100 毫秒
                break

    # 创建最终图像
    max_width = crop_size[0]*10
    images_per_row = 10
    rows = []
    current_row = []

    for i, img in enumerate(collected_images):
        if img.shape[0] == crop_size[0] and img.shape[1] == crop_size[1]:  # 确保图像为 crop_size
            current_row.append(img)
            if len(current_row) == images_per_row or i == len(collected_images) - 1:
                # 检查当前行的图像形状
                if current_row:
                    row_image = cv2.hconcat(current_row)
                    # 如果合并后的图像宽度小于500，填充到500宽
                    if row_image.shape[1] < max_width:
                        padding = np.zeros((crop_size[1], max_width - row_image.shape[1], 3), dtype=np.uint8)
                        row_image = np.hstack((row_image, padding))
                    rows.append(row_image)
                current_row = []

    # 检查 rows 是否为空
    if len(rows) > 0:
        # 检查每个行图像的形状和类型
        for idx, row in enumerate(rows):
            print(f"Row {idx} shape: {row.shape}, type: {row.dtype}")

        # 将所有行合并成最终图像
        final_image = cv2.vconcat(rows)

        # 显示最终图像
        cv2.imshow("Final Image", final_image)
        cv2.imshow("SIFt Image", sift_image)


        # 保存最终图像
        # cv2.imwrite(f"final_image_{model_ver}.png", final_image)
    else:
        print("没有找到任何有效的图像。")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
