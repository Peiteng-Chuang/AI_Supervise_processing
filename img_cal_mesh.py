
import cv2
import torch
import numpy as np
import os, random
from ultralytics import YOLO
from scipy.spatial import Delaunay

ESC_KEY = 27

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ultralytics_path = "C:/project_file/ultralytics_v11"

# 加载邊角檢測模型
egdt_model_ver = "v13"
egdt_model = YOLO(f'{ultralytics_path}/runs/detect/sp_egdt_{egdt_model_ver}/weights/best.pt', verbose=False)
print(f"egdt_Model loaded successfully! model version: {egdt_model_ver}")
egdt_model = egdt_model.to(device)

# 加载孔洞檢測模型
det_model_ver = "v12"
det_model = YOLO(f'{ultralytics_path}/runs/detect/sp_holes_{det_model_ver}/weights/best.pt', verbose=False)
print(f"det_Model loaded successfully! model version: {det_model_ver}")
det_model = det_model.to(device)


def have_edge(cropped_img):

    results = egdt_model(cropped_img, verbose=False)
    egdt_flag=False
    score_threshold=[0.4,0.5,0.6,0.7]
    if results:
          
        for result in results:
            boxes = result.boxes  # 获取检测框
            for box in boxes:
                score = box.conf[0].item()  # 获取置信度分數
                if score >= score_threshold[3]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    egdt_flag=True
                    cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,0), 2)

                elif score >= score_threshold[2]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    egdt_flag=True
                    cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(255,255,0), 2)
                elif score >= score_threshold[1]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    egdt_flag=True
                    cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,255), 2)
                elif score >= score_threshold[0]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    # egdt_flag=True
                    # cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,125,255), 2)
                else:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    # cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,0,255), 2)
        return cropped_img,egdt_flag
                
    cropped_img = cv2.putText(cropped_img, f"O", (5,5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    return cropped_img,egdt_flag

def do_otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bgr_thresh=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    # cv2.imshow("otsu_img",bgr_thresh)#驗證
    # print(bgr_thresh.shape)
    return bgr_thresh

def random_img(path):
    # 確保路徑存在
    if not os.path.exists(path):
        raise ValueError(f"The specified path does not exist: {path}")
    
    # 過濾圖片檔案類型
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    img_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and os.path.splitext(f)[1].lower() in img_extensions]
    
    # 如果資料夾中沒有圖片
    if not img_files:
        raise ValueError("No image files found in the specified directory.")
    
    # 隨機選擇一個圖像檔名
    return random.choice(img_files)


def main():

    pth="./saved_img/old_data/"#"C:/Users/Peiteng.Chuang/Desktop/perfect_grid/"
    # f_name=random_img(pth)
    # f_name =  "20241126-134947_img42.jpg" #(排一點/左邊)
    f_name =  "20241127-164551_img545.jpg" #(排一半/中間)
    # f_name =  "20241127-115046_img139.jpg" #(接近排滿/右邊)
    # f_name = "20241127-094357_img13.jpg" #排滿
    img_path = pth+f_name  # 替换为你的图像路径
    print(f"***{img_path}***")

    frame = cv2.imread(img_path)
    height, width, channels = frame.shape

    img_rate=1920
    retangle_rate=20     #多出原圖2/n的大小

    width_set=img_rate
    height_set=int(height/width*img_rate)
    frame = cv2.resize(frame,(width_set,height_set),interpolation=cv2.INTER_AREA)

    crop_size = (128, 128)
    collected_images = []
    points = []

    # 进行物体检测
    results = det_model(frame, verbose=False)

    for result in results:
        boxes = result.boxes  # 获取检测框
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
            green = (0, 255, 0)
            red = (0, 0, 255)
            yellow = (0, 255, 255)

            points.append(((x1+x2)//2,(y1+y2)//2))

            # 使用邊框加大來確認是否為邊緣圖像
            by = ((y2 - y1) / retangle_rate)
            bx = ((x2 - x1) / retangle_rate)
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

                
                egdt_img,egdt_flag = have_edge(resized_img.copy())
                # egdt_img,egdt_flag = have_edge(do_otsu(resized_img.copy()))


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
                    collected_images.append(egdt_img)

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


    # if points is not None:

    #     points_np = np.array(points, dtype=np.int32)
    #     tri = Delaunay(points_np)

        
    #     image_delaunay = np.zeros((height, width, 3), dtype=np.uint8)

    #     for simplex in tri.simplices:
    #         pts = points_np[simplex]
    #         cv2.polylines(image_delaunay, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
        

    #     image_delaunay=cv2.resize(image_delaunay,(image_delaunay.shape[1]//2,image_delaunay.shape[0]//2))   #把image_delaunay變成比較好顯示的大小
    #     cv2.imshow("Delaunay image", image_delaunay)
    if points is not None:
        points_np = np.array(points, dtype=np.int32)
        tri = Delaunay(points_np)
    
        image_delaunay = np.zeros((height, width, 3), dtype=np.uint8)

        for simplex in tri.simplices:
            pts = points_np[simplex]

            # 計算三條邊的長度
            edge_lengths = [
                np.linalg.norm(pts[0] - pts[1]),
                np.linalg.norm(pts[1] - pts[2]),
                np.linalg.norm(pts[2] - pts[0])
            ]

            # 判斷是否接近正三角形
            max_edge = max(edge_lengths)
            min_edge = min(edge_lengths)
            for edge_length in edge_lengths:
                if edge_length != max_edge and edge_length != min_edge:
                    mid_edge = edge_length
            if mid_edge is None:
                print("mid_edge is None")
                break
            # if (max_edge - min_edge) / max_edge <= 0.25:  # 如果邊長差異在 n% 內
            #     color = (0, 255, 0)  # 綠色
            # else:
            #     color = (0, 0, 255)  # 紅色

            # # 繪製三條邊
            # cv2.line(image_delaunay, tuple(pts[0]), tuple(pts[1]), color, thickness=3)
            # cv2.line(image_delaunay, tuple(pts[1]), tuple(pts[2]), color, thickness=3)
            # cv2.line(image_delaunay, tuple(pts[2]), tuple(pts[0]), color, thickness=3)
            #================================================================================================================================case2:只畫綠線分離
            if (max_edge - min_edge) / max_edge <= 0.15:  # 如果邊長差異在 n% 內
                color = (0, 255, 0)  # 綠色
                # 繪製三條邊
                cv2.line(image_delaunay, tuple(pts[0]), tuple(pts[1]), color, thickness=3)
                cv2.line(image_delaunay, tuple(pts[1]), tuple(pts[2]), color, thickness=3)
                cv2.line(image_delaunay, tuple(pts[2]), tuple(pts[0]), color, thickness=3)
            #================================================================================================================================case2:只畫綠線分離

        # 調整大小以便顯示
        image_delaunay = cv2.resize(image_delaunay, (image_delaunay.shape[1] // 2, image_delaunay.shape[0] // 2))
        # cv2.imshow("Delaunay image", image_delaunay)
        if collected_images is not None:
            frame_add=cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))
            addWeighted = cv2.addWeighted(frame_add, 0.6, image_delaunay, 0.4, 60)
            cv2.imshow("addWeighted Results", addWeighted)

    print(f"{frame.shape}")
    frame=cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))   #把frame變成比較好顯示的大小
    cv2.imshow("Detection Results", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
