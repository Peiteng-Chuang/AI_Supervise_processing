import os
import cv2
import torch
import numpy as np
from datetime import datetime
from scipy.spatial import Delaunay
from ultralytics import YOLO
from img_delaunay_test import sort_points_with_skew_correction,find_points_numbers_robust

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ultralytics_path = "C:/project_file/ultralytics_v11"

# # 加载邊角檢測模型
egdt_model_ver = "v13"
egdt_model = YOLO(f'{ultralytics_path}/runs/detect/sp_egdt_{egdt_model_ver}/weights/best.pt', verbose=False)
print(f"egdt_Model loaded successfully! model version: {egdt_model_ver}")
egdt_model = egdt_model.to(device)

# 加载孔洞檢測模型
det_model_ver = "v12"
det_model = YOLO(f'{ultralytics_path}/runs/detect/sp_holes_{det_model_ver}/weights/best.pt', verbose=False)
print(f"det_Model loaded successfully! model version: {det_model_ver}")
det_model = det_model.to(device)

global CHANGE_LAYER
CHANGE_LAYER = False



#============================================================================================================================slope
def boxes2points(boxes):
    points = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        points.append((int(cx), int(cy)))
    return points

def calculate_slope_score(points):
    """
    Calculate a slope-based score for a set of 2D points.
    The score is based on the uniformity of slopes for the top four and bottom four closest points.

    Args:
        points (list of tuple): List of (x, y) coordinates.

    Returns:
        float: Total score combining top and bottom slope uniformity.
    """
    # Convert points to a NumPy array for easier manipulation
    points = np.array(points)

    # Sort points by y-coordinate, then by x-coordinate as a tie-breaker
    sorted_points = points[np.lexsort((points[:, 0], points[:, 1]))]

    def calculate_slopes(points_subset):
        """Calculate slopes and evaluate uniformity score."""
        num_points = len(points_subset)
        slopes = []

        # Calculate slopes between each pair of points
        for i in range(num_points):
            for j in range(i + 1, num_points):
                dx = points_subset[j][0] - points_subset[i][0]
                dy = points_subset[j][1] - points_subset[i][1]
                if dx != 0:  # Avoid division by zero
                    slopes.append(dy / dx)

        # Calculate uniformity score
        slopes = np.array(slopes)
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)

        # Penalize scores based on the spread (standard deviation)
        score = 100 / (1 + std_slope)  # Higher std -> lower score

        return score, slopes

    # Step 1: Get top 4 points (lowest y-values)
    top_points = sorted_points[:5]
    top_score, top_slopes = calculate_slopes(top_points)

    # Step 3: Get bottom 4 points (highest y-values)
    bottom_points = sorted_points[-5:]
    bottom_score, bottom_slopes = calculate_slopes(bottom_points)

    # Step 5: Combine scores
    total_score = top_score + bottom_score

    return total_score

def delaunay_change_layer(points,current_layer):

    global CHANGE_LAYER

    points_np = np.array(points, dtype=np.int32)
    tri = Delaunay(points_np)
    #計算綠線
    total_green_length = 0  # 紀錄綠線總長度
    green_count = 0         # 紀錄綠線的總數
    green_score =0
    red_score =0
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
        
        if (max_edge - min_edge) / max_edge <= 0.26:  # 如果邊長差異在 n% 內

            total_green_length += sum(edge_lengths)  # 累加符合條件的三邊長
            green_count += 3  # 每個符合條件的三角形有三條綠線
            green_score += 1

        else:
            red_score += 1

    layer_th=[12,83,142,234,300]#手動設定驗證分層
    layer_th2=[92.5,98.5,105,113,123,136]#先驗知識
    
    if green_count > 0:
        g_score=green_score/99*100
        avg_green_length = total_green_length / green_count

        slope_score=calculate_slope_score(points)

        avg_score= 100*(1-abs(layer_th2[current_layer]-avg_green_length)/avg_green_length)
        if avg_green_length>layer_th2[current_layer]+1:
            avg_score=100
        cl_score=(avg_score+ g_score)/2
        
        total_score=(g_score+avg_score+slope_score)/4

        print(f"Avg_length : {avg_green_length:6.2f} , G/R : {green_score:3}/{red_score:2}, avg score : {avg_score:7.3f}, green line score : {g_score:7.3f}, slope_score : {slope_score:6.2f}, total score : {total_score}")

        if g_score>=98 and avg_score>=99.3 and slope_score>=195 and total_score>=98.7:        #綜合分數高於98.7
            print(f"### cl_score={cl_score}, Changing layer , current layer : {current_layer+1} > {current_layer+2} ###")
            CHANGE_LAYER = True
            if current_layer+1<len(layer_th2):
                current_layer+=1
            else:
                print("已達到最高層次，你的演算法爆掉了")
                return 0
    else:
        print("沒有符合條件的正三角形（綠線）。")
    return cl_score ,current_layer


#============================================================================================================================
def det_model_draw(frame, model=det_model):
    results = model(frame, verbose=False)
    boxes_img = []
    for result in results:
        boxes = result.boxes  # 获取检测框
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
            boxes_img.append((x1, y1, x2, y2))

            # green = (0, 255, 0)
            # red = (0, 0, 255)
            # # 使用邊框加大來確認是否為邊緣圖像
            # by = ((y2 - y1) / 10)
            # bx = ((x2 - x1) / 10)
            # yb1, yb2 = y1 - by, y2 + by
            # xb1, xb2 = x1 - bx, x2 + bx
            # if yb1 < 0 or yb2 > frame.shape[0] or xb1 < 0 or xb2 > frame.shape[1]:
            #     continue  # 跳過出界的检测框
            # cropped_img = frame[int(yb1):int(yb2), int(xb1):int(xb2)]
            # cpisz = cropped_img.size
            # if cpisz > 0:  # 检查裁剪是否成功
            #     # 根據裁剪圖像大小選擇最佳縮放方法
            #     if cpisz >= 100 * 100:  # 确保图像尺寸足够大
            #         resized_img = cv2.resize(cropped_img, (100, 100), interpolation=cv2.INTER_AREA)
            #     else:
            #         resized_img = cv2.resize(cropped_img, (100, 100), interpolation=cv2.INTER_CUBIC)
            #     egdt_img, egdt_flag = have_edge(resized_img)
            #     # 画出检测框
            #     if egdt_img is not None:
            #         if not egdt_flag:
            #             cv2.putText(frame, f"O", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
            #             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
            #         if egdt_flag:
            #             cv2.putText(frame, f"Q", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
            #             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), green, 2)
    # 顯示即時影像
    # cv2.imshow("Detection Results", frame)
    return frame,boxes_img

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
                    cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,0), 1)

                elif score >= score_threshold[2]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    egdt_flag=True
                    cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(255,255,0), 1)

                elif score >= score_threshold[1]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    egdt_flag=True
                    cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,255), 1)

                elif score >= score_threshold[0]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    # egdt_flag=True
                    # cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,125,255), 2)
                else:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    # cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,0,255), 2)
        return cropped_img,egdt_flag
    else:            
        cropped_img = cv2.putText(cropped_img, f"O", (5,5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        return cropped_img,egdt_flag

def find_points_numbers(points, different_points, tolerance=15):
    """
    將點按網格模式排序並返回指定點的編號。
    
    Args:
        points (list[tuple]): 原始的點座標列表。
        different_points (list[tuple]): 需要查找編號的點座標列表。
        tolerance (int): x 坐標分組的誤差範圍，默認為 10。
        
    Returns:
        list[tuple]: 每個目標點的 (num, x, y)，其中 num 是編號。
    """
    # 將 points 分組處理
    points_grouped = {}
    for i, (x, y) in enumerate(points):
        group_key = round(x / tolerance) * tolerance  # 分組鍵值
        if group_key not in points_grouped:
            points_grouped[group_key] = []
        points_grouped[group_key].append((x, y))

    # 排序每個組內的點 (按 y 坐標遞減)
    for key in points_grouped:
        points_grouped[key].sort(key=lambda p: -p[1])  # 按 y 坐標遞減排序

    # 排序所有分組鍵並合併
    sorted_points = []
    for key in sorted(points_grouped.keys()):  # 按 x 分組鍵排序
        sorted_points.extend(points_grouped[key])  # 合併結果

    # 給每個點編號
    sorted_points_with_index = [(i + 1, coord[0],coord[1]) for i, coord in enumerate(sorted_points)]

    # 查找 different_points 的編號
    different_points_num = []
    for diff_point in different_points:
        for idx, coord_x,coordy in sorted_points_with_index:
            if (coord_x,coordy) == diff_point:
                different_points_num.append((idx, coord_x,coordy))
#                 Different points with numbers:
#                   (41, 1068, 777)
#                   (47, 1140, 823)
#                   (60, 1295, 835)
                break
        else:
            different_points_num.append((None, diff_point[0], diff_point[1]))  # 如果沒找到，返回 None
    # print(sorted_points_with_index)
    return different_points_num,sorted_points_with_index

def get_image_difference(prev_boxes, current_boxes, image, current_layer):
    
    # 儲存兩張圖片中的檢測框
    boxes_img1 = prev_boxes  # 過去圖片的檢測框
    boxes_img2 = current_boxes  # 目前圖片的檢測框


    # 判斷新增或移動的物件
    def is_different(box, boxes_reference, threshold=5):
        """檢查當前檢測框是否與參考框清單中的任一框不匹配"""
        #================================================================
        #測試計算match數與new數
        match_items=0
        new_items=0
        #================================================================
        #找出prev_boxes被current_boxes覆蓋的點
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 計算中心點
        for ref_box in boxes_reference:
            rx1, ry1, rx2, ry2 = ref_box
            rcx, rcy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
            if abs(cx - rcx) < threshold and abs(cy - rcy) < threshold:
                return False  # 與參考框匹配，只需與1個點匹配就達成目的
        return True  # 不匹配，視為新增或移動的物件

    # 找出在 img2 中新增或移動的框
    different_boxes = [box for box in boxes_img2 if is_different(box, boxes_img1, threshold=5)]

    different_points = boxes2points(different_boxes)

    print(f"new = {len(different_boxes)} box and {len(different_points)} diff points.")
    #轉換點點，使用點點，成為點點
    layer_num_gap_th=[92.5,98.5,105,113,123,136]#先驗知識
    tolerance_cal=layer_num_gap_th[current_layer]//2-10

    points=boxes2points(boxes_img2)

    different_points_num,spwi=find_points_numbers(points,different_points,tolerance=tolerance_cal)
    # different_points_num,spwi=sort_points_with_skew_correction(points,different_points)
    # different_points_num,spwi=find_points_numbers_robust(points,different_points)


    # print(different_points_num)
    for idx, locy, locx in different_points_num:
        print(f"new = {current_layer+1} - {idx} ,{(locy, locx)}")
        

    # 在 img2 上繪製標記
    for x1, y1, x2, y2 in different_boxes:
        cropped_img = image[int(y1):int(y2), int(x1):int(x2)]
        egdt_img, egdt_flag = have_edge(cropped_img)
        if egdt_flag:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 綠色框表示GOOD物件
            cv2.putText(image, "Good", (int(x1)+10, int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 紅色框表示BAD物件
            cv2.putText(image, "Bad", (int(x1)+10, int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # for idx, locx, locy in different_points_num:
        #     if x1<locx<x2 and y1<locy<y2 :
        #         cv2.putText(image, f"{idx}", (int(x1)-10, int(y1)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    for idx, locy,locx in spwi:
            cv2.putText(image, f"{idx}", (locy-10,locx-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    return image


#================================================================
def get_date_string():
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")
#存起來的函數
def take_picture_and_save_to(path,image):
    date_string=get_date_string()
    file_name = path+f'{date_string}.png'
    cv2.imwrite(file_name, image)
    return
#================================================================


def main(folder_path):
    """
    主函式：讀取資料夾中的圖片並依序顯示。
    """
    global CHANGE_LAYER
    # 獲取資料夾中的圖片並排序
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_files.sort()

    if not image_files:
        print("資料夾中沒有圖片檔案！")
        return

    # 初始化變數
    prev_image = None
    prev_boxes =None
    current_boxes=0
    index = 0

    current_box=1
    current_layer=0     #0-5以index算，所以輸出時要記得+1

    while index < len(image_files):
        image_path = os.path.join(folder_path, image_files[index])
        current_image = cv2.imread(image_path)
        height, width, channels = current_image.shape

        p1_img,current_boxes=det_model_draw(current_image.copy())       #det_model_draw從if移出
        points = boxes2points(current_boxes)                            #計算points
            
        if prev_image is not None:
            display_image = get_image_difference(prev_boxes, current_boxes, p1_img,current_layer)
        else:
            display_image = p1_img

        if points is not None and len(points)>=3:
            for point in points:        #對得出的points數據加工
                display_image=cv2.circle(display_image,point,3,(0,255,0),-1)
            
            

            cl_score ,current_layer=delaunay_change_layer(points,current_layer)     #cl_score=change_layer_score，delaunay必須至少有3個點
            print(f"change_layer_score={cl_score:6.3f}, current_layer={current_layer+1}")
            if CHANGE_LAYER:
                take_picture_and_save_to("C:/project_file/AI_Supervise_processing/saved_img/layer_full/",display_image)
                CHANGE_LAYER=False


        display_image=cv2.resize(display_image,(width//3*2,height//3*2))
        cv2.imshow("Image Viewer", display_image)

        key = cv2.waitKey(0)

        if key == 27:  # 按下 ESC 鍵中斷
            print("程式中斷。")
            break
        elif key == 32:  # 按下空白鍵切換
            prev_image = current_image
            prev_boxes = current_boxes
            index += 1
        else:
            print("無效按鍵，請按空白鍵切換或 ESC 鍵中斷。")

    print("所有圖片已顯示完成。")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = "C:/Users/Peiteng.Chuang/Desktop/cb/sample"  #C:/Users/Peiteng.Chuang/Desktop/diff_good
    # folder_path = "C:/Users/Peiteng.Chuang/Desktop/cb/change_box_test_2"  #C:/Users/Peiteng.Chuang/Desktop/diff_good
    main(folder_path)