import cv2
import sys
import time
import math
import torch
import random
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from scipy.spatial import Delaunay

global WORK_STAGE, CHANGE_BOX
global CHANGE_LAYER
CHANGE_LAYER = False
CHANGE_BOX = False
work_stage_log_path="C:/Users/Peiteng.Chuang/Desktop/work_stage.log"     # 定義work_strage.log路境

ESC_KEY = 27

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ultralytics_path = "C:/project_file/ultralytics_v11"

egdt_model_ver = "v13"
egdt_model = YOLO(f'{ultralytics_path}/runs/detect/sp_egdt_{egdt_model_ver}/weights/best.pt', verbose=False)
print(f"egdt_Model loaded successfully! model version: {egdt_model_ver}")
egdt_model = egdt_model.to(device)

det_model_ver = "v12"
det_model = YOLO(f'{ultralytics_path}/runs/detect/sp_holes_{det_model_ver}/weights/best.pt', verbose=False)
print(f"det_Model loaded successfully! model version: {det_model_ver}")
det_model = det_model.to(device)


# 設置解析度 (超過會預設相機最大解析度)
resolution_w, resolution_h = 1920, 1080
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_h)

#================================================================================================
#偵測改變與比對改變區域並辨識的程式功能區
def det_model_getbox(frame, model=det_model):
    results = model(frame, verbose=False)
    img_boxes = []
    for result in results:
        boxes = result.boxes  # 获取检测框
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
            img_boxes.append((x1, y1, x2, y2))

    return frame,img_boxes

def have_edge(cropped_img, score_threshold=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], th_lv=5):     #th_lv有10階，LV數字越小，對精準要求越高

    results = egdt_model(cropped_img, verbose=False)
    egdt_flag=False

    if results:
          
        for result in results:
            boxes = result.boxes  # 获取检测框
            for box in boxes:
                score = box.conf[0].item()  # 获取置信度分數
                for i in range(th_lv):

                    egdt_flag=True if score >= score_threshold[i] else False

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
                    cropped_img = cv2.rectangle(cropped_img, (int(x1), int(y1)), (int(x2), int(y2)),(0,255,0), 1)
                
        return cropped_img,egdt_flag
    else:            
        cropped_img = cv2.putText(cropped_img, f"O", (5,5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        return cropped_img,egdt_flag

def dis(p1, p2):
    x1, y1 = p1[0],p1[1]
    x2, y2 = p2[0],p2[1]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    print(f"x1,y1={x1},{y1},x2,y2={x2},{y2},d={distance}")
    return distance


def get_image_difference(prev_boxes, current_boxes, image):

    # 儲存兩張圖片中的檢測框
    boxes_img1 = prev_boxes  # 過去圖片的檢測框
    boxes_img2 = current_boxes  # 目前圖片的檢測框


    # 判斷新增或移動的物件
    def is_different(box, boxes_reference, threshold=5):
        """檢查當前檢測框是否與參考框清單中的任一框不匹配"""
        
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 計算中心點
        for ref_box in boxes_reference:
            rx1, ry1, rx2, ry2 = ref_box
            rcx, rcy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
            # if abs(cx - rcx) < threshold and abs(cy - rcy) < threshold:
            #     return False  # 與參考框匹配
            p1=(cx,cy)
            p2=(rcx,rcy)
            dt=dis(p1,p2)
            if dt < threshold :
                print(f"匹配:dis((cx,cy), (rcx,rcy))={dt}")
                return False  # 與參考框匹配
            
        return True  # 不匹配，視為新增或移動的物件

    # 找出在 img2 中新增或移動的框
    different_boxes = [box for box in boxes_img2 if is_different(box, boxes_img1, threshold=7)] #抓出所有不匹配的box，再以have_edge過濾鍵槽
    new_box_number=len(different_boxes)
    # 在 img2 上繪製標記
    for x1, y1, x2, y2 in different_boxes:
        cropped_img = image[int(y1):int(y2), int(x1):int(x2)]
        egdt_img, egdt_flag = have_edge(cropped_img.copy())#套用have_edge功能來跑模型
        if egdt_flag:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 綠色框表示GOOD物件
            cv2.putText(image, "Good", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 紅色框表示BAD物件
            cv2.putText(image, "Bad", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image,new_box_number
#================================================================================================
#給點點加上index的程式

#================================================================================================
#判斷分層程式
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
            green_count += 3    #每個符合條件的三角形有三條綠線
            green_score += 1    #綠線分數+1

        else:
            red_score += 1      #紅線分數+1   

    # layer_th=[12,83,142,234,300]#手動設定驗證分層，資料夾驗證專用
    layer_th2=[92.5,98.5,105,113,123,136]#手動設定門檻值分層，先驗知識
    
    if green_count > 0:
        score=green_score/99*100
        avg_green_length = total_green_length / green_count

        slope_score=calculate_slope_score(points)

        avg_score= 100*(1-abs(layer_th2[current_layer]-avg_green_length)/avg_green_length)
        if avg_green_length>layer_th2[current_layer]+1:
            avg_score=100
        # cl_score=(avg_score+ score)/2
        
        total_score=(score+avg_score+slope_score)/4

        print(f"Avg_length : {avg_green_length:6.2f} , G/R : {green_score:3}/{red_score:2}, avg score : {avg_score:7.3f}, green line score : {score:7.3f}, slope_score : {slope_score:6.2f}, total score : {total_score}")

        if score>=98 and avg_score>=99.3 and slope_score>=195 and total_score>=98.7:        #綜合分數高於98.7
            print(f"### total_score={total_score}, Changing layer , current layer : {current_layer+1} > {current_layer+2} ###")
            CHANGE_LAYER=True
            if current_layer+1<len(layer_th2):
                current_layer+=1
            else:
                print("已達到最高層次，你的演算法爆掉了")
                return 0
    else:
        print("沒有符合條件的正三角形（綠線）。")
    return total_score ,current_layer

#================================================================================================
def read_workstage_log(log_file_path = work_stage_log_path):

    '''讀取 work_stage.log 檔案，取得最後一筆記錄的 box, layer 和 items_count並回傳。
    每一條log格式 : time=20250102-133956,box=1,layer=1,items_count=63,change_box=0
    Returns:
        tuple: (box, layer, items_count) 最後更新的數據。'''

    try:
        with open(log_file_path, "r") as file:
            lines = file.readlines()

        if not lines:
            raise ValueError("Log file is empty.")

        last_line = lines[-1].strip()
        parts = last_line.split(",")

        data = {}
        for part in parts:
            key, value = part.split("=")
            data[key.strip()] = value.strip()

        box = int(data["box"])
        layer = int(data["layer"])
        items_count = int(data["items_count"])
        change_box_flag = True if int(data["change_box"])==1 else False
        return box, layer, items_count, change_box_flag

    except FileNotFoundError:
        print(f"Error: {log_file_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None

WORK_STAGE = read_workstage_log()   #讀取log

def get_date_string():
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")

def update_workstage_log(box,layer,count,change_box,path=work_stage_log_path):

    now_date = get_date_string()
    if change_box==True:
        cb_flag=1 
    else:
        cb_flag=0
    write_str=f"write {now_date},{box},{layer},{count},{cb_flag}"
    
    with open(path, 'a') as f:
        if change_box:
            print(f"box{box} has been loaded completely, waiting for the box to be changed...") # waiting for the box to be changed...
            f.write(f"time={now_date},box={box},layer={layer},items_count={count},change_box={cb_flag}\n")
        else:
            print(write_str)
            f.write(f"time={now_date},box={box},layer={layer},items_count={count},change_box={cb_flag}\n")

def detect_new_box():
    global CHANGE_BOX
    if CHANGE_BOX==True:    
        CHANGE_BOX=random.choices([True, False], weights=[95, 5], k=1)[0]
        if CHANGE_BOX==False:
            print(f"Change complete. Starting detect new box.")
        return CHANGE_BOX
    else:
        return CHANGE_BOX
    
def get_change_box_flag():
    global CHANGE_BOX
    return CHANGE_BOX

def switch_change_box_flag():
    global CHANGE_BOX
    if CHANGE_BOX==False:
        CHANGE_BOX=True   
    else:
       CHANGE_BOX=False

def is_in_time_range(work_time=(8,12,13,17)):       #確認工作時間
    current_time = datetime.now()
    # 設置工作時間範圍 08:00-12:00 和 13:00-17:00
    if (current_time.hour >= work_time[0] and current_time.hour < work_time[1]) or (current_time.hour >= work_time[2] and current_time.hour < work_time[3]):
        return True
    return False

def calculate_mse(frame1, frame2):
    return np.mean((frame1.astype("float") - frame2.astype("float")) ** 2)
#================================================================================================

#full-layer存起來的函數(含計數)
def take_picture_and_save_to(path,image):
    date_string=get_date_string()
    file_name = path+f'{date_string}.png'
    cv2.imwrite(file_name, image)
    return
#================================================================================================


def main():
    #================================================================================================
    #確保log正確載入
    if WORK_STAGE is not None:
        print(WORK_STAGE)
        print(f"work_stage最後更新的數據為: Box={WORK_STAGE[0]}, Layer={WORK_STAGE[1]}, Items Count={WORK_STAGE[2]}")
    else:
        print("Work stage not found. Stopping process")
        sys.exit(1)
    #================================================================================================
    
    if not cap.isOpened():
        print("無法開啟相機")
        return

    #判斷式:mse篩選工人比對+多工計時拍照(60秒一張)
    box,layer,items_count = WORK_STAGE[0],WORK_STAGE[1],WORK_STAGE[2]

    time.sleep(2)
    for i in range(5):
        cap.read()

    last_action_time = time.time()  # 記錄最後一次動作的時間
    last_frame = None  # 記錄Live的最後一幀
    #設置一些初始參數
    interval = 10           # 每隔幾秒拍一次照
    prev_image = None       # 隔n秒前拍的那一張照片
    prev_boxes =None        # 前一張照片的掃描結果
    prev_points = []

    current_image=None      # 當前照片的存放
    current_boxes=0         # 當前照片的掃描結果
    
    mse_threshold =250      #前後圖片mse低於n才算是停止
    mse_count=0             #mse的計數暫存變數
    mse_count_frame=10      #在超過閾值n次後才算是stop畫面，可以開始拍照

    while True:
        global CHANGE_LAYER
        current_points=[]
        # 檢查是否在允許的時間範圍內
        if not is_in_time_range():
            # 若不在時間範圍內，暫停一段時間再檢查
            time.sleep(60)  # 每分鐘檢查一次
            continue

        ret, frame = cap.read()
        if not ret:
            print("無法接收畫面 (相機已關閉)")
            break

        #這邊是計時程式
        current_time = time.time()
        elapsed_time = current_time - last_action_time
        if elapsed_time >= interval:
            
            if last_frame is not None:

                mse = calculate_mse(last_frame, frame)
                print(f"MSE: {mse}")

                if mse < mse_threshold:  # 您可以釋述MSE開關
                    mse_count+=1
                    print(f"mc={mse_count}")

                    if mse_count>mse_count_frame:     #在超過閾值n次後才算是stop畫面，可以開始拍照
                        mse_count=0

                        print("Performing detection task...")   #已排除移動物體，開始掃描物件
                        last_action_time = current_time  # 更新最後一次執行時間
                        current_image=frame.copy()
                        date_string = get_date_string()

                        p1_img,current_boxes=det_model_getbox(current_image)
                        current_points=boxes2points(current_boxes)
                        display_image = p1_img

                        if prev_image is not None:
                            display_image,new_b_n = get_image_difference(prev_boxes, current_boxes, p1_img) #以兩組boxes數據計算出差異，這個boxes可以拿去Delaunay
                            print(f"new box : {new_b_n}")

                        if current_points is not None  and len(current_points)>=3:          #取得current_points數據，開始加工
                            # print(f"current_points detected: {current_points}")
                            for point in current_points:
                                display_image=cv2.circle(display_image,point, 3, (0, 255, 0), -1)
                            # temp_layer=current_layer.copy()
                            # print(f"temp_layer detected: {temp_layer}")
                            cl_score ,current_layer=delaunay_change_layer(current_points,current_layer)             #cl_score=change_layer_score(裡面是 total_score )
                            if CHANGE_LAYER:
                                take_picture_and_save_to("C:/project_file/AI_Supervise_processing/saved_img/layer_full/",display_image)
                                CHANGE_LAYER=False
                            print(f"change_layer_score={cl_score:6.3f}, current_layer={current_layer+1}")

                        cv2.imshow("Diff Viewer", display_image)
                        #假裝拍照
                        cv2.imshow('Origin pic', frame)
                        print(f"照片'{date_string}'已拍攝")
                else:
                    print(f"mc reset")
                    mse_count=0


            prev_image = current_image
            prev_boxes = current_boxes
            prev_points = current_points        #設置prev_points，讓它可以找出孔洞數差異

        last_frame=frame.copy()    
        countdown =interval-elapsed_time
        resized = cv2.resize(frame.copy(), (960, 540))
        if countdown>=10 and countdown >=0:#倒數
            resized = cv2.putText(resized,f"{countdown:.1f}",(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )
        elif countdown<0:       #超時，等待穩定
            resized = cv2.putText(resized,f"wait untill stable...",(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )
        else:
            resized = cv2.putText(resized,f"{countdown:.1f}",(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )
        
        cv2.imshow('Live Camera', resized)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        #================================================================================================
        # 換箱 & 紀錄(在測試計時系統時暫時關閉)
        # change_box_flag=get_change_box_flag()           #抓取change_box_flag，True/False
        # if change_box_flag==False:
                                            
        #     if True:                                        #辨識到差異=true，這裡要跟get_image_difference()的box數整合
        #         items_count+=1                              #*****判斷 : 偵測到物件+1*****

        #     update_workstage_log(box,layer,items_count,change_box_flag) #先紀錄

        #     if items_count//65==layer:             #判斷滿層+1 或是 滿箱>>進入換箱辨識
        #         if layer==6:
        #             print("full, switch flag")
        #             switch_change_box_flag()        #切換flag，進入換箱模式
        #         else:
        #             layer+=1
        # else:
        #     change_box_flag=detect_new_box()        #當偵測到換完新箱子時才會回傳False，換箱子時是True
        #     if change_box_flag==False:              #換箱完畢，重置數據
        #         box+=1
        #         layer=1
        #         items_count=0
        #     update_workstage_log(box,layer,items_count,change_box_flag)


#整合程式以及主要執行
# 1.讀取之前的log紀錄，抓取目前箱數、層數、零件數(?)
# 2.yolo detect 洞洞計數
# 3.綜合算法分層(連線分數、先驗分數)
# 4.洞洞EGDT
# 5.分配&計算位置(1-1~10-6)
# 5-1.結合(3)(4)(5)辨識，滿層拍照記錄
# 5-2.使用(3)+diff程式來找出層數、目前出錯位置
# 6.根據(4)，錯誤佔辨識80% or 辨識過少、錯誤過多>>換箱辨識(掃黃)
# 6-1.if 換箱，box+1,layer=1,items_count=0，(但是何時開始生產?如何確定開始?攝影機開始時的判斷)
# 7.紀錄log，箱數、層數、零件數

if __name__=="__main__":
    main()
