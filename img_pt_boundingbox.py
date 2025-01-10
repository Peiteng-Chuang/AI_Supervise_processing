import numpy as np
import cv2
import torch
from ultralytics import YOLO
from scipy.spatial import Delaunay
import random
from scipy.spatial import cKDTree


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ultralytics_path = "C:/project_file/ultralytics_v11"

# 加载孔洞檢測模型
det_model_ver = "v12"
det_model = YOLO(f'{ultralytics_path}/runs/detect/sp_holes_{det_model_ver}/weights/best.pt', verbose=False)
print(f"det_Model loaded successfully! model version: {det_model_ver}")
det_model = det_model.to(device)


def box2point(box):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    point=(int(cx), int(cy))
    return point

def det_model_getpoints(image, model=det_model):

    results = model(image, verbose=False)
    boxes_4pt = []
    points = []
    for result in results:
        boxes = result.boxes  # 获取检测框
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 获取边界框坐标
            boxes_4pt.append((x1, y1, x2, y2))
            point=box2point((x1, y1, x2, y2))
            points.append(point)

    return boxes_4pt,points

def calculate_min_bounding_box_angle(points):
    """
    計算給定點的最小包圍方框以及旋轉角度。

    Args:
        points (list[tuple]): 點的列表，例如 [(x1, y1), (x2, y2), ...]

    Returns:
        angle (float): 方框的旋轉角度（角度制）。
        box_points (np.ndarray): 最小包圍方框的四個頂點坐標。
    """
    # 將點轉換為 NumPy 數組
    points_array = np.array(points, dtype=np.float32)

    # 計算最小包圍方框
    rect = cv2.minAreaRect(points_array)

    # 取得方框的四個頂點
    box_points = cv2.boxPoints(rect)
    box_points = np.int0(box_points)

    # 取得旋轉角度
    angle = rect[2]

    # 校正角度範圍，使其符合垂直方向
    if angle < -45:
        angle += 90

    return angle, box_points
def extend_box(box_points, n):
    """
    擴展最小包圍方框的邊界。

    Args:
        box_points (numpy.ndarray): 最小包圍方框的四個頂點座標。
        n (int): 要擴展的像素數。

    Returns:
        numpy.ndarray: 擴展後的方框頂點座標。
    """
    # 計算方框中心點
    center = np.mean(box_points, axis=0)

    # 擴展每個頂點
    extended_box = []
    for point in box_points:
        direction = point - center
        extended_point = point + n * direction / np.linalg.norm(direction)
        extended_box.append(extended_point)

    return np.array(extended_box, dtype=np.int0)

def calibrate_image_with_box(image, n=60):
    """
    校正圖像，使其根據最小包圍方框旋轉並進行擴展。

    Args:
        image (numpy.ndarray): 原始圖像。
        points (list[tuple]): 圖像中的點集。
        det_model_getpoints (function): 檢測模型返回的點。
        n (int): 擴展最小包圍方框的像素值。

    Returns:
        numpy.ndarray: 校正後的圖像。
    """
    boxes_4pt,points=det_model_getpoints(image)
    angle, box_points = calculate_min_bounding_box_angle(points)
    ex_box_pts = extend_box(box_points, n)

    print(f"旋轉角度: {angle} 度")
    print(f"最小包圍方框頂點: {box_points}")


    # 繪製擴展後的方框
    extended_box = np.array(ex_box_pts, dtype=np.float32)

    # 計算旋轉的仿射矩陣
    rect = cv2.minAreaRect(extended_box)
    box_center = rect[0]  # 中心
    box_size = rect[1]    # 寬和高
    angle = rect[2]       # 旋轉角度

    # 如果角度為負，旋轉角度加 90 度，因為cv2.minAreaRect默認逆時針
    if angle < -45:
        angle += 90

    # 獲取旋轉仿射變換矩陣
    rotation_matrix = cv2.getRotationMatrix2D(box_center, angle, scale=1.0)

    # 旋轉圖像
    height, width = image.shape[:2]
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # 剪裁旋轉後的圖像
    box_width, box_height = map(int, box_size)
    x, y = map(int, box_center)

    cropped_image = rotated_image[max(0, y - box_height // 2):y + box_height // 2, 
                                   max(0, x - box_width // 2):x + box_width // 2]
    w,h,_= cropped_image.shape
    if h<w:
        cropped_image=cv2.flip(cv2.transpose(cropped_image),1)
    return cropped_image

#================================================================================================
#計算號碼
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
    tolerance=delaunay_layer_avg_langth(points)/1.5

    print(f"tolerance :{tolerance}")
    points_grouped = {}
    for i, (x, y) in enumerate(points):
        group_key = round(x / tolerance) * tolerance  # 分組鍵值
        if group_key not in points_grouped:
            points_grouped[group_key] = []
        points_grouped[group_key].append((x, y))

    # 排序每個組內的點 (按 y 坐標遞減)
    for key in points_grouped:
        points_grouped[key].sort(key=lambda p: -p[1])  # 按 y 坐標遞減排序

    # 排序所有分組鍵並合併後給每個點編號
    sorted_points = []
    for key in sorted(points_grouped.keys()):  # 按 x 分組鍵排序
        sorted_points.extend(points_grouped[key])  # 合併結果

    sorted_points_with_index = [(i + 1, coord[0], coord[1]) for i, coord in enumerate(sorted_points)]

    # 查找 different_points 的編號
    different_points_num = []
    for diff_point in different_points:
        for idx, coord_x, crood_y in sorted_points_with_index:
            if (coord_x, crood_y) == diff_point:
                different_points_num.append((idx, coord_x, crood_y))
                break
        else:
            different_points_num.append((None, diff_point[0], diff_point[1]))  # 如果沒找到，返回 None

    return different_points_num,sorted_points_with_index



def delaunay_layer_avg_langth(points):
    points_np = np.array(points, dtype=np.int32)
    tri = Delaunay(points_np)
    #計算綠線
    total_green_length = 0  # 紀錄綠線總長度
    green_count = 0         # 紀錄綠線的總數

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
    
    if green_count > 0:
        avg_green_length = total_green_length / green_count
        print(f"avg_green_length : {avg_green_length}")
        return avg_green_length
    else:
        print("沒有符合條件的正三角形（綠線）。")
        return None
#================================================================================================
def change_index(sorted_points):
    #origin= [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13]
    become = [ 1, 8, 2, 9, 3,10, 4,11, 5,12, 6,13, 7]
    pro_caculate_moves=[-6,0,6,-1,5,-2,4,-3,3,-4,2,-5,1]

    new_sorted_points = []
    count=0
    batch=0
    for point in sorted_points:
        print(point[0])
        index=point[0]%13
        new_point=(point[0]+(pro_caculate_moves[index]), point[1], point[2])
        new_sorted_points.append(new_point)
    return new_sorted_points

def get_index_v1(points):
    sorted_points_x = sorted(points, key=lambda coord: coord[0])
    subpoints=[]
    sorted_points=[]
    batch=0
    if len(sorted_points_x)%13==0:
        for i in range(65):
            if (i+1)%13==0:
                subpoints.append(sorted_points_x[i])

                sorted_points_y = sorted(subpoints, key=lambda coord: -coord[1])
                for index, coord in enumerate(sorted_points_y):
                    points_with_index=(index+1+(batch*13), coord[0], coord[1])
                    sorted_points.append(points_with_index)
                subpoints=[]
                batch+=1
            else:
                subpoints.append(sorted_points_x[i])
        sorted_points=change_index(sorted_points)
        # print(sorted_points)
        return sorted_points
    return None
    # sorted_points_y = sorted(points, key=lambda coord: coord[1])


#================================================================================================

# 測試程式
if __name__ == "__main__":


    layer=2
    image = cv2.imread(f"C:/Users/Peiteng.Chuang/Desktop/perfect_grid/layer_{layer}.jpg")
    

    #================================================================
    #點+角度+小框+大框+切割+旋轉
    calibrate_image=calibrate_image_with_box(image)
    calibrate_image_2=calibrate_image.copy()
    #=================================================================主要編號判斷
    boxes_4pt,points=det_model_getpoints(calibrate_image)
    if len(points) >= 3:
        different_points = random.sample(points, 3)
        print("random test points : ", different_points)

        different_points_num,spwi=find_points_numbers(points, different_points)       #找出點

        print("different points with numbers : " )
        for num, coord_x,cord_y in spwi:
            x, y = map(int, (coord_x,cord_y))
            cv2.circle(calibrate_image, (x, y), 5, (0, 255, 0), -1)  # 綠色小圓點
            cv2.putText(calibrate_image, str(num), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)  # 標註編號
        for idx,dy,dx in different_points_num:
            print(f"(num:{idx}, y {dy}, x {dx})")
            # dx,dy=map(int, dp)
            cv2.circle(calibrate_image, (dy,dx), 6, (0, 0, 255), -1)  # 紅色小圓點
        
    cv2.imshow("calibrate_image_with_box", calibrate_image)
    #================================================================
    new_spwi = get_index_v1(points)
    if new_spwi is not None:
        for num, coord_x,cord_y in new_spwi:
                x, y = map(int, (coord_x,cord_y))
                cv2.circle(calibrate_image_2, (x, y), 5, (0, 255, 0), -1)  # 綠色小圓點
                cv2.putText(calibrate_image_2, str(num), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)  # 標註編號
        cv2.imshow("calibrate_image2_with_box", calibrate_image_2)
    
    #================================================================

    # 顯示結果
    # cv2.imshow("image_0 Bounding Box", image_rz0)
    # cv2.imshow("image Bounding Box", image_rzo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
