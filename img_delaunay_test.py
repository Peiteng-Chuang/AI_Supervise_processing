import numpy as np
import cv2
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
from ultralytics import YOLO
import torch
import random
#================================================================================================
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
    tolerance=delaunay_layer_avg_langth(points)//3
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
#                 Different points with numbers:
#                   (41, 1068, 777)
#                   (47, 1140, 823)
#                   (60, 1295, 835)
                break
        else:
            different_points_num.append((None, diff_point[0], diff_point[1]))  # 如果沒找到，返回 None

    return different_points_num,sorted_points_with_index
#================================================================================================
def find_points_numbers_pro(points, different_points, tolerance=20):
    """
    將點按網格模式排序並返回指定點的編號。

    Args:
        points (list[tuple]): 原始的點座標列表。
        different_points (list[tuple]): 需要查找編號的點座標列表。
        tolerance (int): x 坐標分組的誤差範圍，默認為 10。

    Returns:
        list[tuple]: 每個目標點的 (num, x, y)，其中 num 是編號。

    """
    tolerance=delaunay_layer_avg_langth(points)-20
    # 先按 y 排序，再按 x 排序
    sorted_points = sorted(points, key=lambda p: (p[0], -p[1]))  # 按 y 遞減，再按 x 遞增排序

    # 校準 y 座標，確保每一列 y 排序一致
    calibrated_points = []
    current_y_group = []
    current_y = sorted_points[0][1]

    for point in sorted_points:
        if abs(point[1] - current_y) <= tolerance:  # 如果 y 在容差範圍內，加入當前組
            current_y_group.append(point)
        else:
            # 將當前 y 組按 x 再次排序後加入校準結果
            calibrated_points.extend(sorted(current_y_group, key=lambda p: p[0]))
            current_y_group = [point]  # 開始新的一組
            current_y = point[1]

    # 處理最後一組
    if current_y_group:
        calibrated_points.extend(sorted(current_y_group, key=lambda p: p[0]))

    # 為每個點編號
    sorted_points_with_index = [(i + 1, coord[0], coord[1]) for i, coord in enumerate(calibrated_points)]

    # 查找 different_points 的編號
    different_points_num = []
    for diff_point in different_points:
        for idx, coord_x, coord_y in sorted_points_with_index:
            if (coord_x, coord_y) == diff_point:
                different_points_num.append((idx, coord_x, coord_y))
                break
        else:
            different_points_num.append((None, diff_point[0], diff_point[1]))  # 如果沒找到，返回 None

    return different_points_num, sorted_points_with_index
#================================================================================================
def find_points_numbers_robust(points, different_points):
    """
    根據最近鄰查找目標點的編號。

    Args:
        points (list[tuple]): 原始點的座標列表。
        different_points (list[tuple]): 需要查找的目標點座標列表。

    Returns:
        list[tuple]: 每個目標點的 (num, x, y)，其中 num 是編號。
    """
    # 建立 KD 樹
    tree = cKDTree(points)

    # 最近鄰查找
    different_points_num = []
    for diff_point in different_points:
        dist, idx = tree.query(diff_point)  # 找到最近的點
        if dist < 1e-6:  # 距離足夠小，確定是同一個點（避免浮點誤差）
            num = idx + 1  # 編號從 1 開始
            different_points_num.append((num, points[idx][0], points[idx][1]))
        else:
            # 如果距離過大，可能不是正確點
            different_points_num.append((None, diff_point[0], diff_point[1]))

    # 按網格模式排序點 (y 降序，x 升序)
    sorted_indices = sorted(
        enumerate(points),
        key=lambda x: (x[1][0], -x[1][1])  # y 先降序，x 再升序
    )
    sorted_points_with_index = [(i + 1, p[0], p[1]) for i, (_, p) in enumerate(sorted_indices)]
    print(sorted_points_with_index)
    return different_points_num, sorted_points_with_index


#================================================================================================
def sort_points_with_skew_correction(points, different_points):
    # 確保 points 是 numpy array
    points = np.array(points)

    # Step 1: 計算與 y 軸最接近的線段角度
    angles = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            diff = points[j] - points[i]
            angle = np.arctan2(diff[1], diff[0])  # 計算角度
            angles.append(angle)

    # 將角度轉換為與 y 軸的偏差（目標角度接近 ±90°）
    angles_to_y_axis = [abs(abs(a) - np.pi / 2) for a in angles]

    # 找到偏差最小的角度（與 y 軸最接近）
    # closest_angles = [angles[i] for i in range(len(angles)) if angles_to_y_axis[i] < 0.1]  # 偏差小於 0.1 弧度
    closest_angles = [angles[i] for i in range(len(angles)) if angles_to_y_axis[i] < np.deg2rad(30)]
    # 計算這些角度的平均值作為校正角度
    if closest_angles:
        dominant_angle = np.mean(closest_angles)
    else:
        dominant_angle = 0  # 如果沒有接近 y 軸的角度，默認為 0

    # Step 2: 旋轉校正，使網格變水平
    rotation_matrix = np.array([
        [np.cos(-dominant_angle), -np.sin(-dominant_angle)],
        [np.sin(-dominant_angle), np.cos(-dominant_angle)]
    ])
    rotated_points = points @ rotation_matrix.T  # 點乘進行旋轉校正

    # Step 3: 按校正後的點排序
    sorted_indices = np.lexsort((rotated_points[:, 1], rotated_points[:, 0]))  # 按 y 再按 x 排序
    sorted_points_with_index = [(i + 1, points[i][0], points[i][1]) for i in sorted_indices]
    # print(sorted_points_with_index)

    # Step 4: 建立 KD-Tree 查找不同點的編號
    tree = cKDTree(points)
    different_points_num = []

    for diff_point in different_points:
        dist, idx = tree.query(diff_point)  # 找到最近的點
        if dist < 1e-6:  # 距離足夠小，確定是同一個點（避免浮點誤差）
            num = sorted_indices.tolist().index(idx) + 1  # 獲取正確的排序編號
            different_points_num.append((num, points[idx][0], points[idx][1]))
        else:
            # 如果距離過大，可能不是正確點
            different_points_num.append((None, diff_point[0], diff_point[1]))

    return different_points_num, sorted_points_with_index

def box2point(box):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    point=(int(cx), int(cy))
    return point

def det_model_draw(image, model):

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

    return image,boxes_4pt,points
#================================================================================================
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

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ultralytics_path = "C:/project_file/ultralytics_v11"

    # egdt_model_ver = "v13"
    # egdt_model = YOLO(f'{ultralytics_path}/runs/detect/sp_egdt_{egdt_model_ver}/weights/best.pt', verbose=False)
    # print(f"egdt_Model loaded successfully! model version: {egdt_model_ver}")
    # egdt_model = egdt_model.to(device)

    # 加载孔洞檢測模型
    det_model_ver = "v12"
    det_model = YOLO(f'{ultralytics_path}/runs/detect/sp_holes_{det_model_ver}/weights/best.pt', verbose=False)
    print(f"det_Model loaded successfully! model version: {det_model_ver}")
    det_model = det_model.to(device)

    layer=6
    image = cv2.imread(f"C:/Users/Peiteng.Chuang/Desktop/perfect_grid/layer_{layer}.jpg")
    image,boxes_4pt,points=det_model_draw(image, det_model)
    
    if len(points) >= 3:
        different_points = random.sample(points, 3)
        print("random test points : ", different_points)

    result_num,sort_p_w_i = find_points_numbers(points, different_points)                   #GOOD但泛用性低
    # result_num,sort_p_w_i = find_points_numbers_pro(points, different_points)                   #GOOD但泛用性低
    # result_num,sort_p_w_i = find_points_numbers_robust(points, different_points)        #亂找ID
    # result_num,sort_p_w_i = sort_points_with_skew_correction(points, different_points)
    # print(sort_p_w_i)

    # 輸出結果
    print("Different points with numbers:")
    for item in result_num:
        print(item)


    # # 創建畫布
    img_size = (1920, 1080)
    image_sort = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    # 在畫布上繪製點和編號
    for num, coord_x,cord_y in sort_p_w_i:
        x, y = map(int, (coord_x,cord_y))
        cv2.circle(image_sort, (x, y), 5, (0, 255, 0), -1)  # 綠色小圓點
        cv2.putText(image_sort, str(num), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 標註編號
    for dp in different_points:
        dx,dy=map(int, dp)
        cv2.circle(image_sort, (dx,dy), 6, (0, 0, 255), -1)  # 紅色小圓點
    # #================================================================================================================================



    # # 儲存結果
    # # cv2.imwrite('triangulation.png', image)

    image_sort=cv2.resize(image_sort,(img_size[0]//2, img_size[1]//2))
    cv2.imshow("Points with Numbers", image_sort)

    cv2.waitKey(0)
