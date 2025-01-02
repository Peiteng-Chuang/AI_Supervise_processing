import numpy as np
import cv2
from scipy.spatial import Delaunay

# 假設這是你的點的列表
points = np.array([
    (724.0, 507.0), (725.0, 412.0), (1099.0, 282.0), (1068.0, 777.0), (851.0, 823.0), (922.0, 873.0), (1140.0, 823.0), 
    (800.0, 459.0), (867.0, 714.0), (790.0, 760.0), (868.0, 514.0), (794.0, 664.0), (1161.0, 422.0), 
    (1228.0, 474.0), (938.0, 663.0), (711.0, 803.0), (716.0, 707.0), (926.0, 778.0), (637.0, 752.0), (1085.0, 573.0), 
    (1085.0, 473.0), (1000.0, 830.0), (774.0, 869.0), (876.0, 410.0), (941.0, 467.0), (653.0, 360.0), 
    (951.0, 363.0), (998.0, 727.0), (939.0, 564.0), (799.0, 367.0), (1072.0, 880.0), (1012.0, 517.0), (1014.0, 423.0), 
    (1143.0, 727.0), (793.0, 562.0), (1088.0, 379.0), (1222.0, 782.0), (1295.0, 835.0), (730.0, 311.0), 
    (1311.0, 336.0), (640.0, 652.0), (720.0, 609.0), (951.0, 274.0), (1236.0, 385.0), (1167.0, 333.0), (865.0, 612.0), 
    (1310.0, 438.0), (805.0, 267.0), (1024.0, 321.0), (1154.0, 523.0), (1068.0, 676.0), (877.0, 316.0), 
    (1009.0, 613.0), (1240.0, 285.0), (1150.0, 629.0), (646.0, 559.0), (649.0, 459.0), (658.0, 259.0), (1306.0, 525.0), 
    (1299.0, 626.0), (1215.0, 876.0), (1223.0, 683.0), (1223.0, 579.0), (633.0, 851.0), (1298.0, 732.0)
    ], dtype=np.int32)

# 創建 Delaunay 三角剖分
tri = Delaunay(points)

# 建立全黑的圖像 (遮罩)
img_size = (1920, 1080)  # 圖片大小
image = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)  # 全黑圖像 (高度, 寬度, 色彩通道)

# 繪製三角形邊線
for simplex in tri.simplices:
    pts = points[simplex]  # 獲取三角形的頂點
    cv2.polylines(image, [pts], isClosed=True, color=(255, 255, 255), thickness=3)  # 繪製白色邊線

if points is not None:
    print(points)
    points_np = np.array(points, dtype=np.int32)
    tri = Delaunay(points_np)
    image_delaunay = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
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
        if (max_edge - min_edge) / max_edge <= 0.25:  # 如果邊長差異在 n% 內
            color = (0, 255, 0)  # 綠色
            cv2.line(image_delaunay, tuple(pts[0]), tuple(pts[1]), color, thickness=3)
            cv2.line(image_delaunay, tuple(pts[1]), tuple(pts[2]), color, thickness=3)
            cv2.line(image_delaunay, tuple(pts[2]), tuple(pts[0]), color, thickness=3)
        # else:
        #     color = (0, 0, 255)  # 紅色
        # 繪製三條邊
        

# 儲存結果
# cv2.imwrite('triangulation.png', image)
image=cv2.resize(image,(img_size[0]//2, img_size[1]//2))
image_delaunay=cv2.resize(image_delaunay,(img_size[0]//2, img_size[1]//2))
cv2.imshow("Delaunay",image)
cv2.imshow("image_delaunay",image_delaunay)
cv2.waitKey(0)
