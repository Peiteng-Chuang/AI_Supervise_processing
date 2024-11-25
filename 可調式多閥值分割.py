import cv2
import numpy as np

# 定義回調函數
def nothing(x):
    pass

# 讀取影像
image = cv2.imread('sift_img.png', cv2.IMREAD_GRAYSCALE)

# 建立視窗
cv2.namedWindow('Tri-Threshold')

# 創建Trackbars用於調整兩個閥值
cv2.createTrackbar('Low Threshold', 'Tri-Threshold', 0, 255, nothing)
cv2.createTrackbar('High Threshold', 'Tri-Threshold', 0, 255, nothing)

while True:
    # 獲取當前的閥值
    low_th = cv2.getTrackbarPos('Low Threshold', 'Tri-Threshold')
    high_th = cv2.getTrackbarPos('High Threshold', 'Tri-Threshold')

    # 建立一個全黑的影像
    tri_binary = np.zeros_like(image)

    # 將範圍分為黑、灰、白三部分
    tri_binary[image < low_th] = 0   # 黑色部分
    tri_binary[(image >= low_th) & (image < high_th)] = 127  # 灰色部分
    tri_binary[image >= high_th] = 255  # 白色部分

    # 顯示結果
    cv2.imshow('Tri-Threshold', tri_binary)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(f"threadhole_img.png", tri_binary)
        break

# 釋放視窗
cv2.destroyAllWindows()
