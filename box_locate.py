import cv2
import numpy as np

def find_box(image_path):
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    image = cv2.resize(image,(image.shape[1]//2,image.shape[0]//2))
    # 轉灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊 (降噪)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 邊緣檢測
    edges = cv2.Canny(blurred, 50, 150)

    # 輪廓檢測
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化最大面積與對應的矩形
    max_area = 0
    best_box = None

    for contour in contours:
        # 忽略過小的輪廓
        if cv2.contourArea(contour) < 1000:
            continue

        # 最小外接矩形 (允許旋轉)
        rotated_rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rotated_rect)
        box = np.int0(box)

        # 計算該矩形的面積
        width = rotated_rect[1][0]
        height = rotated_rect[1][1]
        area = width * height

        # 更新最大面積的矩形
        if area > max_area:
            max_area = area
            best_box = box

    if best_box is not None:
        # 繪製結果
        output_image = image.copy()
        cv2.drawContours(output_image, [best_box], -1, (0, 255, 0), 3)

        # 顯示圖片
        cv2.imshow("Detected Box", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No suitable box found.")

# 替換成你的圖片路徑
if __name__ == "__main__":

    pth="C:/Users/Peiteng.Chuang/Desktop/box/"
    f_name="box_3.jpg"
    img_path = pth+f_name  # 替换为你的图像路径

    find_box(img_path)

