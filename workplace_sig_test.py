import cv2
import numpy as np

def set_work_area():        # 調整螢幕劃分的區域
    horizon_threshold = 0.2
    saw_area_x, saw_area_y = (0.425, 0.475), (0, 0.5)
    processing_area_x, processing_area_y = (0.375, 0.525), (0.35, 0.55)
    work_table_area_x, work_table_area_y = (0.3, 0.55), (0.5, 1.0)
    return horizon_threshold, saw_area_x, saw_area_y, processing_area_x, processing_area_y, work_table_area_x, work_table_area_y

def apply_transparent_filter(image, area_x, area_y, filter_color=(255, 0, 0), alpha=0.5):
    # 讀取圖片的寬高
    height, width, _ = image.shape
    # print(f"{width} x {height}")      #除錯用的
    
    # 計算區域的邊界
    x_start = int(area_x[0] * width)
    x_end = int(area_x[1] * width)
    y_start = int(area_y[0] * height)
    y_end = int(area_y[1] * height)

    # 建立圖片副本
    img_array = np.copy(image)

    # 濾鏡顏色要先轉換成三通道（RGB/BGR）
    filter_color = np.array(filter_color, dtype=np.uint8)

    # 將濾鏡顏色應用於指定區域，並使用 alpha 混合來保持背景物件的可見性
    for c in range(3):  # 對 R, G, B 通道逐一操作
        img_array[y_start:y_end, x_start:x_end, c] = (
            img_array[y_start:y_end, x_start:x_end, c] * (1 - alpha) + filter_color[c] * alpha
        ).astype(np.uint8)

    return img_array

def draw_horizontal_line(image,ht,color=(0,0,255)):
    height, width, _ = image.shape
    y=int(ht*height)
    cv2.line(image, (0, y), (width, y), color, 1)
    return image

def main():
    img = cv2.imread('./working.png')
    if img is None:
        print("Error: Could not read the image.")
        return

    ht, sax, say, pax, pay, wtax, wtay = set_work_area()

    # 將透明濾鏡應用到圖片
    result = apply_transparent_filter(img, sax, say, filter_color=(255, 0, 0), alpha=0.25)
    result = apply_transparent_filter(result, pax, pay, filter_color=(0, 0, 255), alpha=0.25)
    result = apply_transparent_filter(result, wtax, wtay, filter_color=(0, 255, 0), alpha=0.25)
    result = draw_horizontal_line(result, ht, color=(0,0,255))

    # 顯示結果圖片
    cv2.imwrite("working_sig.png", result)
    cv2.imshow("Transparent Color Filter Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
