import cv2
import numpy as np

def set_work_area():        # 調整螢幕劃分的區域
    horizon_threshold = 0.2
    saw_area_x, saw_area_y = (0.68, 0.75), (0.2, 0.8)
    processing_area_x, processing_area_y = (0.62, 0.75), (0.73, 0.95)
    work_table_area_x, work_table_area_y = (0.45, 0.95), (0.65, 1.0)
    botton_area_x, botton_area_y = (0.32,0.48),(0.75,1.0)
    return horizon_threshold, saw_area_x, saw_area_y, processing_area_x, processing_area_y, work_table_area_x, work_table_area_y,botton_area_x, botton_area_y

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
#圖片用
def main():
    img = cv2.imread('C:/project_file/AI_Supervise_processing/frames/frame_0465.png')
    if img is None:
        print("Error: Could not read the image.")
        return

    ht, sax, say, pax, pay, wtax, wtay, bx, by = set_work_area()

    # 將透明濾鏡應用到圖片
    result = apply_transparent_filter(img, sax, say, filter_color=(255, 0, 0), alpha=0.25)
    result = apply_transparent_filter(result, pax, pay, filter_color=(0, 0, 255), alpha=0.25)
    result = apply_transparent_filter(result, wtax, wtay, filter_color=(0, 255, 0), alpha=0.25)
    result = apply_transparent_filter(result, bx, by, filter_color=(0, 255, 255), alpha=0.25)
    result = draw_horizontal_line(result, ht, color=(0,0,255))

    # 顯示結果圖片
    cv2.imwrite("working_sig.png", result)
    cv2.imshow("Transparent Color Filter Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#影片用
# def main():
#     input_video_path = 'C:/Users/Peiteng.Chuang/Desktop/factor/video/2024-11-26_11-25-30.avi'
#     output_video_path = 'output_video.mp4'

#     # 開啟影片檔案
#     cap = cv2.VideoCapture(input_video_path)
#     if not cap.isOpened():
#         print("Error: Could not open the video file.")
#         return

#     # 取得影片的基本資訊
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # 初始化影片寫入器
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#     ht, sax, say, pax, pay, wtax, wtay = set_work_area()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 對每幀應用濾鏡效果
#         result = apply_transparent_filter(frame, sax, say, filter_color=(255, 0, 0), alpha=0.20)
#         result = apply_transparent_filter(result, pax, pay, filter_color=(0, 0, 255), alpha=0.20)
#         result = apply_transparent_filter(result, wtax, wtay, filter_color=(0, 255, 0), alpha=0.20)
#         result = draw_horizontal_line(result, ht, color=(0, 0, 255))

#         # 將處理後的影像寫入影片檔案
#         out.write(result)

#         # 同時顯示處理後的影像（可選）
#         cv2.imshow('Processed Frame', result)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # 釋放資源
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
