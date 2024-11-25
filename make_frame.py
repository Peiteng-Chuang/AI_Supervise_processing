import cv2
import os

# 設定影片路徑和儲存 frames 的資料夾
video_path = "IMG_5100.mp4"  # 替換成你的影片檔名或路徑
output_folder = "frames"

# 建立 frames 資料夾（如果不存在的話）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 讀取影片
cap = cv2.VideoCapture(video_path)
frame_count = 0
img_count = 0
# 循環讀取影片的每一幀
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 儲存幀影像到 frames 資料夾
    if frame_count%20==0:
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        img_count += 1
    frame_count += 1

# 釋放影片資源
cap.release()
print(f"總共儲存了 {img_count} 個 frames 到資料夾 {output_folder}")
