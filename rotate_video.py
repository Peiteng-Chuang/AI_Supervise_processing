import cv2

# 讀取影片檔案
input_file = './IMG_2.mov'
output_file = 'output_rotated.mov'

# 打開影片檔案
cap = cv2.VideoCapture(input_file)

# 取得影片的幀率、原始寬度與高度
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 旋轉後的寬高
rotated_width, rotated_height = height, width

# 定義影片寫入格式 (四字符編碼、幀率、旋轉後的寬高)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 格式
out = cv2.VideoWriter(output_file, fourcc, fps, (rotated_width, rotated_height))

# 處理每一幀
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 將每一幀逆時針旋轉90度
    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # 寫入旋轉後的幀
    out.write(rotated_frame)

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()

print("影片處理完成，已保存至", output_file)
