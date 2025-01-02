import cv2
import time

# 初始化變數
recording = False
out = None

# 設定 webcam
cap = cv2.VideoCapture(0)

# 確認是否成功打開 webcam
if not cap.isOpened():
    print("無法打開攝影機")
    exit()

# 設定影片編碼與輸出格式
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = "recorded_video.avi"

print("按下空白鍵開始錄影，再次按下空白鍵停止錄影，按下 'ESC' 結束程式。")

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影像")
        break

    # 即時顯示畫面
    cv2.imshow('Webcam', frame)

    # 如果正在錄影，將影像寫入檔案
    if recording and out:
        out.write(frame)

    # 監聽鍵盤事件
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        # 按下 'q' 結束程式
        break
    elif key == 32:  # 空白鍵
        if not recording:
            # 開始錄影
            print("開始錄影...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"recorded_video_{timestamp}.avi"
            out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))
            recording = True
        else:
            # 停止錄影
            print(f"停止錄影，影片已儲存為 {output_file}")
            recording = False
            out.release()
            out = None

# 釋放資源
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
