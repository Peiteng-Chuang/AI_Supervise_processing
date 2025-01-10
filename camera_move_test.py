import cv2
import numpy as np

def get_frame_difference(frame1, frame2):
    """
    計算兩個影格的不同處，使用模糊化處理後比較，回傳差異的綠色方框。
    並判斷是否有偵測到移動。
    """
    g=81
    # 將影格模糊化
    blurred_frame1 = cv2.GaussianBlur(frame1, (g, g), 0)
    blurred_frame2 = cv2.GaussianBlur(frame2, (g, g), 0)

    # 計算模糊後的差異
    diff = cv2.absdiff(blurred_frame1, blurred_frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    has_movement = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 過濾太小的差異
            has_movement = True
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    mse = np.mean((blurred_frame1.astype("float") - blurred_frame2.astype("float")) ** 2)

    return frame2, has_movement,mse

def main():
    """
    主函式：使用攝影機擷取影格並即時顯示。
    """
    cap = cv2.VideoCapture(0)  # 開啟攝影機

    if not cap.isOpened():
        print("無法開啟攝影機！")
        return

    prev_frame = None
    stop_frame_count = 0  # 計算連續 STOP 的影格數

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取攝影機影格！")
            break

        if prev_frame is not None:
            display_frame, has_movement, mse = get_frame_difference(prev_frame, frame.copy())
        else:
            mse=0
            display_frame = frame.copy()
            has_movement = False

        # 更新 STOP 計數
        if has_movement:
            stop_frame_count = 0
        else:
            stop_frame_count += 1

        # 判斷狀態文字
        if stop_frame_count >= 00:
            status_text = "STOP"
            status_color = (0, 255, 0)
        else:
            status_text = "MOVING"
            status_color = (0, 0, 255)

        print(f"mse={mse:.2f}")
        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("Camera Viewer", display_frame)

        key = cv2.waitKey(1)

        if key == 27:  # 按下 ESC 鍵中斷
            print("程式中斷。")
            break

        # 更新參考影格
        prev_frame = frame

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
