import cv2
import mediapipe as mp

def draw_upper_body(frame, landmarks):
    """
    在給定的影像上繪製上半身骨架線（不包括頭部）。
    """
    connections = [
        (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER),
        (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
        (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
        (mp.solutions.pose.PoseLandmark.LEFT_ELBOW, mp.solutions.pose.PoseLandmark.LEFT_WRIST),
        (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW, mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
        (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
        (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    ]
    
    height, width, _ = frame.shape
    for connection in connections:
        p1 = landmarks[connection[0].value]
        p2 = landmarks[connection[1].value]
        if p1.visibility > 0.5 and p2.visibility > 0.5:  # 只繪製可見性高的點
            x1, y1 = int(p1.x * width), int(p1.y * height)
            x2, y2 = int(p2.x * width), int(p2.y * height)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def main():
    # 初始化 Mediapipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

    # 打開輸入視頻
    cap = cv2.VideoCapture('C:/Users/Peiteng.Chuang/Desktop/factor/video/2024-11-26_14-51-50.avi')
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # 取得視頻資訊
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"FPS: {fps}, Width: {width}, Height: {height}")

    # 設置輸出視頻
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./case3_pose/frames_with_pose_2.mp4', fourcc, fps, (width, height))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        frame_count += 1
        # Mediapipe 處理每幀影像
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # 繪製上半身骨架線
            draw_upper_body(frame, results.pose_landmarks.landmark)

        # 添加當前幀數到影像
        cv2.putText(frame, f'Frame: {frame_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 寫入輸出影像
        out.write(frame)

        # 顯示影像（可選）
        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 停止
            break

    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
