import mediapipe as mp
import cv2
import pandas as pd
import numpy as np

# 初始化 MediaPipe Pose 模組
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 影片讀取
cap = cv2.VideoCapture("trimmed_video.mp4")  # 替換為您的影片路徑

# 動作特徵資料
action_features = {
    "moving_base_part_1": {"start": 2, "end": 40, "features": []},
    "moving_base_part_2": {"start": 208, "end": 320, "features": []},
    "place_item_1": {"start": 469, "end": 473, "features": []},
    "place_item_2": {"start": 490, "end": 525, "features": []},
    "press_button": {"start": 537, "end": 560, "features": []},
    "check_the_hole_1": {"start": 706, "end": 730, "features": []},
    "check_the_hole_2": {"start": 742, "end": 760, "features": []},
    "put_it_into_box": {"start": 772, "end": 788, "features": []},

}

frame_count = 0

def calculate_angle(a, b, c):
    """計算由三個關鍵點所形成的角度"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ab = a - b
    bc = c - b
    cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 進行 Mediapipe 處理
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    # 判斷當前幀是否在任何動作的範圍內
    for action_name, info in action_features.items():
        if info["start"] <= frame_count <= info["end"]:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 提取關鍵點，並計算 x, y, z 以及 visibility
                keypoints = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks[11:23]])

                # 計算所有可能的關鍵點之間的距離和角度
                distances = []
                angles = []
                for i in range(len(keypoints)):
                    for j in range(i + 1, len(keypoints)):
                        distances.append(np.linalg.norm(keypoints[i][:2] - keypoints[j][:2]))
                for i in range(len(keypoints) - 2):
                    angles.append(calculate_angle(keypoints[i][:2], keypoints[i+1][:2], keypoints[i+2][:2]))

                # 特徵字典
                feature_dict = {
                    "Action": action_name,
                    "Frame": frame_count
                }
                # 將距離和角度加入特徵字典
                for k, distance in enumerate(distances):
                    feature_dict[f"Distance_{k}"] = distance
                for k, angle in enumerate(angles):
                    feature_dict[f"Angle_{k}"] = angle

                # 儲存特徵
                info["features"].append(feature_dict)

    frame_count += 1  # 增加幀計數

cap.release()

# 將所有特徵整理至 DataFrame，並儲存為 CSV 檔案
all_features = []

for action_name, info in action_features.items():
    for feature in info["features"]:
        all_features.append(feature)

# 建立 DataFrame 並儲存
df = pd.DataFrame(all_features)
df.to_csv("action_features_with_distances_and_angles.csv", index=False)

print("動作特徵數據已保存至 'action_features_with_distances_and_angles.csv'。")