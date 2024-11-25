import cv2
import mediapipe as mp
import threading
import time,math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

upper_body_connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.RIGHT_WRIST,mp_pose.PoseLandmark.RIGHT_INDEX.value),
    (mp_pose.PoseLandmark.LEFT_WRIST,mp_pose.PoseLandmark.LEFT_INDEX.value)
]



component_x_location=0

#sequence_flags
sf=[False, False, False, False, False, False, False, False]
#place_component()的flag
hands_over_shoulder_flag=sf[0]
hands_straight_down_flag=sf[1]
#start_machine()flag
righthand_to_button_flag=sf[2]
#wait_until_process_finish()flag
processed_flag=sf[3]
#remove_component()flag
lefthand_remove_component_flag=sf[4]
#clean_debris()flag
righthand_approach_flag=sf[5]
righthand_away_flag=sf[6]
#turn_around()flag
turn_around_flag=sf[7]

#event_flags
ef=[False,False,False,False,False,False]
event_name=['place','button','process','remove','clean','turn']
#place_component()的flag
place_component_flag=ef[0]
#start_machine()flag
start_machine_flag=ef[1]
#wait_until_process_finish()flag
wait_until_process_finish_flag=ef[2]
#remove_component()flag
remove_component_flag=ef[3]
#clean_debris()flag
clean_debris_flag=ef[4]
#turn_around()flag
turn_around_flag=ef[5]

event_stage=0

def check_event_stage():
    global ef,event_stage
    event_stage=0
    for e in ef:
        if e:
            event_stage+=1
            

def reset_all_flags():
    global sf,ef
    temp_fs=[False for s in sf]
    temp_ef=[False for e in ef]
    sf=temp_fs
    ef=temp_ef

def reset_countdown():              #重製流程倒數，多線程計時，在秒數內有偵測到function啟動就pass，不然就會重製流程
    pass

def place_component(landmarks):              #1.放置零件
    global sf,ef,component_x_location
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    # global hands_over_shoulder_flag,hands_straight_down_flag
    if(wrist.y < shoulder.y) or sf[0] ==True:   #手高於肩
        sf[0]=True
        if(wrist.y > shoulder.y):
            sf[1]=True
    if sf[0] and sf[1]:
        component_x_location=wrist.x*1.2        #按鈕在更靠右邊
        ef[0]=True
    
def start_machine(landmarks):                #2.開啟機器
    global sf,ef,component_x_location
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    
    if(wrist.x > component_x_location):   #手左於��
        sf[2]=True
        
    if sf[2]:
        ef[1]=True

def wait_until_process_finish():    #3.等待加工20秒
    global ef

    # 定義執行緒的目標函式
    def update_flag():
        time.sleep(18)  # 等待 n 秒
        sf[3] = True
        # print("ef[2] 已被設置為 True")
    if sf[3]:
        ef[2]=True
    # 創建並啟動執行緒
    thread = threading.Thread(target=update_flag)
    thread.start()


def remove_component(landmarks):     #4.移除零件
    global sf,ef,component_x_location
    wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    if(wrist.x > component_x_location):   #手左於��
        sf[4]=True
        
    if sf[4]:
        ef[3]=True
    
def clean_debris(landmarks):                 #5.清理雜碎
    global sf,ef
    wrist_r = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    wrist_l = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    hand_r = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
    hand_l = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]

    wrist_distance = math.sqrt((wrist_r.x - wrist_l.x) ** 2 + (wrist_r.y - wrist_l.y) ** 2)
    hand_distance = math.sqrt((hand_r.x - hand_l.x) ** 2 + (hand_r.y - hand_l.y) ** 2)
    wrist_to_hand_distance = math.sqrt((wrist_r.x - hand_r.x) ** 2 + (wrist_r.y - hand_r.y) ** 2)
    if hand_distance <= wrist_to_hand_distance or sf[5]:
        sf[5]=True
        if hand_distance >= wrist_to_hand_distance:
            sf[6]=True
    if sf[5] and sf[6]:
        ef[4]=True

def turn_around(landmarks):                  #6.轉身離開，完成一個迴圈
    shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    if(shoulder_r.x > shoulder_l.x):
        sf[7]=True
    if sf[7]:
        ef[5]=True

def put_detial(frame):
    global sf,ef,event_name
    n=1
    stage_list=['place1','place2','button','prossed','remove','clean1','clean2','turn']
    for s in sf:
        if s:
            cv2.putText(frame,f'{stage_list[n-1]}', (280, 30*n), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame,f'{stage_list[n-1]}', (280, 30*n), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        n+=1
    # i=1
    # for e in ef:
    #     if e:
    #         # cv2.putText(frame, f'e{i-1}:T', (300, 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    #         cv2.putText(frame, f'{event_name[i-1]}', (300, 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    #     else:
    #         # cv2.putText(frame, f'e{i-1}:F', (300, 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    #         cv2.putText(frame, f'{event_name[i-1]}', (300, 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    #     i+=1
    return frame
def main():
    # 初始化 mediapipe 和 opencv
    
    mp_drawing = mp.solutions.drawing_utils

    # 打開輸入視頻
    cap = cv2.VideoCapture('./pose_tracking/loop1.mp4')

    # 取得視頻資訊
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設置保存為 MP4 格式
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 設置輸出視頻
    out = cv2.VideoWriter('./pose_tracking/test.mp4', fourcc, fps, (width, height))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break
        global sf,ef
        # 轉換影像為 RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 用 mediapipe 進行人體姿態偵測
        result = pose.process(image_rgb)

        # 如果有偵測到姿態，繪製點與連接線
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            landmarks=result.pose_landmarks.landmark

            check_event_stage()

            if event_stage == 0 :
                place_component(landmarks)
            if event_stage == 1 :
                start_machine(landmarks)
            if event_stage == 2 :
                wait_until_process_finish()
            if event_stage == 3 :
                remove_component(landmarks)
            if event_stage == 4 :
                clean_debris(landmarks)
            if event_stage == 5 :
                turn_around(landmarks)
            # 印出所有關節點的座標值
            # for idx, landmark in enumerate(result.pose_landmarks.landmark):
            #     x, y, z = landmark.x, landmark.y, landmark.z
            #     print(f"Frame {frame_count} - Landmark {idx}: (x={x:.4f}, y={y:.4f}, z={z:.4f})")
        frame=put_detial(frame)
        # 在影像上顯示當前幀數
        # cv2.putText(frame, f'frame : {frame_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # 將結果寫入輸出視頻
        out.write(frame)
        # 顯示當前幀影像（可選）
        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 呼叫 main 函式
if __name__ == '__main__':
    main()

