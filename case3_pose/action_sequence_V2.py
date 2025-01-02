import cv2
import mediapipe as mp
import threading
import time,math
import numpy as np

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
event_name=['place','button','process','remove','clean','in box']
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
reset_countdown_flag=False

def set_work_area():        # 調整螢幕劃分的區域
    horizon_threshold = 0.25
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
#===============================================================================
def check_event_stage():
    global ef,event_stage
    event_stage=0
    for e in ef:
        if e:
            event_stage+=1
            

def reset_all_flags():
    global sf,ef,reset_countdown_flag
    temp_fs=[False, False, False, False, False, False, False, False]
    temp_ef=ef=[False,False,False,False,False,False]
    reset_countdown_flag=False
    sf=temp_fs
    ef=temp_ef

reset_lock = threading.Lock()

def reset_countdown():              #重製流程倒數，多線程計時，在秒數內有偵測到function啟動就pass，不然就會重製流程
    global reset_countdown_flag
    if reset_countdown_flag == True:
        return

    def do_restart_flags():
        with reset_lock:
            time.sleep(3)  # 等待 n 秒
            reset_all_flags()

    # 創建並啟動執行緒
    if  reset_countdown_flag == False:
        reset_countdown_flag == True
        if not hasattr(reset_countdown, "thread") or not reset_countdown.thread.is_alive():
            reset_countdown.thread = threading.Thread(target=do_restart_flags, daemon=True)
            reset_countdown.thread.start()
        

def place_component(landmarks):              #1.放置零件
    global sf,ef,component_x_location
    ht, sax, say, pax, pay, wtax, wtay, bx, by = set_work_area()
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    # global hands_over_shoulder_flag,hands_straight_down_flag
    if(wrist.y < shoulder.y) or sf[0] ==True:   #手高於肩
        sf[0]=True
        if(wrist.y > wtay[0]):
            sf[1]=True
    if sf[0] and sf[1]:
        component_x_location=wrist.x*1.2        #按鈕在更靠右邊
        ef[0]=True
    
def start_machine(landmarks):                #2.開啟機器
    global sf,ef
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    ht, sax, say, pax, pay, wtax, wtay, bx, by = set_work_area()
    if(wrist.x > bx[0] and wrist.x<bx[1] and wrist.y>by[0] and wrist.y<by[1]):   #手左於��
        sf[2]=True
        
    if sf[2]:
        ef[1]=True

def wait_until_process_finish():    #3.等待加工20秒
    global ef

    # 定義執行緒的目標函式
    def update_flag():
        time.sleep(5)  # 等待 n 秒
        sf[3] = True
        # print("ef[2] 已被設置為 True")
    if sf[3]:
        ef[2]=True
    # 創建並啟動執行緒
    thread = threading.Thread(target=update_flag)
    thread.start()


def remove_component(landmarks):     #4.移除零件
    global sf,ef,component_x_location
    ht, sax, say, pax, pay, wtax, wtay, bx, by = set_work_area()
    wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    if(wrist.x > pax[1]-(pax[1]-pax[0])/5):   #手左於��
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

# def turn_around(landmarks):                  #6.轉身離開，完成一個迴圈
#     shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
#     shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
#     if(shoulder_r.x > shoulder_l.x):
#         sf[7]=True
#     if sf[7]:
#         ef[5]=True

def in_box(landmarks):
    ht, sax, say, pax, pay, wtax, wtay, bx, by = set_work_area()
    hip_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    if(hip_r.y<ht or hip_l.y<ht):
        sf[7]=True
    if sf[7]:
        ef[5]=True


def put_detial(frame):
    global sf,ef,event_name
    n=1
    stage_list=['place1','place2','button','prossed','remove','clean1','clean2','in box']
    for s in sf:
        if s:
            cv2.putText(frame,f'{stage_list[n-1]}', (frame.shape[1]-150, 35*n), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame,f'{stage_list[n-1]}', (frame.shape[1]-150, 35*n), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
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
    cap = cv2.VideoCapture('C:/Users/Peiteng.Chuang/Desktop/factor/video/2024-11-26_11-49-31.avi')

    # 取得視頻資訊
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設置保存為 MP4 格式
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 設置輸出視頻
    out = cv2.VideoWriter('./pose_tracking/test.mp4', fourcc, fps, (width, height))
    frame_count = 0
    ht, sax, say, pax, pay, wtax, wtay, bx, by = set_work_area()

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
                # turn_around(landmarks)
                in_box(landmarks)
            if event_stage == 6 :
                reset_countdown()
                event_stage==99

        frame=put_detial(frame)
        # 在影像上顯示當前幀數
        # cv2.putText(frame, f'frame : {frame_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # 將結果寫入輸出視頻
        area_sig = apply_transparent_filter(frame, sax, say, filter_color=(255, 0, 0), alpha=0.20)
        area_sig = apply_transparent_filter(area_sig, pax, pay, filter_color=(0, 0, 255), alpha=0.20)
        area_sig = apply_transparent_filter(area_sig, wtax, wtay, filter_color=(0, 255, 0), alpha=0.20)
        area_sig = apply_transparent_filter(area_sig, bx, by, filter_color=(0, 255, 255), alpha=0.20)
        area_sig = draw_horizontal_line(area_sig, ht, color=(0, 0, 255))
        out.write(area_sig)
        # 顯示當前幀影像（可選）
        cv2.imshow('Pose Detection', area_sig)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 呼叫 main 函式
if __name__ == '__main__':
    main()

