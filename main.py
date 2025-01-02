import sys
import time
import random
from datetime import datetime

work_stage_log_path="C:/Users/Peiteng.Chuang/Desktop/work_stage.log"     # 定義work_strage.log路境

def read_workstage_log(log_file_path = work_stage_log_path):

    # 讀取 work_stage.log 檔案，取得最後一筆記錄的 box, layer 和 items_count，並回傳。
    #每一條log格式:time=20250102-133956,box=1,layer=1,items_count=63,change_box=0
    # Returns:
    #     tuple: (box, layer, items_count) 最後更新的數據。

    try:
        with open(log_file_path, "r") as file:
            lines = file.readlines()

        if not lines:
            raise ValueError("Log file is empty.")

        last_line = lines[-1].strip()
        parts = last_line.split(",")

        data = {}
        for part in parts:
            key, value = part.split("=")
            data[key.strip()] = value.strip()

        box = int(data["box"])
        layer = int(data["layer"])
        items_count = int(data["items_count"])
        change_box_flag = True if int(data["change_box"])==1 else False
        return box, layer, items_count, change_box_flag

    except FileNotFoundError:
        print(f"Error: {log_file_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None

def get_date_string():
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")

def update_workstage_log(box,layer,count,change_box,path=work_stage_log_path):

    now_date = get_date_string()
    if change_box==True:
        cb_flag=1 
    else:
        cb_flag=0
    write_str=f"write {now_date},{box},{layer},{count},{cb_flag}"
    
    with open(path, 'a') as f:
        if change_box:
            print(f"box{box} has been loaded completely, waiting for the box to be changed...") # waiting for the box to be changed...
            f.write(f"time={now_date},box={box},layer={layer},items_count={count},change_box={cb_flag}\n")
        else:
            print(write_str)
            f.write(f"time={now_date},box={box},layer={layer},items_count={count},change_box={cb_flag}\n")

def detect_new_box():
    global CHANGE_BOX
    if CHANGE_BOX==True:    
        CHANGE_BOX=random.choices([True, False], weights=[90, 10], k=1)[0]
        if CHANGE_BOX==False:
            print(f"Change complete. Starting detect new box.")
        return CHANGE_BOX
    else:
        return CHANGE_BOX
    
global WORK_STAGE, CHANGE_BOX
WORK_STAGE = read_workstage_log()
CHANGE_BOX = False

def get_change_box_flag():
    global CHANGE_BOX
    return CHANGE_BOX

def switch_change_box_flag():
    global CHANGE_BOX
    if CHANGE_BOX==False:
        CHANGE_BOX=True   
    else:
       CHANGE_BOX=False


if __name__ == "__main__":

    #================================================================================================
    #確保log正確載入
    if WORK_STAGE is not None:
        print(WORK_STAGE)
        print(f"work_stage最後更新的數據為: Box={WORK_STAGE[0]}, Layer={WORK_STAGE[1]}, Items Count={WORK_STAGE[2]}")
    else:
        print("Work stage not found. Stopping process")
        sys.exit(1)
    #================================================================================================
    #換箱寫入log判斷式--eazy mod
    box,layer,items_count = WORK_STAGE[0],WORK_STAGE[1],WORK_STAGE[2]
    t=random.randint(20,100)   #不定時拍照次數
    for i in range(t):                               #無窮迴圈
                                                    #設定拍照程式

        #================================================================================================
        # 換箱 & 紀錄
        change_box_flag=get_change_box_flag()           #抓取change_box_flag，True/False
        if change_box_flag==False:
                                            
            if True:
                items_count+=1                              #*****判斷 : 偵測到物件+1*****

            update_workstage_log(box,layer,items_count,change_box_flag) #先紀錄

            if items_count//65==layer:             #判斷滿層+1 或是 滿箱>>進入換箱辨識
                if layer==6:
                    print("full, switch flag")
                    switch_change_box_flag()        #切換flag，進入換箱模式
                else:
                    layer+=1
        else:
            change_box_flag=detect_new_box()        #當偵測到換完新箱子時才會回傳False，換箱子時是True
            if change_box_flag==False:              #換箱完畢，重置數據
                box+=1
                layer=1
                items_count=0
            update_workstage_log(box,layer,items_count,change_box_flag)
            

    #================================================================================================

#整合程式以及主要執行
# 1.讀取之前的log紀錄，抓取目前箱數、層數、零件數(?)
# 2.yolo detect 洞洞計數
# 3.綜合算法分層(連線分數、先驗分數)
# 4.洞洞EGDT
# 5.分配&計算位置(1-1~10-6)
# 5-1.結合(3)(4)(5)辨識，滿層拍照記錄
# 5-2.使用(3)+diff程式來找出層數、目前出錯位置
# 6.根據(4)，錯誤佔辨識80% or 辨識過少、錯誤過多>>換箱辨識(掃黃)
# 6-1.if 換箱，box+1,layer=1,items_count=0，(但是何時開始生產?如何確定開始?攝影機開始時的判斷)
# 7.紀錄log，箱數、層數、零件數
