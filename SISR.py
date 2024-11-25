# import cv2
# import numpy as np

# # 計算均方誤差 (MSE)，作為圖像相似度的簡易替代方法
# def calculate_mse(image1, image2):
#     gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#     mse = np.mean((gray1 - gray2) ** 2)
#     return mse

# # 超解析度重建函數：多幀平均 + 超解析模型
# def super_resolution_with_model(frames, model_path='ESPCN_x4.pb'):
#     # 將多幀平均
#     avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    
#     # 初始化超解析模型
#     sr = cv2.dnn_superres.DnnSuperResImpl_create()
#     sr.readModel(model_path)
#     sr.setModel("espcn", 4)  # 設定模型為ESPCN，放大4倍
    
#     # 對疊加圖像應用超解析
#     super_res_image = sr.upsample(avg_frame)
#     return super_res_image

# # 初始化攝影機
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("無法打開攝像頭")
#     exit()

# frames = []
# count = 0
# max_frames = 10  # 需要的幀數
# mse_threshold = 8  # 設定均方誤差的門檻值

# while len(frames) < max_frames:
#     ret, frame = cap.read()
#     if not ret:
#         print("無法捕捉影像")
#         break

#     count += 1

#     # 每10幀取1個frame
#     if count % 10 == 0:
#         if len(frames) > 0:
#             last_frame = frames[-1]
#             mse = calculate_mse(last_frame, frame)
#             print(f"當前幀與前一幀的均方誤差: {mse}")

#             if mse > mse_threshold:
#                 print("均方誤差過高，捨棄這段影像，重新抓取10幀。")
#                 frames = []
#                 continue

#         frames.append(frame)
#         print(f"已收集 {len(frames)} 幀")

# # 當達到10幀時進行超解析處理
# if len(frames) == max_frames:
#     super_res_image = super_resolution_with_model(frames)
#     cv2.imshow('超解析度圖像', super_res_image)
#     cv2.imwrite('super_res_output.png', super_res_image)
#     cv2.waitKey(0)

# # 釋放攝影機
# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np

# 計算均方誤差 (MSE)，作為圖像相似度的簡易替代方法
def calculate_mse(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    mse = np.mean((gray1 - gray2) ** 2)
    return mse

# 超解析度重建函數：多幀平均 + 超解析模型
def super_resolution_with_model(frames, model_path='ESPCN_x4.pb'):
    # 將多幀平均
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    
    # 初始化超解析模型
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("espcn", 4)  # 設定模型為ESPCN，放大4倍
    
    # 對疊加圖像應用超解析
    super_res_image = sr.upsample(avg_frame)
    return super_res_image

# 讀取影片檔案
cap = cv2.VideoCapture('m1025_cut.mp4')

if not cap.isOpened():
    print("無法打開影片檔案")
    exit()

frames = []
count = 0
max_frames = 10  # 需要的幀數
mse_threshold = 8  # 設定均方誤差的門檻值

while len(frames) < max_frames:
    ret, frame = cap.read()
    if not ret:
        print("無法捕捉影像")
        break

    count += 1

    # 每10幀取1個frame
    if count % 10 == 0:
        if len(frames) > 0:
            last_frame = frames[-1]
            mse = calculate_mse(last_frame, frame)
            print(f"當前幀與前一幀的均方誤差: {mse}")

            if mse > mse_threshold:
                print("均方誤差過高，捨棄這段影像，重新抓取10幀。")
                frames = []
                continue

        frames.append(frame)
        print(f"已收集 {len(frames)} 幀")

# 當達到10幀時進行超解析處理
if len(frames) == max_frames:
    super_res_image = super_resolution_with_model(frames)
    cv2.imshow('超解析度圖像', super_res_image)
    cv2.imwrite('super_res_output.png', super_res_image)
    cv2.waitKey(0)

# 釋放資源
cap.release()
cv2.destroyAllWindows()
