import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ultralytics_path = "C:/project_file/ultralytics_v11"

# # 加载邊角檢測模型
# egdt_model_ver = "v13"
# egdt_model = YOLO(f'{ultralytics_path}/runs/detect/sp_egdt_{egdt_model_ver}/weights/best.pt', verbose=False)
# print(f"egdt_Model loaded successfully! model version: {egdt_model_ver}")
# egdt_model = egdt_model.to(device)

# 加载孔洞檢測模型
det_model_ver = "v12"
det_model = YOLO(f'{ultralytics_path}/runs/detect/sp_holes_{det_model_ver}/weights/best.pt', verbose=False)
print(f"det_Model loaded successfully! model version: {det_model_ver}")
det_model = det_model.to(device)

def get_image_difference(img1, img2, det_model=det_model):
    size = (1920, 1080)
    img1 = cv2.resize(img1, size, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, size, interpolation=cv2.INTER_AREA)

    # 儲存兩張圖片中的檢測框
    boxes_img1 = []  # 過去圖片的檢測框
    boxes_img2 = []  # 目前圖片的檢測框

    # 進行物體檢測（過去圖片）
    results_1 = det_model(img1, verbose=False)
    for result in results_1:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes_img1.append((x1, y1, x2, y2))

    # 進行物體檢測（目前圖片）
    results_2 = det_model(img2, verbose=False)
    for result in results_2:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes_img2.append((x1, y1, x2, y2))

    # 判斷新增或移動的物件
    def is_different(box, boxes_reference, threshold=5):
        """檢查當前檢測框是否與參考框清單中的任一框不匹配"""
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 計算中心點
        for ref_box in boxes_reference:
            rx1, ry1, rx2, ry2 = ref_box
            rcx, rcy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
            if abs(cx - rcx) < threshold and abs(cy - rcy) < threshold:
                return False  # 與參考框匹配
        return True  # 不匹配，視為新增或移動的物件

    # 找出在 img2 中新增或移動的框
    different_boxes = [box for box in boxes_img2 if is_different(box, boxes_img1)]

    # 在 img2 上繪製標記
    for x1, y1, x2, y2 in different_boxes:
        cv2.rectangle(img2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # 紅色框表示差異物件
        cv2.putText(img2, "New/Changed", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img2

def main(folder_path):
    """
    主函式：讀取資料夾中的圖片並依序顯示。
    """
    # 獲取資料夾中的圖片並排序
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    image_files.sort()

    if not image_files:
        print("資料夾中沒有圖片檔案！")
        return

    # 初始化變數
    prev_image = None
    index = 0

    while index < len(image_files):
        image_path = os.path.join(folder_path, image_files[index])
        current_image = cv2.imread(image_path)

        if prev_image is not None:
            display_image = get_image_difference(prev_image, current_image.copy())
        else:
            display_image = current_image.copy()

        cv2.imshow("Image Viewer", display_image)

        key = cv2.waitKey(0)

        if key == 27:  # 按下 ESC 鍵中斷
            print("程式中斷。")
            break
        elif key == 32:  # 按下空白鍵切換
            prev_image = current_image
            index += 1
        else:
            print("無效按鍵，請按空白鍵切換或 ESC 鍵中斷。")

    print("所有圖片已顯示完成。")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = "C:/Users/Peiteng.Chuang/Desktop/factor/image/20241127"  #C:/Users/Peiteng.Chuang/Desktop/diff_good
    main(folder_path)