import cv2
import numpy as np

def hdr_capture(camera_index=0, exposure_values=None, output_file="hdr_output.hdr"):
    if exposure_values is None:
        exposure_values = [-5,-2,-3]  # 調整曝光值範圍

    resolution_w, resolution_h = 1920, 1080
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # 加速初始化
    if not camera.isOpened():
        print("無法開啟攝像頭！")
        return
    cv2.waitKey(1000)  # 延遲1秒

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_w)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_h)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 禁用自動曝光
    camera.set(cv2.CAP_PROP_AUTO_WB, 0)           # 禁用自動白平衡
    camera.set(cv2.CAP_PROP_WB_TEMPERATURE, 4000) # 設置白平衡色溫
    camera.set(cv2.CAP_PROP_GAIN, 0)              # 禁用自動增益

    captured_images = []
    print("開始拍攝...")

    for exp in exposure_values:
        camera.set(cv2.CAP_PROP_EXPOSURE, exp)
        print(f"設置曝光值: {exp}")
        cv2.waitKey(1000)  # 等待設置生效

        # 捕獲穩定幀
        for _ in range(10):
            ret, frame = camera.read()
        if not ret:
            print(f"無法拍攝曝光值為 {exp} 的圖像！")
            continue

        # 轉換為灰階圖像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame,cv2.COLOR_GRAY2BGR)
        captured_images.append(gray_frame)
        # cv2.imshow("Captured Image (Grayscale)", gray_frame)
        cv2.waitKey(500)

    camera.release()
    cv2.destroyAllWindows()

    if len(captured_images) < len(exposure_values):
        print("拍攝失敗，無法合成HDR圖像。")
        return

    print("開始合成HDR圖像...")
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(captured_images, captured_images)

    times = np.array([1.0 / (2**abs(exp)) for exp in exposure_values], dtype=np.float32)
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(captured_images, times)

    merge_debevec = cv2.createMergeDebevec()
    hdr_image = merge_debevec.process(captured_images, times, response)

    # 灰階 HDR 合成後，OpenCV 可能會將其保留為三通道，需再轉成單通道
    hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(output_file, hdr_image)
    print(f"HDR 圖像已保存為 {output_file}")

    # 不進行色調映射，直接顯示灰階 HDR 圖像
    cv2.imshow("HDR Result (Grayscale)", hdr_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

hdr_capture(output_file="hdr_image_gray.png")
