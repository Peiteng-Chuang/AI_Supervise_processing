import cv2,os

#設置解析度(超過會預設相機最大解析度)
resolution_w, resolution_h=1920,1080

# 初始化相机
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_h)

# 检查相机是否成功开启
if not cap.isOpened():
    print("can't open capture")
    exit()

def count_jpg_files(directory):
    # 确认目录是否存在
    if not os.path.isdir(directory):
        print(f"path {directory} does not exist")
        return 0
    
    # 统计 .jpg 文件的数量
    jpg_count = len([file for file in os.listdir(directory) if file.lower().endswith('.jpg')])
    print(f"Updating images, find {jpg_count} jpg files in {directory}")
    return jpg_count


# 拍照保存次数控制

def main():
    current_pic_num=count_jpg_files("./take_picture/")
    space_press_count = 0
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"setting resolution to {frame_width}x{frame_height}")
    while True:
        # 读取相机画面
        ret, frame = cap.read()
        if not ret:
            print("无法接收画面 (相机已关闭)")
            break


        # 显示画面
        cv2.imshow('Camera', frame)

        # 等待按键输入
        key = cv2.waitKey(1) & 0xFF  

        # 按空白键两次拍照
        if key == ord(' '):
            space_press_count += 1
            if space_press_count == 2:  
                # 提取中心区域
                # cropped_frame = frame[top_left[1]+2:bottom_right[1]-2, top_left[0]+2:bottom_right[0]-2]
                cropped_frame = frame
                # 保存图片
                current_pic_num+=1
                cv2.imwrite(f'./saved_img/image_{current_pic_num}.jpg', cropped_frame)
                print(f"save picture as 'image_{current_pic_num}.jpg'")
                space_press_count = 0

        # 按ESC退出
        if key == 27:
            break

    # 释放相机并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
#================================================================