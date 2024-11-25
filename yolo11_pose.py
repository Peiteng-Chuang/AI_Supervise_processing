from ultralytics import YOLO
import cv2
# Load a pre-trained model
model = YOLO("yolo11s-pose.pt")

img = cv2.imread('./pose_img/working_sig.png')
# img = cv2.imread('./pose_img/runner.jpg')


results=model.predict(img)
 
# The keypoints are mapped in this order
# - 0: Nose
# - 1: Left Eye
# - 2: Right Eye
# - 3: Left Ear
# - 4: Right Ear
# - 5: Left Shoulder
# - 6: Right Shoulder
# - 7: Left Elbow
# - 8: Right Elbow
# - 9: Left Wrist
# - 10: Right Wrist
# - 11: Left Hip
# - 12: Right Hip
# - 13: Left Knee
# - 14: Right Knee
# - 15: Left Ankle
# - 16: Right Ankle
for r in results:
    print(r.keypoints.data)

for result in results:
    kpts = result.keypoints
    nk = kpts.shape[1]
    
    for i in range(nk):
        keypoint = kpts.xy[0, i]
        x, y = int(keypoint[0].item()), int(keypoint[1].item())
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

cv2.imshow('Output', img)
cv2.imwrite("./pose_img/worker pose.png",img)
cv2.waitKey(0)