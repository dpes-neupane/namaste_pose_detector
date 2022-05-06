import mediapipe.python as mp
import cv2 as cv
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
with open("./model.pkl", "rb") as fp:
    model = pickle.load(fp)
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,  min_tracking_confidence=0.7)
draw_utils = mp.solutions.drawing_utils

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    
    if results.pose_landmarks:
        
        draw_utils.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        pl = [[lmk.x, lmk.y, lmk.z, lmk.visibility] for lmk in results.pose_landmarks.landmark] 
        pl = np.around(pl, 5).flatten()
        y = model.predict(pl.reshape(1, -1))
        if y == 1:
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame, "namaste", (10, 100), font, 1, (255, 0, 0), 2)
        
    cv.imshow("frame", frame)
    
    if cv.waitKey(10) & 0xFF == ord("q"):
        print("........")
        break
cap.release()
cv.destroyAllWindows()
