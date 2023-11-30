import cv2
import mediapipe as mp
import time
from ultralytics import YOLO

model = YOLO(r'Downloads\best.pt')
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)
pTime = 0

sitting_start_time = None
standing_start_time = None

happy_start_time = None
sad_start_time = None

happy_duration = 0
sad_duration = 0
sitting_duration = 0

window_timeout = 100  # Time to wait before closing the window (in seconds)
start_time = time.time()

cv2.namedWindow("Pose Detection")
cv2.namedWindow("Facial Expressions")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    ignore, img2 = cap.read()
    img2=cv2.flip(img2,1)
    imgbw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(imgRGB)

    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark

        # Get the y-coordinates of the hips and knees
        hip_y = landmarks[mpPose.PoseLandmark.LEFT_HIP].y + landmarks[mpPose.PoseLandmark.RIGHT_HIP].y
        knee_y = landmarks[mpPose.PoseLandmark.LEFT_KNEE].y + landmarks[mpPose.PoseLandmark.RIGHT_KNEE].y

        # Calculate the difference between hip and knee heights
        height_difference = hip_y - knee_y

        # Threshold values to determine standing or sitting
        standing_threshold = -0.3  # Tweak this value as needed

        # Determine if the person is standing or sitting
        if height_difference > standing_threshold:
            status = "Sitting"
            if sitting_start_time is None:
                sitting_start_time = time.time()
            standing_start_time = None
            if happy_start_time is not None:
                elapsed_time = time.time() - happy_start_time
                happy_duration += elapsed_time
            sitting_duration = time.time() - sitting_start_time
        else:
            status = "Standing"
            if standing_start_time is None:
                standing_start_time = time.time()
            sitting_start_time = None

        cv2.putText(img, status, (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        mpDraw.draw_landmarks(img, results_pose.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(landmarks):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cv2.imshow("Pose Detection", img)

    # Preprocess the image for YOLO model
    img_for_yolo = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    results_yolo = model(source=img_for_yolo, show=True, conf=0.4, save=False)
    
    current_time = time.time()
    if (current_time - start_time) > window_timeout:
        cv2.destroyAllWindows()
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Check if the duration of sitting exceeds 45 seconds
if sitting_duration >= 45:
    print("The Customer is Satisfied")
else:
    print('Customer Not Satisfied')
