import cv2
import time
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)

mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh(max_num_faces=1)

mpdraw = mp.solutions.drawing_utils

drawspec = mpdraw.DrawingSpec(thickness=1, circle_radius=1)

itime = 0

while True:
    ret, frame = cap.read()
    imgrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = facemesh.process(imgrgb)

    if result.multi_face_landmarks:
        for facelms in result.multi_face_landmarks:
            mpdraw.draw_landmarks(frame, facelms, mpfacemesh.FACEMESH_TESSELATION, drawspec, drawspec)

            left_eye = (int(facelms.landmark[23].x * frame.shape[1]), int(facelms.landmark[23].y * frame.shape[0]))
            left_eye1 = (int(facelms.landmark[27].x * frame.shape[1]), int(facelms.landmark[27].y * frame.shape[0]))

            line_length = math.sqrt((left_eye1[0] - left_eye[0]) ** 2 + (left_eye1[1] - left_eye[1]) ** 2)
            line_length = round(line_length, 2)

            cv2.putText(frame, f"Line Length: {line_length}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if line_length < 26:
                text="The eye closed"
            else:
                text="The eye opened"
            cv2.putText(frame, text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    stime = time.time()
    fps = 1 / (stime - itime)
    itime = stime
    cv2.putText(frame, f"FPS: {int(fps)}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Window", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
