import cv2
import time
import mediapipe as mp

cap=cv2.VideoCapture(0)

mphand=mp.solutions.hands
hands=mphand.Hands()

mpdraw=mp.solutions.drawing_utils

itime=0
while True:

    ret,frame=cap.read()
    imgrgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=hands.process(imgrgb)

    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(frame,handlms,mphand.HAND_CONNECTIONS)

        for id,lm in enumerate(handlms.landmark):
            h,w,_=frame.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            if id==8:
                cv2.circle(frame,(cx,cy),3,(255,0,0),-1)
                cv2.putText(frame,"8",(cx-10,cy-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    stime=time.time()
    fps=1/(stime-itime)
    itime=stime
    cv2.putText(frame,str(int(fps)),(30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.putText(frame, "FPS", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Window",frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()