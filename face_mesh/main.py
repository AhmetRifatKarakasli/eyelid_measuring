import cv2
import time
import mediapipe as mp

cap=cv2.VideoCapture(0)

mpfacemesh=mp.solutions.face_mesh
facemesh=mpfacemesh.FaceMesh(max_num_faces=1)

mpdraw=mp.solutions.drawing_utils

drawspec=mpdraw.DrawingSpec(thickness=1,circle_radius=1)

itime=0
while True:

    ret,frame=cap.read()
    imgrgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=facemesh.process(imgrgb)
    #print(result.multi_face_landmarks)
    if result.multi_face_landmarks:
        for facelms in result.multi_face_landmarks:
            mpdraw.draw_landmarks(frame,facelms,mpfacemesh.FACEMESH_TESSELATION,drawspec,drawspec)

        for id,lm in enumerate(facelms.landmark):
            h,w,_=frame.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            #print([id,cx,cy])
            #159=lefteye
            if id==159:
                cv2.circle(frame,(cx,cy),3,(255,0,0),-1)

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