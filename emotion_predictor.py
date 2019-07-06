import pandas as pd
import cv2
import numpy as np
from keras.models import load_model

labels = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

model = load_model('emotion_trainer.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
i=0
while True:
    _,frame = video_capture.read()
	
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray,1.5,5)
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=roi_gray.reshape(1,48,48,1)
        s = model.predict(roi_gray)
        ans=s.argmax()
       
        if ans==0:
            cv2.putText(frame,labels[0],(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1,cv2.LINE_AA)
        elif ans==1:
            cv2.putText(frame,labels[1],(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1,cv2.LINE_AA)
        elif ans==2:
            cv2.putText(frame,labels[2],(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1,cv2.LINE_AA)
        elif ans==3:
            cv2.putText(frame,labels[3],(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1,cv2.LINE_AA)
        elif ans==4:
            cv2.putText(frame,labels[4],(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1,cv2.LINE_AA)
        elif ans==5:
            cv2.putText(frame,labels[5],(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1,cv2.LINE_AA)
        elif ans==6:
            cv2.putText(frame,labels[6],(x,y),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),1,cv2.LINE_AA)
        
    cv2.imshow('Video',frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
