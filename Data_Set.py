import cv2

cam=cv2.VideoCapture('/dev/video0')
cam.set(3,640)
cam.set(4,480)
face_detector=cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
MAXFOTOADET=60
face_id=1
print("\n[INFO]: started to Record")
count=10

while(True):
    ret,img=cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        count +=1
        cv2.imwrite("dataSet/"+str(face_id)+'_'+str(count)+".jpg",gray[y:y+h,x:x+w])
        cv2.imshow('Frame',img)
        print("Recording Count: ",count)
    k=cv2.waitKey(100) & 0xff
    if k==27:
        break
    elif count>=MAXFOTOADET:
        break
print("\n [INFO]: Clear the Memory && program is terminated")
cam.release()
cv2.destroyAllWindows()