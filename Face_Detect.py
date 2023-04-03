import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def print_utf8_text(image,xy,text,color):
    fontName='FreeSerif.ttf'
    font = ImageFont.truetype(fontName,24)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((xy[0],xy[1]),text,font=font,
              fill=(color[0],color[1],color[2],0))
    image = np.array(img_pil)
    return image

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Train/train.yml')
cascadePath="Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font=cv2.FONT_HERSHEY_SIMPLEX

id = 0
names =['None','Ahmet',"Fadik"]

cam = cv2.VideoCapture("/dev/video0")
cam.set(3,1000)
cam.set(4,800)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while(True):
    ret,img=cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize=(int(minW),int(minH)),
    )
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,uyum = recognizer.predict(gray[y:y+h, x:x+w])
        print("\n Uyum: ",uyum)
        id=int(id/100)
        if(uyum<100):
            print("\n id: ",id)
            id = names[id]
            uyum=f"Uyum: {round(uyum,0)}%"
        else:
            id="Unknown"
            uyum=f"Uyum: {round(uyum,0)}%"
        color = (255,255,255)
        img=print_utf8_text(img,(x+5,y-25),str(id),color)
        cv2.putText(img,str(uyum),(x+5,y+h+25),font,1,(0,255,0),1)
    cv2.imshow("Frame",img)
    k=cv2.waitKey(10 & 0xff)
    if k==27 or k==ord('q'):
        break

print("\n [INFO]: Memory Clearing!")
print("[INFO]: Program Terminating...")
cam.release()
cv2.destroyAllWindows()

