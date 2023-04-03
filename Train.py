import cv2
import numpy as np
from PIL import Image
import os


path='dataSet'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    ornekler=[]
    ids=[]
    for imagepath in imagePaths:
        print("tekil Path: ",imagepath)
        PIL_img = Image.open(imagepath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagepath)[-1].split(".")[0])
        print("id: ",id)
        faces= detector.detectMultiScale(img_numpy)
        for(x,y,w,h) in faces:
            ornekler.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return ornekler,ids
print("\n [INFO]: Faces are being trained. Wait a few seconds...")
faces,ids=getImagesAndLabels(path)
recognizer.train(faces,np.array(ids))
recognizer.write('Train/train.yml')
print(f"\n[INFO]: {len(np.unique(ids))} face trained. terminating...")
print(ids)