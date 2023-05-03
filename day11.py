import cv2
import numpy as np
import matplotlib as plt
from time import sleep

fd = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
sm = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

# face = fd.detectMul
vid = cv2.VideoCapture(0)

captured = True
while captured:
    flag,img = vid.read()
    if flag:
        #processing code
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = fd.detectMultiScale(img_gray,1.1,5)
        smiles = sm.detectMultiScale(img_gray,1.1,5)
        # th,img_bw = cv2.threshold(img_gray,170,255,cv2.THRESH_BINARY)
        # m = np.random.randint(0,256)
        # n = np.random.randint(0,256)
        # o = np.random.randint(0,256)
        np.random.seed(50)
        colors = [np.random.randint(0,256,3).tolist() for i in smiles]
        i=0
        for x,y,w,h in faces:
            face = faces[y:y+h,x:x+w].copy()
            smiles = sm.detectMultiScale(
                face,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = (50,50)
            )
            if(len(smiles) == 1):
                cv2.imwrite('myselfie.png',img)
                captured = False
                break

            cv2.rectangle(
                img, pt1 = (x,y), pt2=(x+w,y+h), color=colors[i],thickness=8
            )
            i+=1
        
        cv2.imshow('preview',img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    else:
        print("no frames")
        break
vid.release()    
cv2.destroyAllWindows()