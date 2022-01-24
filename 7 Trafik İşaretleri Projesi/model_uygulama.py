import os
import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# CPU ve GPU sorunu için

model = load_model("model_trained.h5")

def preprocessing(img):
    img = cv2.resize(img, (32,32))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

cam = cv2.VideoCapture(1)
cam.set(3, 480)
cam.set(4, 480)

while True:
    _,frame = cam.read()

    img = np.asarray(frame)   #array'a çevirme
    img = preprocessing(img)
    img = img.reshape(1,32,32,1)

    predictions = model.predict(img)

    value = np.amax(predictions)

    classIndex = int(model.predict_classes(img))

    value = round(value, 2)
    value = value * 100
    value = round(value)

    if value > 80:
        if classIndex == 0:
            cv2.putText(frame,"GIRILMEZ",(20,60),cv2.FONT_HERSHEY_PLAIN,3,[255,0,0],3)
        if classIndex == 1:
            cv2.putText(frame,"DUR",(20,60),cv2.FONT_HERSHEY_PLAIN,3,[255,0,0],3)
        if classIndex == 2:
            cv2.putText(frame,"YAYA GECIDI",(20,60),cv2.FONT_HERSHEY_PLAIN,3,[255,0,0],3)
        if classIndex == 3:
            cv2.putText(frame,"YOL CALISMASI",(20,60),cv2.FONT_HERSHEY_PLAIN,3,[255,0,0],3)
        if classIndex == 4:
            cv2.putText(frame,"YOL VER",(20,60),cv2.FONT_HERSHEY_PLAIN,3,[255,0,0],3)
        if classIndex == 5:
            cv2.putText(frame,"SOLLAMA YASAK",(20,60),cv2.FONT_HERSHEY_PLAIN,3,[255,0,0],3)
        if classIndex == 6:
            cv2.putText(frame,"MAX LIMIT 50",(20,60),cv2.FONT_HERSHEY_PLAIN,3,[255,0,0],3)

        cv2.putText(frame,"%"+str(value),(20,150),cv2.FONT_HERSHEY_DUPLEX,2,[0,255,0])
    
    cv2.imshow("img",frame)

    if cv2.waitKey(25) == 27:
        break

cam.release()
cv2.destroyAllWindows()