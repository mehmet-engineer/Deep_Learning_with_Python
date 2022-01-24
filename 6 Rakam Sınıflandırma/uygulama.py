import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# CPU ve GPU sorunu için

model = load_model("Model\\Rakamlar_model.h5")

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
    # 1 tane (32,32) boyutlu ve 1 gray kanallı

    predictions = model.predict(img)
    #her bir sınıf için...
    #Sınıfa ait olma olasılıklarını liste şeklinde ver

    value = np.max(predictions)
    # en büyük olasılığın değerini al

    classIndex = int(model.predict_classes(img))
    #en büyük olasılığın sınıf indexini ver

    value = round(value, 2)
    # 0.982121212 --> 0.98
    value = value * 100
    # 0.98 --> 98
    value = round(value)

    if value > 70:
        cv2.putText(frame,str(classIndex),(20,100),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,4,[255,0,0],3)
        cv2.putText(frame,"%"+str(value),(20,200),cv2.FONT_HERSHEY_DUPLEX,2,[0,255,0])
    
    cv2.imshow("img",frame)

    if cv2.waitKey(25) == 27:
        break

cam.release()
cv2.destroyAllWindows()