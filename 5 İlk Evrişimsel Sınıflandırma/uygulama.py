import time
import pyautogui
import glob
import uuid
import numpy as np
from mss import mss
from PIL import Image
from tensorflow.keras.models import load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# CPU ve GPU sorunu içindi

"""
http://www.trex-game.skipser.com/
"""

bolge = {"top":294, "left":722, "width":250, "height":140}
width = 125
height = 70

screen = mss()

model = load_model("model\\TREX_model.h5")

""" bu şekilde de model alınabilirdi;
from tensorflow.keras.models import model_from_json
model = model_from_json(open("model\\TREX_model.json","r").read())
model.load_weights("model\\TREX_weights.h5")
"""

# int_encoded --> up=2 / down=0 / right=1
labels = ["Down","Right","Up"]

frame_time = time.time()  #şuanki zaman
counter = 0
i = 0
delay = 0.4
isDOWN = False

time.sleep(5)

while True:

    img = screen.grab(bolge)
    im = Image.frombytes("RGB",img.size,img.rgb)
    im2 = np.array(im.convert("L").resize((width,height)))
    im2 = im2 / 255
    X = np.array([im2])
    X = X.reshape(X.shape[0],width,height,1) 
    results = model.predict(X)
    result_index = np.argmax(results)

    if result_index == 0:
        pyautogui.keyDown("down")
        time.sleep(0.2)
        isDOWN = True

    elif result_index == 2:

        if isDOWN == True:
            pyautogui.keyUp("down")
            isDOWN = False

        pyautogui.keyDown("up")
        time.sleep(0.2)

        #oyun hızlanıyor...
        if i < 1500:
            time.sleep(0.3)
        elif 1500 < i and i < 5000:
            time.sleep(0.2)
        else:
            time.sleep(0.17)

        #initial position'a geri dön
        pyautogui.keyDown("down")
        time.sleep(0.2)
        pyautogui.keyUp("down")

    counter = counter + 1

    #geçen süreyi hesapla (1 saniye geçmişse)
    if (time.time() - frame_time) > 1:
        counter = 0
        frame_time = time.time()

        """if i <= 1500:
            delay = delay - 0.003
        else:
            delay = delay - 0.005
        if delay < 0:
            delay = 0"""

        print("------------------------------")
        print("Down: {}".format(results[0][0]))
        print("Right: {}".format(results[0][1]))
        print("Up: {}".format(results[0][2]))

        i = i + 1