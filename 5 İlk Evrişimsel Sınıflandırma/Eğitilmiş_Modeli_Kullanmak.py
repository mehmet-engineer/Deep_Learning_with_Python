import glob
import uuid
import numpy as np
from mss import mss
from PIL import Image
from tensorflow.keras.models import load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# CPU ve GPU sorunu için

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

img = screen.grab(bolge)
im = Image.frombytes("RGB",img.size,img.rgb)
im2 = np.array(im.convert("L").resize((width,height)))
im2 = im2 / 255
X = np.array([im2])
X = X.reshape(X.shape[0],width,height,1) 
results = model.predict(X)
result_index = np.argmax(results)

print("Down: {}".format(results[0][0]))
print("Right: {}".format(results[0][1]))
print("Up: {}".format(results[0][2]))