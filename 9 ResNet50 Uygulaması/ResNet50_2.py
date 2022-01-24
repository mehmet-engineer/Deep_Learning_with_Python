import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, exists, expanduser
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

fig, ax = plt.subplots(1, figsize=(12, 10))
img = image.load_img('images/dog.png')
img = image.img_to_array(img)
ax.imshow(img / 255.) 
ax.axis('off')
plt.show()

resnet = ResNet50(weights='imagenet')
img = image.load_img('images/dog.png', target_size=(224, 224))
img = image.img_to_array(img)
plt.imshow(img / 255.)
x = preprocess_input(np.expand_dims(img.copy(), axis=0))
preds = resnet.predict(x)
print(decode_predictions(preds, top=5))