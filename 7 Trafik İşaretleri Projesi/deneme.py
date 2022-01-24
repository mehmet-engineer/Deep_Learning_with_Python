import os
import cv2
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

images = []
classNo = []
class_sayısı = 7
path = "Traffic_Data"

for i in range(class_sayısı):
    img_folders = os.listdir(path + "\\" + str(i))
    for j in img_folders:
        img = cv2.imread(path + "\\" + str(i) + "\\" + j)
        img = cv2.resize(img, (32,32))
        images.append(img)
        classNo.append(i)

print("resimler hafızaya alındı...")

images = np.array(images)
labels = np.array(classNo)
new_images = []

def scaling_process(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

for img in images:
    img = scaling_process(img)
    new_images.append(img)

print("preprocessing işlemi bitti...")

new_images = np.array(new_images)

x_train,x_test,y_train,y_test = train_test_split(new_images,labels,test_size=0.1)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=0.2)

x_train = x_train.reshape(-1,32,32,1)
x_test = x_test.reshape(-1,32,32,1)
x_validation = x_validation.reshape(-1,32,32,1)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
dataGenerator = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   rotation_range=10)
dataGenerator.fit(x_train)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train,class_sayısı)
y_test = to_categorical(y_test,class_sayısı)
y_validation = to_categorical(y_validation,class_sayısı)

print("model oluşturuluyor...")

model = models.Sequential()
girdi = (32,32,1)

model.add(layers.Conv2D(32,kernel_size=(5,5),input_shape=girdi,padding="same",activation="relu"))
model.add(layers.Conv2D(64,kernel_size=(3,3),padding="same",activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2)))         
model.add(layers.Conv2D(32,kernel_size=(5,5),padding="same",activation="relu"))
model.add(layers.Conv2D(32,kernel_size=(3,3),padding="same",activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
          
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(class_sayısı, activation="softmax"))
          
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

batch = 40
generator = dataGenerator.flow(x_train,y_train,batch_size=batch)
steps = 80

print("eğitim başlatılıyor...")

history = model.fit_generator(generator, epochs=12,
                              validation_data = (x_validation,y_validation),
                              steps_per_epoch = steps, shuffle=1)

score_train = model.evaluate(x_train,y_train)
print("Eğitim Doğruluğu: %",score_train[1]*100)
score_test = model.evaluate(x_test,y_test)
print("Test Doğruluğu: %",score_test[1]*100)

model.save("model_trained.h5")

print()
print("Eğitim tamamlandı, model kaydedildi.")