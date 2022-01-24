#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import cv2
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

from tensorflow.keras import models
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split


# In[3]:


images = []
classNo = []
class_sayısı = 10
path = "myData"

for i in range(class_sayısı):
    img_folders = os.listdir(path + "\\" + str(i))
    
    for j in img_folders:
        img = cv2.imread(path + "\\" + str(i) + "\\" + j)
        img = cv2.resize(img, (32,32))
        images.append(img)
        classNo.append(i)
    
len(images)   #toplam veri sayısı


# In[4]:


# her bir rakamdan kaç tane resim verisi var?
print(classNo.count(0))
print(classNo.count(1))
print(classNo.count(2))


# In[5]:


images = np.array(images)
labels = np.array(classNo)
new_images = []
print(images.shape)

def scaling_process(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img
    
for img in images:
    img = scaling_process(img)
    new_images.append(img)
    
new_images = np.array(new_images)
    
print(new_images.shape)


# In[6]:


#veriyi validasyonlu bir şekilde ayırma
x_train,x_test,y_train,y_test = train_test_split(new_images,labels,test_size=0.4)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=0.2)

print(images.shape)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)


# In[7]:


x_train = x_train.reshape(-1,32,32,1)
x_test = x_test.reshape(-1,32,32,1)
x_validation = x_validation.reshape(-1,32,32,1)

print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)


# In[8]:


#evrişimsel sinir ağı resim içindeki nesne konumuna duyarlıdır
#x_train verisini zoom, rotasyon gibi değişiklik yaparak çeşitlendir
from tensorflow.keras.preprocessing.image import ImageDataGenerator
dataGenerator = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   rotation_range=10)
dataGenerator.fit(x_train)


# In[9]:


#farklı bir şekilde OneHotEncoder işlemi
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train,class_sayısı)
y_test = to_categorical(y_test,class_sayısı)
y_validation = to_categorical(y_validation,class_sayısı)


# In[10]:


model = models.Sequential()

girdi = (32,32,1)

# "same" padding --> 1 sıra padding
model.add(layers.Conv2D(8,kernel_size=(5,5),input_shape=girdi,padding="same",activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(16,kernel_size=(3,3),padding="same",activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Dropout(0.2))
model.add(layers.Flatten())

#Sınıflandırma Katmanları
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(class_sayısı, activation="softmax"))

#optimizasyon
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[11]:


batch_size = 100
generator = dataGenerator.flow(x_train,y_train,batch_size=batch_size)
steps = x_train.shape[0] // batch_size
# 4876 // 100 --> steps = 48    (kalansız bölüm)
# shuffle --> veriyi karıştırır

history = model.fit_generator(generator, epochs = 20,
                              validation_data = (x_validation,y_validation),
                              steps_per_epoch = steps, shuffle = 1)


# In[12]:


history.history.keys()


# In[13]:


fig,axes = plt.subplots(1,2, figsize=(10,4))
fig.suptitle("Loss ve Accuracy")

axes[0].plot(history.history["loss"], label="train loss")
axes[0].plot(history.history["val_loss"], label="validation loss")
axes[0].set_title("Loss Değerleri")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="train accuracy")
axes[1].plot(history.history["val_accuracy"], label="validation accuracy")
axes[1].set_title("Accuracy Değerleri")
axes[1].legend()

plt.tight_layout()
plt.show()


# In[14]:


score_train = model.evaluate(x_train,y_train)
print("Eğitim Doğruluğu: %",score_train[1]*100)
score_test = model.evaluate(x_test,y_test)
print("Test Doğruluğu: %",score_test[1]*100)


# In[15]:


# Resim Tahminleri Doğruluk Skalası
from sklearn.metrics import confusion_matrix

y_predict = model.predict(x_test)
y_predict_class = np.argmax(y_predict, axis = 1)
Y_true = np.argmax(y_test, axis = 1)

cm = confusion_matrix(Y_true, y_predict_class)
fig, axes = plt.subplots(figsize=(8,8))
sbn.heatmap(cm, annot = True, linewidths = 0.01, cmap = "Greens",
            linecolor = "gray", fmt = ".1f", ax=axes)
plt.xlabel("Predicted")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# In[16]:


open("Rakamlar_model.json","w").write(model.to_json())
model.save("Rakamlar_model.h5")
model.save_weights("Rakamlar_weights.h5")


# In[ ]:




