#!/usr/bin/env python
# coding: utf-8

# In[62]:


import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# In[63]:


images_up = glob.glob("./veriseti/up/*.png")
images_down = glob.glob("./veriseti/down/*.png")
images_right = glob.glob("./veriseti/right/*.png")


# In[64]:


width = 125
height = 70

X = []   #resimler
Y = []   #label yani etiket


# In[65]:


for img in images_up:
    im = np.array(Image.open(img).convert("L").resize((width,height)))
    im = im / 255    # 0-1 arası ölçeklendirme
    X.append(im)
    Y.append("up")
    
for img in images_down:
    im = np.array(Image.open(img).convert("L").resize((width,height)))
    im = im / 255
    X.append(im)
    Y.append("down")
    
for img in images_right:
    im = np.array(Image.open(img).convert("L").resize((width,height)))
    im = im / 255
    X.append(im)
    Y.append("right")


# In[66]:


X = np.array(X)    # listeyi array'e çevirme
X = X.reshape(X.shape[0], width, height, 1)     # shape --> (289, 125, 70, 1)
#shape[0] toplam resim sayısı --- 1 parametresi siyah beyaz kanala çevirir


# In[67]:


A = ["up","down","right"]
B = [Y.count("up"),Y.count("down"),Y.count("right")]
plt.bar(A,B)    #veri setinin dağılımı


# In[68]:


# up,down,right label'larını 0-1-2 şeklinde kodlama
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
int_encoded = encoder.fit_transform(Y)
int_encoded # up=2 / down=0 / right=1


# In[69]:


int_encoded = int_encoded.reshape(len(int_encoded),1)

from sklearn.preprocessing import OneHotEncoder
OneEncoder = OneHotEncoder(sparse=False)
OneEncoder = OneEncoder.fit_transform(int_encoded)
Y_encoded = OneEncoder
Y_encoded
# 001 --> 2 up
# 100 --> 0 down
# 010 --> 1 right


# In[70]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y_encoded,test_size=0.25)


# In[71]:


import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
girdi = (width,height,1) #başlangıç katmanında belirtmek yeterli

#Evrişim Katmanları
model.add(layers.Conv2D(32,kernel_size=(3,3),input_shape=girdi,activation="relu"))
model.add(layers.Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())

#Sınıflandırma Katmanları
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(3, activation="softmax"))  #3 label çıktısı

#optimizasyon (metrics --> değerlendirmeyi doğruluğa göre ölç)
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[72]:


# Eğitme
# Eğitim yaparken CPU %92'lerde işlem yaptı Batch oranını düşürebilirsin
model.fit(x_train,y_train,epochs=30,batch_size=64)


# In[73]:


score_train = model.evaluate(x_train,y_train)
print("Eğitim Doğruluğu: %",score_train[1]*100)

score_test = model.evaluate(x_test,y_test)
print("Test Doğruluğu: %",score_test[1]*100)


# In[74]:


#Modeli Kaydetme
open("TREX_model.json","w").write(model.to_json())
model.save("TREX_model.h5")
model.save_weights("TREX_weights.h5")


# In[ ]:




