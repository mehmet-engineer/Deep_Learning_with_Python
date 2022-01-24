#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt


# In[2]:


dataFrame = pd.read_excel("bisiklet_fiyatlari.xlsx")


# In[3]:


dataFrame


# In[4]:


# y = wx + b
# y --> label yani fiyat
y = dataFrame["Fiyat"].values  #numpy array formatına çevir

# x --> feature yani bisiklet özellikleri
x = dataFrame[["BisikletOzellik1","BisikletOzellik2"]].values


# In[5]:


from sklearn.model_selection import train_test_split
#veri setini hem eğitim hem de test için ağırlıklı bi şekilde ikiye ayırmak için

#test_size --> test için ayrılacak veri yüzdesi %33
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33)


# In[6]:


x_train


# In[10]:


from sklearn.preprocessing import MinMaxScaler
#scaling --> tüm x değerlerini 0-1 arası değerlere ölçeklendirme

scaler = MinMaxScaler()
scaler.fit(x_train)      #ön hazırlık için uydur

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[11]:


x_train


# In[12]:


import tensorflow as tf
from tensorflow.keras import models   # model sınıfı
from tensorflow.keras import layers   # katman sınıfı


# In[13]:


# Sinir Ağı Modeli Oluşturma

model = models.Sequential()

model.add(layers.Dense(4, activation="relu"))  #1.gizli katman ve 4 nöron
model.add(layers.Dense(4, activation="relu"))  #2.gizli katman ve 4 nöron
model.add(layers.Dense(4, activation="relu"))  #3.gizli katman ve 4 nöron

model.add(layers.Dense(1))   #çıkış katmanı


# In[14]:


# optimizasyon
model.compile(optimizer="rmsprop", loss="mse")


# In[15]:


# eğitme
model.fit(x_train,y_train, epochs=250)


# In[17]:


# Loss değerlerinin minimize olma eğrisi
loss = model.history.history["loss"]
axis = range(0,len(loss))
plt.plot(axis,loss)


# In[18]:


# Loss kayıplarının değerlendirilmesi
# train ve test kayıpları ne kadar az olursa o kadar iyidir
# train ve test kayıplarının birbirine yakın değerler olması sağlıklıdır

trainLoss = model.evaluate(x_train,y_train,verbose=0)
testLoss = model.evaluate(x_test,y_test,verbose=0)


# In[19]:


trainLoss


# In[20]:


testLoss


# In[26]:


# Tahmin denemeleri

tahminler = model.predict(x_test)
# y --> fiyat tahminlerini numpy array şeklinde verir

tahminler.shape


# In[27]:


tahminler = pd.Series(tahminler.reshape(330,))
tahminler


# In[28]:


# Tahminlerin gerçek değerlerle karşılaştırılması

compareFrame = pd.DataFrame(y_test,columns=["Gerçek Fiyat"])
compareFrame = pd.concat([compareFrame,tahminler],axis=1)
compareFrame.columns = ["Gerçek Fiyat","Tahmin"]


# In[29]:


compareFrame


# In[44]:


plt.scatter(y_test,tahminler)


# In[46]:


# Hata oranı kabul edilebilir mi?
from sklearn.metrics import mean_absolute_error

mean_absolute_error(compareFrame["Gerçek Fiyat"],compareFrame["Tahmin"])


# In[47]:


dataFrame["Fiyat"].mean()


# In[48]:


# ortalama 872 liralık fiyatlardan ortalama 6.94 tl sapabilir


# In[57]:


# modeli kaydetme
model.save("bisiklet_modeli.h5")

# modeli alma
# from tensorflow.keras.models import load_model
# my_model = load_model("bisiklet_modeli.h5")


# In[ ]:




