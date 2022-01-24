#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


dataFrame = pd.read_excel("websites.xlsx")
dataFrame


# In[5]:


y = dataFrame["Type"].values
x = dataFrame.drop("Type",axis=1).values


# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.28)


# In[7]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[21]:


import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers

model1 = models.Sequential()

#kaç tane sütun (x özelliği) varsa ilk katmanda o kadar nörön olması önerilir
model1.add(layers.Dense(30,activation="relu"))

#çıkış katmanı nöron sayısı ile ilk katmandaki nöron sayısı arasında olması önerilir 
model1.add(layers.Dense(15,activation="relu"))
model1.add(layers.Dense(15,activation="relu"))

#binary classification ve 0-1 arası çıktılar için sigmoid fonksiyonu
model1.add(layers.Dense(1,activation="sigmoid"))

model1.compile(optimizer="adam",loss="binary_crossentropy")


# In[22]:


model1.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=700)


# In[23]:


loss_df = model1.history.history

loss_train = loss_df["loss"]
loss_train = np.array(loss_train)
loss_test = loss_df["val_loss"]
loss_test = np.array(loss_test)

axis = range(0,700)
plt.plot(axis,loss_train, label="loss")
plt.plot(axis,loss_test, label="val_loss")
plt.legend()

# overfitting aşırı uydurma durumu (epoch fazla geldi)


# In[25]:


# modeli tekrar oluşturma

model2 = models.Sequential()

model2.add(layers.Dense(30,activation="relu"))
model2.add(layers.Dense(15,activation="relu"))
model2.add(layers.Dense(15,activation="relu"))
model2.add(layers.Dense(1,activation="sigmoid"))

model2.compile(optimizer="adam",loss="binary_crossentropy")


# In[26]:


# Early Stopping (Erken Durdurma) ayarlı eğitme
from tensorflow.keras import callbacks

stopping = callbacks.EarlyStopping(monitor="val_loss",mode="min",patience=25,verbose=1)
# val_loss'u minimum olacak şekilde takip et, 25 ayarında erken durdurma yap

model2.fit(x_train, y_train, validation_data=(x_test,y_test),
          epochs=700, callbacks=[stopping])


# In[28]:


loss_df = model2.history.history

loss_train = loss_df["loss"]
loss_train = np.array(loss_train)
loss_test = loss_df["val_loss"]
loss_test = np.array(loss_test)

axis = range(0,52)  # 52. epoch'da erken durdurma
plt.plot(axis,loss_train, label="loss")
plt.plot(axis,loss_test, label="val_loss")
plt.legend()


# In[30]:


# Dropout (fazla nöronların düşürülmesi - kısa süreli unutma)
# modeli tekrar oluşturma

model3 = models.Sequential()

model3.add(layers.Dense(30,activation="relu"))
model3.add(layers.Dropout(0.4))   #rastgele nöronları düşürmeyi dene

model3.add(layers.Dense(15,activation="relu"))
model3.add(layers.Dropout(0.4))

model3.add(layers.Dense(15,activation="relu"))
model3.add(layers.Dropout(0.4))

model3.add(layers.Dense(1,activation="sigmoid"))

model3.compile(optimizer="adam",loss="binary_crossentropy")


# In[31]:


model3.fit(x_train, y_train, validation_data=(x_test,y_test),
          epochs=700, callbacks=[stopping])


# In[32]:


loss_df = model3.history.history

loss_train = loss_df["loss"]
loss_train = np.array(loss_train)
loss_test = loss_df["val_loss"]
loss_test = np.array(loss_test)

axis = range(0,70)  # 70. epoch
plt.plot(axis,loss_train, label="loss")
plt.plot(axis,loss_test, label="val_loss")
plt.legend()


# In[36]:


# sınıflandırma sonuçlarını tahmin et

tahminler = model3.predict_classes(x_test)
tahminler


# In[37]:


sayı = tahminler.shape[0]
tahminler = pd.Series(tahminler.reshape(sayı,))

resultFrame = pd.DataFrame(y_test,columns=["Gerçek Tür"])
resultFrame["Tahmin"] = tahminler
resultFrame


# In[40]:


# sınıflandırma sonuçlarını değerlendirme

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test,tahminler))
# %93 accuracy score


# In[41]:


print(confusion_matrix(y_test,tahminler))


# In[42]:


# 0'lar (virüs siteleri tahmininde) 90 doğru, 5 yanlış / %94 doğruluk
# 1'lar (güvenli siteleri tahmininde) 53 doğru, 6 yanlış / %91 doğruluk


# In[ ]:




