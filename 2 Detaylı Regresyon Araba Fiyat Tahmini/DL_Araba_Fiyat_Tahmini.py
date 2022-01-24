#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn


# In[4]:


dataFrame = pd.read_excel("mercedes.xlsx")


# In[5]:


dataFrame


# In[6]:


dataFrame.describe()


# In[7]:


#null değerleri temizleme
dataFrame = dataFrame.dropna(axis=0) #null içeren satırları sil
dataFrame


# In[8]:


# veri dağılımını kontrol etme (histogram)
plt.hist(dataFrame["price"])


# In[9]:


# veri dağılımını kontrol etme (distribution)
sbn.distplot(dataFrame["price"])

# çok pahalı arabalar azınlıkta, modelimizi olumsuz etkileyebilir
# fiyatı 75 bin'den pahalıları silsek iyi olur
# zira onlar için ayrı bir model oluşturulabilir


# In[10]:


sbn.countplot(dataFrame["year"])


# In[11]:


# Korelasyon analizi
# veri özellikleri arasında doğrusal bir ilişki olup olmadığını,
# varsa bu ilişkinin katsayısını veren matematiksel yöntemdir.
# örneğin araba yılı ile araba fiyatı arasında pozitif korelasyon vardır
# az da olsa hata oranı barındırır

dataFrame.corr()["price"].sort_values()   #fiyatı etkileyen korelasyonlar


# In[12]:


# String Sütununu silme (Transmission sütunu)
dataFrame = dataFrame.drop("transmission", axis=1)
dataFrame


# In[13]:


# en pahalı arabaları listele
dataFrame.sort_values("price",ascending=False).head(50)


# In[14]:


# fiyatı yüksek olanlara göre sıralanmış dataframe üzerinde,
# ilk 50 tanesini bırak 50.sıradan sonuna kadar dataFrame'i güncelle

dataFrame = dataFrame.sort_values("price",ascending=False).iloc[50:]
sbn.distplot(dataFrame["price"])


# In[15]:


# araba yılına göre ortalama fiyatlar
dataFrame.groupby("year").mean()["price"]

# 1970 model arabaların ort fiyatı uyumsuz


# In[16]:


# bozuk veriyi sorgulayarak silme
dataFrame = dataFrame.query("year != 1970")
dataFrame.groupby("year").mean()["price"]


# In[17]:


# x (özellikler) ve y (label hedefi) değerlerini oluşturma

y = dataFrame["price"].values
x = dataFrame.drop("price",axis=1).values


# In[18]:


# veri setinin test-train oranına ayrılması

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)


# In[19]:


# veri ölçeklendirme

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[20]:


# modeli oluşturma

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
# 5 araba özelliği olduğundan bir katmanda en az 5 nöron olmalı

model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dense(1))

model.compile(optimizer="adam",loss="mse")


# In[21]:


# modeli eğitme

#validation --> ek olarak test verilerine göre de loss oranını çıkar
model.fit(x_train,y_train, validation_data=(x_test,y_test), batch_size=500, epochs=400)


# In[22]:


# Loss değerlerinin minimize olma eğrisi

loss_df = model.history.history

loss_train = loss_df["loss"]     # type --> liste
loss_train = np.array(loss_train)   # train verisine göre loss
loss_test = loss_df["val_loss"]    # test verisine göre validasyon loss
loss_test = np.array(loss_test)

axis = range(0,400)   # epoch --> 400
plt.plot(axis,loss_train, label="loss")
plt.plot(axis,loss_test, label="val_loss")
plt.legend()


# In[23]:


# Tahminler

tahminler = model.predict(x_test)
sayı = tahminler.shape[0]
tahminler = pd.Series(tahminler.reshape(sayı,))

resultFrame = pd.DataFrame(y_test,columns=["Gerçek Fiyat"])
resultFrame["Tahmin"] = tahminler
resultFrame


# In[24]:


plt.scatter(y_test,tahminler)
plt.plot(y_test,y_test,color="green")


# In[25]:


# Hata oranı değerlendirilmesi

from sklearn.metrics import mean_absolute_error

sapma = round(mean_absolute_error(resultFrame["Gerçek Fiyat"],resultFrame["Tahmin"]))
ort_fiyat = round(dataFrame["price"].mean())

# yüzde hesabı
# 24078 pound fiyatta 3170 pound sapıyorsa % yüzde kaç sapar?
yuzde_sapma = (100 * sapma) / ort_fiyat
dogruluk_oranı = round(100 - yuzde_sapma)

dogruluk_oranı  # --> %87


# In[26]:


deneme = pd.Series([2019,1000,145,22.1,4.0])   # gerçek fiyat --> 75729 
deneme = scaler.transform(deneme.values.reshape(-1,5))   # 5 özelliği matrise çevir
model.predict(deneme)


# In[ ]:




