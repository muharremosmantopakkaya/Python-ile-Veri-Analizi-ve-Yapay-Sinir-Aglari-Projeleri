#!/usr/bin/env python
# coding: utf-8

# #  Market Mutluluk Anketi

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Stilleri Seaborn olarak ayarlayın
sns.set()
# Sklearn ile k-means kümeleme yapabilmemiz için KMeans modülünü içe aktarın
from sklearn.cluster import KMeans


# In[4]:


# Veriyi yükleyelim
data = pd.read_csv ('../../Downloads/market_tatmin_sadakat.csv')


# In[5]:


# Verinin içinde ne var kontrol edelim
data


# In[6]:


# İki değişkenin saçılım grafiğini oluşturalım
plt.scatter(data['tatmin'],data['sadakat'])
# Eksenleri isimlendirelim 
plt.xlabel('Tatmin')
plt.ylabel('Sadakat')


# In[7]:


# Veri değişkeninin bir kopyasını oluşturarak her iki özelliği de seçiyoruz
x = data.copy()


# In[8]:


# Bir nesne oluşturalım (buna kmeans diyelim)
# Parantez içindeki sayı K, ya da hedeflediğimiz küme sayısıdır.
kmeans = KMeans(2)
# Datayı fit edelim (uyduralım)
kmeans.fit(x)


# In[9]:


# Input verilerinin bir kopyasını oluşturalım
clusters = x.copy()
# Öngörülen kümeleri not edelim
clusters['kume_tahmin']=kmeans.fit_predict(x)


# In[10]:


# c (color-renk) bir değişkenle kodlanabilen bir argümandır
# Bu durumda değişken, plt.scatter'a iki renk olduğunu gösteren 0,1 değerlerine sahiptir (0,1)
# Küme 0'daki tüm noktalar aynı renk, küme 1'deki tüm noktalar - başka bir renk vb.
# cmap renk haritasıdır. Gökkuşağı seçelim, ama başkalarını seçmek için: https://matplotlib.org/users/colormaps.html
plt.scatter(clusters['tatmin'],clusters['sadakat'],c=clusters['kume_tahmin'],cmap='rainbow')
plt.xlabel('Tatmin')
plt.ylabel('Sadakat')


# In[11]:


# Bunu kolayca yapabilen bir kütüphaneyi içe aktaralım
from sklearn import preprocessing
# Girdileri ölçeklendirelim (scale)
# preprocessing.scale her değişkeni (x'deki kolon) kendisine göre ölçeklendirir
# Yeni sonuç bir dizidir
x_scaled = preprocessing.scale(x)
x_scaled


# In[12]:


# Boş bir liste oluşturalım
kikt =[]

# Olası tüm küme çözümlerini bir döngü ile oluşturalım
# 1 ila 9 kümeden çözüm elde etmeyi seçtik; dilerseniz bunu değiştirebilirsiniz
for i in range(1,10):
    # i kümeleriyle küme çözümü
    kmeans = KMeans(i)
    # STANDARTLAŞTIRILMIŞ veriyi fit ediyoruz
    kmeans.fit(x_scaled)
    # Yineleme değerlerini kikt'ye ekliyoruz
    kikt.append(kmeans.inertia_)
    
# Sonucu kontrol ediyoruz
kikt


# In[13]:


# Küme sayısı vs kikt grafiğini çiziyoruz
plt.plot(range(1,10),kikt)
# Eksenleri isimlendirelim
plt.xlabel('Küme Sayısı')
plt.ylabel('Küme-içi Kareler Toplamı')


# In[14]:


# Küme sayısı için 2,3,4 ve 5'i deneyebilirsiniz.
kmeans_new = KMeans(4)
# Veriyi fit et
kmeans_new.fit(x_scaled)
# Öngörülen kümelerle yeni bir veri çerçevesi oluşturalım
clusters_new = x.copy()
clusters_new['kume_tahmin'] = kmeans_new.fit_predict(x_scaled)


# In[15]:


# Her şeyin doğru görünüp görünmediğini kontrol edelim
clusters_new


# In[16]:


# Çizim
plt.scatter(clusters_new['tatmin'],clusters_new['sadakat'],c=clusters_new['kume_tahmin'],cmap='rainbow')
plt.xlabel('Tatmin')
plt.ylabel('Sadakat')

