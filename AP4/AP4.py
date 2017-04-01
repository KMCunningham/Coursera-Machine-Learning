
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from collections import Counter


# In[26]:

#read in data
names = ["Area", "Perimeter", "Compactness", "Length of Kernel", "Width of Kernel",
       "Asymmetry Coefficient", "Length of Kernel Groove", "Variety of Wheat"]
raw_data = pd.read_table("https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt", delimiter = "\t",
                         na_values = ["?"], names = names)
data = raw_data[names[0:len(names)-1]]
variety = raw_data["Variety of Wheat"]

print(data.shape)

#data.head(10)

data = data.dropna()
print(data.shape)
#data.head(10)

k = 3

kmeans = KMeans(n_clusters = k)
clusters = kmeans.fit_predict(data)
cvar = [[] for _ in range(k)]
total = 0

for i in range(0, k):
    c = np.where(clusters==i)[0].tolist()
    cv = variety[c]
    count = Counter(cv).most_common(1)
    cvar[i] = count[0][0]
    total = total + count[0][1]

print("Accuracy:  ")
print(float(total)/len(variety))

centers = kmeans.cluster_centers_.tolist()

print("\nCentroids: ")
for i in range(0,k):
    centers[i].append(cvar[i])
    print zip(names, centers[i])


# In[ ]:



