Cluster For Linear Data 
#Importing Libraries

import numpy as np
import pandas as pd
from matplotlib import pyplot as py
from sklearn.cluster import KMeans

#Creating Data

dataset = pd.read_excel("heart_linear.xlsx")
print("Input Data and Shape")
print(dataset.shape)
dataset.head()

#Plotting Points

c1 = dataset['age'].values
c2 = dataset['ejection_fraction'].values
X = np.array(list(zip(c1, c2)))
py.scatter(c1, c2, c='black', s=7)

# Number of clusters

Kmeans = KMeans(n_clusters=3)

# Fitting the input data

Kmeans.fit(X)

# Getting the cluster labels

Labels = Kmeans.predict(X)

# Centroid values

Centroids = Kmeans.cluster_centers_
print("Centroid values")
print(Centroids) 

#Clustering Data

k= 3 
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = py.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if Labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], c=colors[i], s=7)
        
ax.scatter(Centroids[:, 0], Centroids[:, 1], marker='*', s=200, c='#050505')
