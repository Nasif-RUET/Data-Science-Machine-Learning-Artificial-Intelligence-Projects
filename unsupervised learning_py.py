import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('Wholesale_customers_data.csv')
####################### Wholesale_customers_datasets ##########################
#####  The aim of this problem is to segment the clients of a wholesale distributor
#####  based on their annual spending on diverse product categories, like milk,
#####  grocery, region, etc.
################################################################################
data.head()
######first normalize the data and bring all the variables to the same scale
from sklearn.preprocessing import normalize
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
data_scaled.head()


import scipy.cluster.hierarchy as shc
data_scaled=data_scaled.transpose()
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms using Ward")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.show()
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms using Single")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='single'))
plt.show()
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms using Complete")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='complete'))
plt.show()
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms using Average")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='average'))
plt.show()
from sklearn.cluster import AgglomerativeClustering
##data_scaled=data_scaled.transpose()
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data_scaled)
cluster2 = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')  
cluster2.fit_predict(data_scaled)

##plt.figure(figsize=(10, 7))  
##plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_)
##plt.show()


####### Calculating v_measure_score
##from sklearn.metrics.cluster import v_measure_score
####### v_measure_score(True_labels,cluster.labels_)
##v_measure_score(cluster.fit_predict(data_scaled),cluster.labels_)

###################### KMeans Clustering ##############################
### import KMeans
from sklearn.cluster import KMeans
### create kmeans object
kmeans = KMeans(n_clusters=4)
### fit kmeans object to data
kmeans.fit(data_scaled)
### print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
### save new clusters for chart
y_km = kmeans.fit_predict(data_scaled)
