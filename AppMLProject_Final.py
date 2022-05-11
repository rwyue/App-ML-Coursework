#!/usr/bin/env python
# coding: utf-8

# # Customer Personality Analysis

# ## Importing the dataset

# In[105]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
data = pd.read_csv("marketing_campaign.csv", sep="\t")
print("The amount of datapoints is", len(data))
data.head()


# In[106]:


# Summary of customer data
data_Transpose = data.describe()
data_Transpose.T


# In[107]:


print("The dimension of dataset is", data.shape)


# In[108]:


# Data observations
pd.DataFrame(data.nunique()).sort_values(0).rename( {0: 'Unique Values'}, axis=1)


# ### Data Engineering
# #### 1) Data Cleaning

# In[109]:


# Information relating to every feature
data.info()


# Categorical data for Education, Marital_Status
# Dt_Customer is not date actually because it's type is called "object"

# In[110]:


print(data["Dt_Customer"])


# In[111]:


# Convert "Dt_Customers" to data-type
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
dates_array = []
for date in data["Dt_Customer"]:
    date = date.date()
    dates_array.append(date) 


# In[112]:


data["Dt_Customer"]


# In[113]:


# Check on "Marital_Status" categorial features (8 different types)
data["Marital_Status"].value_counts()


# In[114]:


# Re-encode by grouping categorial features (Split into 2 groups, "Single", "Relationship")
# There are too many maritial status, which might affects the efficiency of classification algorithms.
data["Marital_Status"] = data["Marital_Status"].replace({"Married": "Relationship", "Together": 
"Relationship", "Single": "Single", "Divorced": "Single", "Widow": "Single", "Alone": "Single", "Absurd": "Single", "YOLO": "Single"})
data.Marital_Status.value_counts()


# In[115]:


# Check on "Education" categorial features (5 different types)
data["Education"].value_counts()


# In[116]:


# Re-encode by grouping categorial features 
data["Education"] = data["Education"].replace({"Graduation": "Undergraduate", 
"PhD": "Postgraduate", "Master": "Postgraduate", "2n Cycle": "Postgraduate", "Basic": "Undergraduate"})


# In[117]:


data["Education"].value_counts()


# In[118]:


# Check our data
data.head(10)


# In[119]:


# "Z_CostContact" and "Z_Revenue" have only one repating value.
data["Z_CostContact"].value_counts()


# In[120]:


data["Z_Revenue"].value_counts()


# In[121]:


data["Complain"].value_counts()


# #### 2. Features Engineering

# In[122]:


# Age of customer today
data["Age"] = 2022-data["Year_Birth"]

# Total children living in the household
data["Children"]=data["Kidhome"]+data["Teenhome"]

# Number of days when customer start shopping

# Total amount of spent by cusomter on various items
data["Total_Spend"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] 
+ data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]

# Total purchases from different channels
data["Total_Purchases"] = data["NumStorePurchases"] + data["NumWebPurchases"] + data["NumCatalogPurchases"]

# Total promotions accepted
data["Total_Accepted_Offers"] = data["AcceptedCmp1"] + data["AcceptedCmp2"] + data["AcceptedCmp3"] + data["AcceptedCmp4"] + data["AcceptedCmp5"] + data["Response"]


# In[123]:


# Number of relative days the cutomer is engaged with the company
earliest_customer = data['Dt_Customer'].max()
data['Engaged_Days'] = data['Dt_Customer'].apply(lambda x: (earliest_customer - x).days)
data['Engaged_Days']


# In[124]:


data["Total_Accepted_Offers"].value_counts()


# Issues found: Missing income value (2216/2240)

# In[125]:


# Check for NULL
data.isnull().sum()


# In[126]:


# Replace NULL with zero 
data['Income'].fillna(0, inplace = True)


# In[127]:


# Check for NULL again
data.isnull().sum()


# In[128]:


# Drop the redudant features
to_drop = ["Dt_Customer","Z_CostContact", "Z_Revenue", "Year_Birth", "MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts",
"MntGoldProds","Kidhome","Teenhome","AcceptedCmp1","AcceptedCmp2", "AcceptedCmp3","AcceptedCmp4","AcceptedCmp5", "Response", "ID"]
data = data.drop(to_drop, axis=1)


# In[129]:


# Drop further features
to_drop = ["Complain"]
data = data.drop(to_drop, axis=1)


# In[130]:


data.info()


# ### Data Preprocessing
# #### 1. Label Encoding of categorial features

# In[131]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


# In[132]:


cat_indicator = (data.dtypes == "object")
cat_cols = list(cat_indicator[cat_indicator].index)
# Our categorial features
cat_cols


# In[133]:


# Encoding categorial features
Encoder = LabelEncoder()
for feature in cat_cols:
  data[feature] = data[[feature]].apply(Encoder.fit_transform)


# In[134]:


# Now every feature is numerical
data.info()


# In[135]:


data.describe()


# #### 2. Dealing with outliers
# It seems that max age of 128 years and max income of 666666 are something indicating an existence of the outliers. 
# We can look at our data with a broader view.

# In[136]:


plt.figure(figsize = (10,10))

plt.subplot(1, 2, 1)
sns.boxplot(y = data.Income, palette="terrain_r")
plt.title("\"Income\" feature box-plot")

plt.subplot(1, 2, 2)
sns.boxplot(y = data.Age, palette="terrain_r")
plt.title("\"Age\" feature Box-plot")


# In[137]:


# Let's define a funtion to find interquantile range borders
def find_IQR(data, column):
    q_25, q_75 = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)
    IQR = q_75 - q_25
    whiskers_range = IQR * 1.5
    lower, upper = q_25 - whiskers_range, whiskers_range + q_75
    return lower, upper


# In[138]:


# Find LQ and UQ for Income
lower_income, upper_income = find_IQR(data, "Income")
print(lower_income, upper_income)


# In[139]:


# Find LQ and UQ for Age
lower_age, upper_age = find_IQR(data, "Age")
print(lower_age, upper_age)


# In[140]:


# Drop the outliers
data = data[(data["Age"] < upper_age)]
data = data[(data["Income"] < upper_income)]


# In[141]:


data.info()


# #### 3. Scaling the data

# In[142]:


# Scale the data
Scaler = StandardScaler()
Scaler.fit(data)
data_scaled = pd.DataFrame(Scaler.transform(data), columns= data.columns)
data_scaled.head()


# In[143]:


#correlation matrix
matrix = np.triu(data_scaled.corr())
plt.figure(figsize=(15,15)) 
sns.heatmap(data_scaled.corr(), vmin=-1, vmax=1, center= 0, cmap= 'coolwarm',annot=True,annot_kws={"size": 10.5},linewidths=.5, mask=matrix)


# ### Dimensionality reduction, PCA

# There are lots of feature but it's a bad decision to take a lot of features because of the correlations between them (see correlation matrix done earlier on) - information taken from correlated factors is redundant, so that we need nearly independent ones. Dimensionality reduction techniques can help us to deal with these problems.
# 
# Principal Component Analysis (PCA) - is one of such techniques. It will help us to minimize the information loss and increase the possibility of data to be interpreted better. Before start, the amount of principal vectors should be chosen - I will choose the amount of 3, because I want to have opportunity to visualize the data and save as much information as I can at the same time

# In[144]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(data_scaled)
principal_data = pd.DataFrame(pca.transform(data_scaled), columns=(["PC1","PC2", "PC3"]))
principal_data.head()
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# In[145]:


# Define vector comtaining coordinates (row = object)
x = principal_data['PC1']
y = principal_data['PC2']
z = principal_data['PC3']


# In[146]:


# Plot data
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x, y, z, c='green', marker='o')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title("Clusters shown in 3D applying PCA dimensionality reduction")
plt.show()


# ### Clustering
# We don't know the true distribution of target variable so that we can relate the task to "Unsupervised Clustering" 
# - our model will need to find the relations in data. The main problem is that we don't
# know the amount of clusters and to specify it three methods will be used: 
# "Elbow method", "Silhouette method", "DB Index"

# ### Elbow method for K-Means

# In[147]:


# Use elbow method to find optimal number of clusters for K-Means
from sklearn.cluster import KMeans
wcss = []   # save results in this empty list
for i in range(1,11):   # run this from 1 to 10 times
    kmeans = KMeans(n_clusters = i,
    init = 'k-means++', max_iter = 500, n_init = 10)
    kmeans.fit(principal_data)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,8))
plt.plot(range(1,11), wcss,'bx-')
plt.title("K-Means- The Elbow Method")
plt.xlabel("Number Of Clusters")
plt.ylabel("WCSS")
plt.show()


# In[148]:


# Using KneeLocator to find the optimal k value
get_ipython().system('pip install kneed')
from kneed import KneeLocator
wcss_knee = KneeLocator(
        x=range(1,11), 
        y=wcss, 
        S=0.1, curve="convex", direction="decreasing", online=True)

K_wcss= wcss_knee.elbow   
print("elbow at k =", f'{K_wcss:.0f} clusters')


# ### Silhouette method for K-Means

# In[149]:


get_ipython().system('pip install yellowbrick')


# In[150]:


from sklearn.cluster import KMeans 
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster.silhouette import SilhouetteVisualizer


# In[151]:


from yellowbrick.cluster import SilhouetteVisualizer
fig, ax = plt.subplots(2, 2, figsize=(15,8))
ax[0, 0].set_title('Average silhouette of 0.437 with 2 clusters')
ax[0, 1].set_title('Average silhouette of 0.382 with 3 clusters')
ax[1, 0].set_title('Average silhouette of 0.298 with 4 clusters')
ax[1, 1].set_title('Average silhouette of 0.285 with 5 clusters')
for i in [2, 3, 4, 5]:
    '''Create KMeans instance for different number of clusters'''
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=1000, random_state=42)
    q, mod = divmod(i, 2)
    '''Create SilhouetteVisualizer instance with KMeans instance Fit the visualizer'''
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(principal_data)
    km.fit_predict(principal_data)
    # Calculate Silhoutte Score
    score = silhouette_score(principal_data, km.labels_, metric='euclidean')
    print(f'Silhouetter Score: for {i} clusters is {score}')


# ### DB index for K-Means

# In[152]:


# 2 clusters
kmeans = KMeans(n_clusters=2, random_state=1).fit(principal_data)
# we store the cluster labels
labels = kmeans.labels_
print(davies_bouldin_score(principal_data,labels))


# In[153]:


labels


# In[154]:


principal_data["Clusters"] = labels
principal_data


# In[155]:


# distribution of clusters : are they fairly distributed ?
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
principal_data["Clusters"] = labels
plt.figure(figsize=(8,6))
count_plot = sns.countplot(x=principal_data["Clusters"], palette="flare")
count_plot.set_title("Distribution Cluster Observations", fontsize=14)
for p in count_plot.patches:
    count_plot.annotate(f'\n{p.get_height()}', (p.get_x()+0.4, p.get_height()), ha='center', va='top', color='black', size=16)
plt.xlabel("Cluster")
plt.ylabel("Number of Observations")
plt.show()


# In[159]:


data["Clusters"] = labels
data.groupby(['Clusters']).mean().T


# ### Agglomerative Clustering model - one of the Hierarchical Clustering models

# In[92]:


features = ['PC1','PC2','PC3']
x = principal_data[features]
x.head()


# In[93]:


principal_data.head()


# In[94]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
plt.figure(figsize =(8, 8)) 
plt.title('Visualising the data') 
dendro = sch.dendrogram(sch.linkage(principal_data,method = 'ward'))


# ### Silhouette Score - Agglomerative

# In[96]:


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
silhouette_scores = [] 
for n_cluster in range(2, 11):
    silhouette_scores.append( 
        silhouette_score(principal_data, AgglomerativeClustering(n_clusters = n_cluster).fit_predict(principal_data)))
#plot a graph to compare the results
plt.plot(range(2,11),silhouette_scores )
plt.title("")
plt.xlabel("Number Of Clusters", fontsize = 10)
plt.ylabel("Silhouette Score", fontsize = 10)
plt.title('Agglomerative Clustering - Number Of Clusters vs Silhouette Score') 
plt.show()
print(silhouette_scores)


# In[97]:


# 2 clusters
from sklearn.cluster import AgglomerativeClustering
# init the Agglomerative Clustering model 
AC = AgglomerativeClustering(n_clusters=2)
# fit our data and make prediction
y_pred = AC.fit_predict(principal_data)
principal_data["Clusters"] = y_pred
# also let's add the feature to original data
data["Clusters"] = y_pred
principal_data.head()


# ### DB Index - Agglomerative

# In[98]:


# 2 clusters
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

print(davies_bouldin_score(principal_data,y_pred))


# ## DBSCAN Clustering
# 

# In[99]:


import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pylab import rcParams
import csv
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
sns.set()
data_scaled.head()


# In[100]:


pca = PCA(n_components=10)
pca.fit(data_scaled)
variance = pca.explained_variance_ratio_ 
var=np.cumsum(np.round(variance, 3)*100)
plt.figure(figsize=(12,6))
plt.ylabel('% Variance Explained')
plt.xlabel('Number of Features')
plt.title('PCA Analysis')
plt.ylim(0,100,10)
plt.xlim(0,6)
plt.plot(var)


# In[101]:


#selection of eps value       
from sklearn.neighbors import NearestNeighbors
nbrs=NearestNeighbors().fit(principal_data)
distances, indices = nbrs.kneighbors(principal_data,20)
kDis = distances[:,10]
kDis.sort()
kDis = kDis[range(len(kDis)-1,0,-1)]
plt.plot(range(0,len(kDis)),kDis)
plt.xlabel('Distance')
plt.ylabel('eps')
plt.title('K-distance Graph To Estimate Optimal Value Of Epsilon')
plt.show()


# Choosing eps parameter using the K-distance graph:
# We used the below k distance graph from sklearn to find the optimal value of eps, which was found to be 1.5 (also from model iterations) and chose min_samples to be 4 according to the criteria that min_samples >= D+1 where D is the dimensions of the data (Ren, 2019) (here, we had 3 numerical variables).

# In[102]:


#DBSCAN Algorithm
from sklearn.cluster import DBSCAN
dbs_1= DBSCAN(eps=1.5, min_samples=4)
results = dbs_1.fit(principal_data).labels_


# In[103]:


#Visualize DBSCAN clustering 
df_DBSCAN=principal_data
df_DBSCAN['Cluster_id_DBSCAN']=results
print (df_DBSCAN['Cluster_id_DBSCAN'].value_counts())
sns.pairplot(df_DBSCAN,hue='Cluster_id_DBSCAN',palette='Dark2',diag_kind='kde')


# In[104]:


# Applying DBSCAN shows only 1 cluster is found (-1 being the cluster for noise)
dbscancluster = DBSCAN(eps = 1.5, min_samples = 4)
labels = dbscancluster.fit_predict(principal_data)
np.unique(labels)


# ## K-Prototypes

# In[160]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
data = pd.read_csv("marketing_campaign.csv", sep="\t")
print("The amount of datapoints is", len(data))
data.head()


# In[161]:


# Convert "Dt_Customers" to data-type
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
dates_array = []
for date in data["Dt_Customer"]:
    date = date.date()
    dates_array.append(date) 


# In[162]:


# Re-encode by grouping categorial features (Split into 2 groups, "Alone", "Not Alone")
data["Marital_Status"] = data["Marital_Status"].replace({"Married": "Relationship", "Together": 
"Relationship", "Single": "Single", "Divorced": "Single", "Widow": "Single", "Alone": "Single", "Absurd": "Single", "YOLO": "Single"})
data.Marital_Status.value_counts()


# In[163]:


# Re-encode by grouping categorial features (Split into 2 groups, "UG", "PG")
data["Education"] = data["Education"].replace({"Graduation": "Undergraduate", 
"PhD": "Postgraduate", "Master": "Postgraduate", "2n Cycle": "Postgraduate", "Basic": "Undergraduate"})
data.Education.value_counts()


# In[164]:


# Age of customer today
data["Age"] = 2022-data["Year_Birth"]

# Total children living in the household
data["Children"]=data["Kidhome"]+data["Teenhome"]

# Number of days when customer start shopping

# Total amount of spent by cusomter on various items
data["Total_Spend"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] 
+ data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]

# Total purchases from different channels
data["Total_Purchases"] = data["NumStorePurchases"] + data["NumWebPurchases"] + data["NumCatalogPurchases"]

# Total promotions accepted
data["Total_Accepted_Offers"] = data["AcceptedCmp1"] + data["AcceptedCmp2"] + data["AcceptedCmp3"] + data["AcceptedCmp4"] + data["AcceptedCmp5"] + data["Response"]


# In[165]:


# Number of relative days the cutomer is engaged with the company
earliest_customer = data['Dt_Customer'].max()
data['Engaged_Days'] = data['Dt_Customer'].apply(lambda x: (earliest_customer - x).days)
data['Engaged_Days']


# In[166]:


# Replace NULL with zero 
data['Income'].fillna(0, inplace = True)


# In[167]:


to_drop = ["Complain", "Dt_Customer","Z_CostContact", "Z_Revenue", "Year_Birth", "MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts",
"MntGoldProds","Kidhome","Teenhome","AcceptedCmp1","AcceptedCmp2", "AcceptedCmp3","AcceptedCmp4","AcceptedCmp5", "Response", "ID"]
data = data.drop(to_drop, axis=1)


# In[168]:


plt.figure(figsize = (10,10))

plt.subplot(1, 2, 1)
sns.boxplot(y = data.Income, palette="terrain_r")
plt.title("\"Income\" Box-plot")

plt.subplot(1, 2, 2)
sns.boxplot(y = data.Age, palette="terrain_r")
plt.title("\"Age\" Box-plot")


# In[169]:


# Let's define a funtion to find interquantile range borders
def find_IQR(data, column):
    q_25, q_75 = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)
    IQR = q_75 - q_25
    whiskers_range = IQR * 1.5
    lower, upper = q_25 - whiskers_range, whiskers_range + q_75
    return lower, upper


# In[170]:


# Find LQ and UQ for Income
lower_income, upper_income = find_IQR(data, "Income")
print(lower_income, upper_income)


# In[171]:


# Find LQ and UQ for Age
lower_age, upper_age = find_IQR(data, "Age")
print(lower_age, upper_age)


# In[172]:


# Drop the outliers
data = data[(data["Age"] < upper_age)]
data = data[(data["Income"] < upper_income)]


# In[173]:


data.info()


# In[174]:


get_ipython().system('pip install kmodes')
from kmodes.kprototypes import KPrototypes
dfMatrix = data.to_numpy()


# In[175]:


# getting categorical columns and their indices.
catColumnsPos = [data.columns.get_loc(col) for col in list(data.select_dtypes('object').columns)]
print('Categorical columns           : {}'.format(list(data.select_dtypes('object').columns)))
print('Categorical columns position  : {}'.format(catColumnsPos))


# In[176]:


cat_indicator = (data.dtypes == "object")
cat_cols = list(cat_indicator[cat_indicator].index)
# Our categorial features
cat_cols


# In[177]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
# Encoding categorial features
Encoder = LabelEncoder()
for feature in cat_cols:
  data[feature] = data[[feature]].apply(Encoder.fit_transform)


# In[178]:


data.info()


# In[179]:


# select columns to cluster
cluster_columns = ['Education', 'Marital_Status', 'Income', 'Recency', 'NumDealsPurchases','NumCatalogPurchases','NumStorePurchases'
                   ,'NumWebVisitsMonth','Age','Children','Total_Spend','Total_Purchases','Total_Accepted_Offers','Engaged_Days']
data = data[cluster_columns]


# In[180]:


from sklearn.preprocessing import StandardScaler
# define numerical and categorical columns
numerical_columns = ['Income', 'Recency', 'NumDealsPurchases','NumCatalogPurchases','NumStorePurchases'
                   ,'NumWebVisitsMonth','Age','Children','Total_Spend','Total_Purchases','Total_Accepted_Offers','Engaged_Days']
categorical_columns = ['Education', 'Marital_Status']
scaler = StandardScaler()
# create a copy of our data to be scaled
df_scale = data.copy()
# standard scale numerical features
for c in numerical_columns:
    df_scale[c] = scaler.fit_transform(data[[c]])


# In[181]:


from kmodes.kprototypes import KPrototypes
categorical_indexes = []
for c in categorical_columns:
    categorical_indexes.append(data.columns.get_loc(c))
categorical_indexes


# ### The Elbow Method 
# This method calculates the total cluster variance (cost) for varying numbers of clusters. As we increase the number of clusters we expect the total cluster variance to decrease. 

# In[182]:


import seaborn as sns
num_clusters = list(range(2, 11))
cost_values = []
# calculate cost values for each number of clusters (2 to 10)
for k in num_clusters:
    kproto = KPrototypes(n_clusters=k, init='Huang', random_state=42)
    kproto.fit_predict(df_scale, categorical= categorical_indexes)
    cost_values.append(kproto.cost_)
# plot cost against number of clusters
ax = sns.lineplot(x=num_clusters, y=cost_values, marker="o")
ax.set_title('K-Prototype The Elbow Method', fontsize=14)
ax.set_xlabel('No of clusters', fontsize=11)
ax.set_ylabel('Cost', fontsize=11)


# ### The Average Silhouette Method
# We plot the average silhouette values for varying levels of clusters and look for the number of clusters that result in the maximum average silhouette value.

# In[184]:


from sklearn.metrics import silhouette_score
silhouette_avg = []
# calculate average silhouette score for each number of cluster (2 to 10)
for k in num_clusters:
    kproto = KPrototypes(n_clusters=k, init='Huang', random_state=42)
    kproto.fit_predict(df_scale, categorical= categorical_indexes)
    cluster_labels = kproto.labels_
    silhouette_avg.append(silhouette_score(df_scale, cluster_labels))
# plot average silhouette score against number of clusters
ax = sns.lineplot(x=num_clusters, y=silhouette_avg, marker="o")
ax.set_title('Average Silhouette for K-Prototype', fontsize=13)
ax.set_xlabel('No of clusters', fontsize=10)
ax.set_ylabel('score', fontsize=10)
print(silhouette_avg)


# ### THE DB index

# In[185]:


from sklearn.metrics import davies_bouldin_score
label = kproto.fit_predict(df_scale, categorical = categorical_indexes)
davies_bouldin_score(df_scale, label)


# ### Elbow method for K-Prototypes

# In[17]:


# Clustering
#Elbow method to detect number of K
from kmodes.kprototypes import KPrototypes

cost = []
for cluster in range(1, 8):
    try:
        kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0)
        kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)
        cost.append(kprototype.cost_)
        print('Cluster initiation: {}'.format(cluster))
    except:
        break

plt.plot(cost)
plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
plt.title("K-Prototypes- The Elbow Method")
plt.show


# In[118]:


# cost (sum distance): confirm visual clue of elbow plot
# KneeLocator class will detect elbows if curve is convex; if concave, will detect knees
get_ipython().system('pip install kneed')
from kneed import KneeLocator
cost_knee_c3 = KneeLocator(
        x=range(1,8), 
        y=cost, 
        S=0.1, curve="convex", direction="decreasing", online=True)
K_cost_c3 = cost_knee_c3.elbow   
print("elbow at k =", f'{K_cost_c3:.0f} clusters')


# In[20]:


from sklearn.metrics import silhouette_score
silhouette_avg = []
num_clusters = list(range(2, 7))
# calculate average silhouette score for each number of cluster (2 to 6)
for k in num_clusters:
    kproto = KPrototypes(n_clusters=k, init='Huang', random_state=42)
    kproto.fit_predict(dfMatrix, categorical= catColumnsPos)
    cluster_labels = kproto.labels_
    silhouette_avg.append(silhouette_score(dfMatrix, cluster_labels))
# plot average silhouette score against number of clusters
ax = sns.lineplot(x=num_clusters, y=silhouette_avg, marker="o")
ax.set_title('Average Silhouette', fontsize=14)
ax.set_xlabel('No of clusters', fontsize=11)
ax.set_ylabel('score', fontsize=11)


# In[119]:


kprototype = KPrototypes(n_jobs = -1, n_clusters = 2, init = 'Huang', random_state = 0)
data['clusters']= kprototype.fit_predict(dfMatrix, categorical = catColumnsPos)

