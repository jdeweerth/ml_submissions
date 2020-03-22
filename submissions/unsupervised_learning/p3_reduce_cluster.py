from time import time
import os

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import mixture

import matplotlib.pyplot as plt

# EVIL CODE
import warnings
warnings.filterwarnings("ignore")

root_path = os.getcwd()


print("########## Importing Covertype Data... ##########")

# Load the covertype dataset
cov_data = fetch_covtype()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(cov_data.data, cov_data.target,
                                                    train_size=10000,
                                                    test_size=5000,
                                                    shuffle=True,
                                                    random_state=3)

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()
adult_tst_y = y_test_hot


print("########## Importing Shopping Data... ##########")

# shopping data
shop_df = pd.read_csv("data/online_shoppers_intention.csv")
shop_df.dropna(inplace=True)

shop_df['Month'].replace({'Feb': 2, 'Mar': 3, 'May': 5, 'June': 6, 'Jul': 7,
                          'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12},
                         inplace=True)
shop_df['VisitorType'].replace({'Returning_Visitor': 0, 'New_Visitor': 1, 'Other': 2},
                               inplace=True)
shop_df['Weekend'].replace({False: 0, True: 1}, inplace=True)
shop_df['Revenue'].replace({False: 0, True: 1}, inplace=True)



def dstr_stats(clust_before, clust_after):
    dstr_before = np.unique(cluster_labels_before, return_counts=True)
    dstr_after = np.unique(cluster_labels_after, return_counts=True)
    print("cluster dstr before dim reduction: {}".format(dstr_before))
    print("cluster dstr after dim reduction:  {}".format(dstr_after))
    score = adjusted_rand_score(cluster_labels_before, cluster_labels_after)
    print("Adjusted random score: {}".format(score))

    return score


combo = []
scores = []

print("########## 1: KMeans - PCA - Covertype Data... ##########")

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_before = clusterer.fit_predict(X_train_scaled)

pca = PCA(n_components=5).fit(X_train_scaled)
# data_recon = np.dot(pca.transform(X_train_scaled), np.linalg.pinv(pca.components_.T))
data_recon = pca.fit_transform(X_train_scaled)

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_after = clusterer.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "KMeans - PCA - Cover"
combo.append(name)
scores.append(score)


print("########## 2: KMeans - ICA - Covertype Data... ##########")

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_before = clusterer.fit_predict(X_train_scaled)

ica = FastICA(n_components=16)
data_fitted = pd.DataFrame(ica.fit_transform(X_train_scaled))
# data_recon = pd.DataFrame(ica.inverse_transform(data_fitted))
data_recon = ica.fit_transform(X_train_scaled)

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_after = clusterer.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "KMeans - ICA - Cover"
combo.append(name)
scores.append(score)


print("########## 3: KMeans - RP - Covertype Data... ##########")

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_before = clusterer.fit_predict(X_train_scaled)

grp = GRP(n_components=16).fit(X_train_scaled)
# data_recon = np.dot(grp.transform(X_train_scaled), np.linalg.pinv(grp.components_.T))
data_recon = grp.fit_transform(X_train_scaled)

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_after = clusterer.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "KMeans - RP - Cover"
combo.append(name)
scores.append(score)


print("########## 4: KMeans - KernelPCA - Covertype Data... ##########")

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_before = clusterer.fit_predict(X_train_scaled)

pca = KernelPCA(n_components=5, kernel='rbf', fit_inverse_transform=True).fit(X_train_scaled)
# data_recon = np.linalg.pinv(pca.X_transformed_fit_.T)
data_recon = pca.fit_transform(X_train_scaled)

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_after = clusterer.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "KMeans - KPCA - Cover"
combo.append(name)
scores.append(score)


print("########## 5: KMeans - PCA - Shopping Data... ##########")

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_before = clusterer.fit_predict(shop_df)

pca = PCA(n_components=2).fit(shop_df)
# data_recon = np.dot(pca.transform(shop_df), np.linalg.pinv(pca.components_.T))
data_recon = pca.fit_transform(shop_df)

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_after = clusterer.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "KMeans - PCA - Shop"
combo.append(name)
scores.append(score)


print("########## 6: KMeans - ICA - Shopping Data... ##########")

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_before = clusterer.fit_predict(shop_df)

ica = FastICA(n_components=8)
data_fitted = pd.DataFrame(ica.fit_transform(shop_df))
# data_recon = pd.DataFrame(ica.inverse_transform(data_fitted))
data_recon = ica.fit_transform(shop_df)

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_after = clusterer.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "KMeans - ICA - Shop"
combo.append(name)
scores.append(score)


print("########## 7: KMeans - RP - Shopping Data... ##########")

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_before = clusterer.fit_predict(shop_df)

grp = GRP(n_components=16).fit(shop_df)
# data_recon = np.dot(grp.transform(shop_df), np.linalg.pinv(grp.components_.T))
data_recon = grp.fit_transform(shop_df)

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_after = clusterer.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "KMeans - RP - Shop"
combo.append(name)
scores.append(score)


print("########## 8: KMeans - KernelPCA - Shopping Data... ##########")

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_before = clusterer.fit_predict(shop_df)

pca = KernelPCA(n_components=2, kernel='rbf', fit_inverse_transform=True).fit(shop_df)
# data_recon = np.linalg.pinv(pca.X_transformed_fit_.T)
data_recon = pca.fit_transform(shop_df)

clusterer = KMeans(n_clusters=3, random_state=42)
cluster_labels_after = clusterer.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "KMeans - KPCA - Shop"
combo.append(name)
scores.append(score)


print("########## 9: EM - PCA - Covertype Data... ##########")

gmm = mixture.GaussianMixture(n_components=8,
                              covariance_type='full')
cluster_labels_before = gmm.fit_predict(X_train_scaled)

pca = PCA(n_components=5).fit(X_train_scaled)
# data_recon = np.dot(pca.transform(X_train_scaled), np.linalg.pinv(pca.components_.T))
data_recon = pca.fit_transform(X_train_scaled)

gmm = mixture.GaussianMixture(n_components=8,
                              covariance_type='full')
cluster_labels_after = gmm.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "EM - PCA - Cover"
combo.append(name)
scores.append(score)


print("########## 10: EM - ICA - Covertype Data... ##########")

gmm = mixture.GaussianMixture(n_components=8,
                              covariance_type='full')
cluster_labels_before = gmm.fit_predict(X_train_scaled)

ica = FastICA(n_components=16)
data_fitted = pd.DataFrame(ica.fit_transform(X_train_scaled))
# data_recon = pd.DataFrame(ica.inverse_transform(data_fitted))
data_recon = ica.fit_transform(X_train_scaled)

gmm = mixture.GaussianMixture(n_components=8,
                              covariance_type='full')
cluster_labels_after = gmm.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "EM - ICA - Cover"
combo.append(name)
scores.append(score)


print("########## 11: EM - RP - Covertype Data... ##########")

gmm = mixture.GaussianMixture(n_components=8,
                              covariance_type='full')
cluster_labels_before = gmm.fit_predict(X_train_scaled)

grp = GRP(n_components=16).fit(X_train_scaled)
# data_recon = np.dot(grp.transform(X_train_scaled), np.linalg.pinv(grp.components_.T))
data_recon = grp.fit_transform(X_train_scaled)

gmm = mixture.GaussianMixture(n_components=8,
                              covariance_type='full')
cluster_labels_after = gmm.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "EM - RP - Cover"
combo.append(name)
scores.append(score)


print("########## 12: EM - KernelPCA - Covertype Data... ##########")

gmm = mixture.GaussianMixture(n_components=8,
                              covariance_type='full')
cluster_labels_before = gmm.fit_predict(X_train_scaled)

pca = KernelPCA(n_components=5, kernel='rbf', fit_inverse_transform=True).fit(X_train_scaled)
# data_recon = np.linalg.pinv(pca.X_transformed_fit_.T)
data_recon = pca.fit_transform(X_train_scaled)

gmm = mixture.GaussianMixture(n_components=8,
                              covariance_type='full')
cluster_labels_after = gmm.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "EM - KPCA - Cover"
combo.append(name)
scores.append(score)


print("########## 13: EM - PCA - Shopping Data... ##########")

gmm = mixture.GaussianMixture(n_components=3,
                              covariance_type='full')
cluster_labels_before = gmm.fit_predict(shop_df)

pca = PCA(n_components=2).fit(shop_df)
# data_recon = np.dot(pca.transform(shop_df), np.linalg.pinv(pca.components_.T))
data_recon = pca.fit_transform(shop_df)

gmm = mixture.GaussianMixture(n_components=3,
                              covariance_type='full')
cluster_labels_after = gmm.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "EM - PCA - Shop"
combo.append(name)
scores.append(score)


print("########## 14: EM - ICA - Shopping Data... ##########")

gmm = mixture.GaussianMixture(n_components=3,
                              covariance_type='full')
cluster_labels_before = gmm.fit_predict(shop_df)

ica = FastICA(n_components=8)
data_fitted = pd.DataFrame(ica.fit_transform(shop_df))
# data_recon = pd.DataFrame(ica.inverse_transform(data_fitted))
data_recon = ica.fit_transform(shop_df)

gmm = mixture.GaussianMixture(n_components=3,
                              covariance_type='full')
cluster_labels_after = gmm.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "EM - ICA - Shop"
combo.append(name)
scores.append(score)


print("########## 15: EM - RP - Shopping Data... ##########")

gmm = mixture.GaussianMixture(n_components=3,
                              covariance_type='full')
cluster_labels_before = gmm.fit_predict(shop_df)

grp = GRP(n_components=16).fit(shop_df)
# data_recon = np.dot(grp.transform(shop_df), np.linalg.pinv(grp.components_.T))
data_recon = grp.fit_transform(shop_df)

gmm = mixture.GaussianMixture(n_components=3,
                              covariance_type='full')
cluster_labels_after = gmm.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "EM - RP - Shop"
combo.append(name)
scores.append(score)


print("########## 16: EM - KernelPCA - Shopping Data... ##########")

gmm = mixture.GaussianMixture(n_components=3,
                              covariance_type='full')
cluster_labels_before = gmm.fit_predict(shop_df)

pca = KernelPCA(n_components=2, kernel='rbf', fit_inverse_transform=True).fit(shop_df)
# data_recon = np.linalg.pinv(pca.X_transformed_fit_.T)
data_recon = pca.fit_transform(shop_df)

gmm = mixture.GaussianMixture(n_components=3,
                              covariance_type='full')
cluster_labels_after = gmm.fit_predict(data_recon)

score = dstr_stats(cluster_labels_before, cluster_labels_after)
name = "EM - KPCA - Shop"
combo.append(name)
scores.append(score)

fig, ax1 = plt.subplots()

ax1.bar(combo, scores)

for tick in ax1.get_xticklabels():
    tick.set_rotation(90)

ax1.set_ylabel('Adjusted Random Score')
ax1.set_title('Reduced Data vs. Original Data - Cluster Similarity')
plt.tight_layout()
fig.savefig("{}/plots/p3_reduce_cluster/cluster_sim_scores.png".format(root_path))