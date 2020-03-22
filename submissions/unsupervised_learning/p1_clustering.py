from time import time
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

import itertools

from scipy import linalg
import matplotlib as mpl
from sklearn import mixture

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

# Silhouette method to determine ideal cluster count
print("########## Performing Covertype KMeans Silhouette Analysis... ##########")
range_n_clusters = range(2, 10)

X = X_train_scaled

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots()
    fig.set_size_inches(9, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X)

    if n_clusters == 7:
        print("Cluster label stats:")
        print(np.unique(cluster_labels, return_counts=True))
        print("True label stats:")
        print(np.unique(y_train, return_counts=True))
        print("Adjusted random score:")
        print(adjusted_rand_score(cluster_labels, y_train))

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Covertype Silhouette Plot")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis - KMeans clustering - Covertype dataset "
                  "- clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    fig.savefig("{}/plots/p1_clustering/kmeans/cover/silhouette_{}.png".format(root_path,
                str(n_clusters)))

comps = X_train_scaled.shape[1]
pca = PCA(n_components=comps).fit(X_train_scaled)

fig, ax1 = plt.subplots()

ax1.bar(range(1, comps + 1), pca.explained_variance_)

# ax1.set_xticks(range(1, comps + 1))
# ax1.set_ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
ax1.set_title('KMeans - Feature contributions via PCA - Covertype Dataset')
ax1.set_xlabel('Feature number')
ax1.set_ylabel('Eigenvalue')
ax1.grid()

fig.savefig("{}/plots/p1_clustering/kmeans/cover/feature_contribution.png".format(root_path))

print("########## Performing Covertype EM BIC Analysis... ##########")

# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)

X = X_train_scaled

lowest_bic = np.infty
bic = []
n_components_range = range(1, 14)
cv_types = ['full']
# cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)

        cluster_labels = gmm.fit_predict(X)
        if n_components == 7:
            print("Cluster label stats:")
            print(np.unique(cluster_labels, return_counts=True))
            print("True label stats:")
            print(np.unique(y_train, return_counts=True))
            print("Adjusted random score:")
            print(adjusted_rand_score(cluster_labels, y_train))

        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array([-b for b in bic])
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
fig, ax1 = plt.subplots()
fig.set_size_inches(9, 7)

for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
ax1.set_xticks(n_components_range)
ax1.set_ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
ax1.set_title('BIC score')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('BIC score')
ax1.legend([b[0] for b in bars], cv_types)

# ax1.set_xticks(())
# ax1.set_yticks(())
ax1.set_title('BIC Analysis - Expectation Maximization - Covertype dataset')

fig.savefig("{}/plots/p1_clustering/em/cover/bic.png".format(root_path))

print("########## Performing Shopping Silhouette Analysis... ##########")
X = shop_df

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots()
    fig.set_size_inches(9, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X)

    print("Cluster label stats:")
    print(np.unique(cluster_labels, return_counts=True))

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Shopping Silhouette Plot")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis - KMeans clustering - Shopping dataset "
                  "- clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    fig.savefig("{}/plots/p1_clustering/kmeans/shop/silhouette_{}.png".format(root_path,
                str(n_clusters)))

comps = shop_df.shape[1]
pca = PCA(n_components=comps).fit(shop_df)

fig, ax1 = plt.subplots()

ax1.bar(range(1, comps + 1), pca.explained_variance_)

# ax1.set_xticks(range(1, comps + 1))
# ax1.set_ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
ax1.set_title('KMeans - Feature contributions via PCA - Shopping Dataset')
ax1.set_xlabel('Feature number')
ax1.set_ylabel('Eigenvalue')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
ax1.grid()

fig.savefig("{}/plots/p1_clustering/kmeans/shop/feature_contribution.png".format(root_path))

print("########## Performing Shopping EM BIC Analysis... ##########")

# Number of samples per component
n_samples = 500

# Generate random sample, two components
np.random.seed(0)

X = shop_df

lowest_bic = np.infty
bic = []
n_components_range = range(1, 8)
cv_types = ['full']
# cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)

        cluster_labels = gmm.fit_predict(X)
        print("Cluster label stats:")
        print(np.unique(cluster_labels, return_counts=True))

        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array([-b for b in bic])
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
fig, ax1 = plt.subplots()
fig.set_size_inches(9, 7)

for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
ax1.set_xticks(n_components_range)
ax1.set_ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
ax1.set_title('BIC score')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('BIC score')
ax1.legend([b[0] for b in bars], cv_types)

# ax1.set_xticks(())
# ax1.set_yticks(())
ax1.set_title('BIC Analysis - Expectation Maximization - Shopping dataset')

fig.savefig("{}/plots/p1_clustering/em/shop/bic.png".format(root_path))
