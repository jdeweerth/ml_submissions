import os

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import neural_network
from sklearn.metrics import accuracy_score
from sklearn import mixture


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# EVIL CODE
import warnings
warnings.filterwarnings("ignore")

root_path = os.getcwd()


print("########## Importing Covertype Data... ##########")

# Load the covertype dataset
cov_data = fetch_covtype()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(cov_data.data,
                                                    cov_data.target,
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


print("########## KMeans... ##########")

c_count = []
train_scores = []
test_scores = []
max_c = 50

for c in range(1, max_c):
    print("Clusters: {}".format(c))

    clusterer = KMeans(n_clusters=c, random_state=42)
    # clusterer = clusterer.fit(X_train_scaled)
    cluster_labels_train = clusterer.fit_predict(X_train_scaled)
    cluster_labels_test = clusterer.fit_predict(X_test_scaled)

    cluster_labels_train = cluster_labels_train.reshape(-1, 1)
    cluster_labels_test = cluster_labels_test.reshape(-1, 1)

    nn = neural_network.MLPClassifier(max_iter=10000, hidden_layer_sizes=(10))
    nn.fit(cluster_labels_train, y_train_hot)

    y_train_pred = nn.predict(cluster_labels_train)
    y_test_pred = nn.predict(cluster_labels_test)

    train_score = accuracy_score(y_train_hot, y_train_pred)
    test_score = accuracy_score(y_test_hot, y_test_pred)
    train_scores.append(train_score)
    test_scores.append(test_score)

    print("KMeans train accuracy with {} clusters: {}".format(c, train_score))
    print("KMeans test accuracy with {} clusters: {}".format(c, test_score))


fig, ax = plt.subplots()

ax.plot(range(1, max_c), train_scores, label="train")
ax.plot(range(1, max_c), test_scores, label="test")

# ax[0].set_xticks(range(1, max_c))
ax.set_title('Accuracy vs. Number of KMeans Clusters')
ax.set_xlabel('Number of clusters used')
ax.set_ylabel('Accuracy score')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend()
ax.grid()


fig.savefig("{}/plots/p5_cluster_net/kmeans_performance.png".format(root_path))

print("########## EM... ##########")

gmm = mixture.GaussianMixture(n_components=c,
                              covariance_type='full')
cluster_labels = gmm.fit_predict(X_train_scaled)

print(cluster_labels.shape)
print(cluster_labels)

c_count = []
train_scores = []
test_scores = []
max_c = 50

for c in range(1, max_c):
    print("Clusters: {}".format(c))

    gmm = mixture.GaussianMixture(n_components=c,
                                  covariance_type='full')
    gmm = gmm.fit(X_train_scaled)
    cluster_labels_train = gmm.predict(X_train_scaled)
    cluster_labels_test = gmm.predict(X_test_scaled)

    cluster_labels_train = cluster_labels_train.reshape(-1, 1)
    cluster_labels_test = cluster_labels_test.reshape(-1, 1)

    nn = neural_network.MLPClassifier(max_iter=10000, hidden_layer_sizes=(10))
    nn.fit(cluster_labels_train, y_train_hot)
    y_train_pred = nn.predict(cluster_labels_train)
    y_test_pred = nn.predict(cluster_labels_test)

    train_score = accuracy_score(y_train_hot, y_train_pred)
    test_score = accuracy_score(y_test_hot, y_test_pred)
    train_scores.append(train_score)
    test_scores.append(test_score)

    print("EM train accuracy with {} clusters: {}".format(c, train_score))
    print("EM test accuracy with {} clusters: {}".format(c, test_score))


fig, ax = plt.subplots()

ax.plot(range(1, max_c), train_scores, label="train")
ax.plot(range(1, max_c), test_scores, label="test")

# ax[0].set_xticks(range(1, max_c))
ax.set_title('Accuracy vs. Number of EM Clusters')
ax.set_xlabel('Number of clusters used')
ax.set_ylabel('Accuracy score')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend()
ax.grid()

fig.savefig("{}/plots/p5_cluster_net/em_performance.png".format(root_path))
