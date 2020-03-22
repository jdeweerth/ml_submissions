from time import time
import os

import numpy as np

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn import neural_network
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import generate_plots as gp


# EVIL CODE
import warnings
warnings.filterwarnings("ignore")

root_path = os.getcwd()


do_pca = True
do_ica = True
do_rp = True
do_kpca = True


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


if do_pca:
    print("########## Performing Covertype PCA... ##########")
    c = 1
    interval = 1
    max_c = X_train_scaled.shape[1]
    # max_c = 3

    train_scores = []
    test_scores = []
    comp_count = []
    times = []
    done = False

    while c <= max_c:
        comp_count.append(c)
        print("Components: {}".format(c))

        start = time()

        pca = PCA(n_components=c).fit(X_train_scaled)
        X_train_reduced = pca.transform(X_train_scaled)
        X_test_reduced = pca.transform(X_test_scaled)

        nn = neural_network.MLPClassifier(max_iter=10000, hidden_layer_sizes=(10))
        nn.fit(X_train_reduced, y_train_hot)
        y_train_pred = nn.predict(X_train_reduced)
        y_test_pred = nn.predict(X_test_reduced)

        train_score = accuracy_score(y_train_hot, y_train_pred)
        test_score = accuracy_score(y_test_hot, y_test_pred)
        train_scores.append(train_score)
        test_scores.append(test_score)

        print("PCA train accuracy with {} components: {}".format(c, train_score))
        print("PCA test accuracy with {} components: {}".format(c, test_score))

        end = time()
        total_time = end - start
        times.append(total_time)
        print("Time elapsed: {}".format(total_time))

        c += interval
        interval += 1

        if done:
            break
        elif c >= max_c:
            c = max_c
            done = True

    fig, ax = plt.subplots(2, sharex=True)
    plt.subplots_adjust(hspace=0.1)

    ax[0].plot(comp_count, train_scores, label="train")
    ax[0].plot(comp_count, test_scores, label="test")

    # ax[0].set_xticks(range(1, max_c))
    ax[0].set_title('Time and Accuracy vs. Number of PCA Features')
    # ax[0].set_xlabel('# of features used')
    ax[0].set_ylabel('Accuracy score')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(comp_count, times)

    ax[1].set_xticks(range(1, max_c, 5))
    # ax[1].set_title('Training Time vs. Number of Features')
    ax[1].set_xlabel('# of features used')
    ax[1].set_ylabel('Training time (s)')
    ax[1].grid()

    fig.savefig("{}/plots/p4_reduce_net/pca_performance.png".format(root_path))


if do_ica:
    print("########## Performing Covertype ICA... ##########")
    c = 1
    interval = 1
    max_c = X_train_scaled.shape[1]
    # max_c = 3

    train_scores = []
    test_scores = []
    comp_count = []
    times = []
    done = False

    while c <= max_c:
        comp_count.append(c)
        print("Components: {}".format(c))

        start = time()

        ica = FastICA(n_components=c).fit(X_train_scaled)
        X_train_reduced = ica.transform(X_train_scaled)
        X_test_reduced = ica.transform(X_test_scaled)

        nn = neural_network.MLPClassifier(max_iter=10000, hidden_layer_sizes=(10))
        nn.fit(X_train_reduced, y_train_hot)
        y_train_pred = nn.predict(X_train_reduced)
        y_test_pred = nn.predict(X_test_reduced)

        train_score = accuracy_score(y_train_hot, y_train_pred)
        test_score = accuracy_score(y_test_hot, y_test_pred)
        train_scores.append(train_score)
        test_scores.append(test_score)

        print("ICA train accuracy with {} components: {}".format(c, train_score))
        print("ICA test accuracy with {} components: {}".format(c, test_score))

        end = time()
        total_time = end - start
        times.append(total_time)
        print("Time elapsed: {}".format(total_time))

        c += interval
        interval += 1

        if done:
            break
        elif c >= max_c:
            c = max_c
            done = True

    fig, ax = plt.subplots(2, sharex=True)
    plt.subplots_adjust(hspace=0.1)

    ax[0].plot(comp_count, train_scores, label="train")
    ax[0].plot(comp_count, test_scores, label="test")

    # ax[0].set_xticks(range(1, max_c))
    ax[0].set_title('Time and Accuracy vs. Number of ICA Features')
    # ax[0].set_xlabel('# of features used')
    ax[0].set_ylabel('Accuracy score')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(comp_count, times)

    ax[1].set_xticks(range(1, max_c, 5))
    # ax[1].set_title('Training Time vs. Number of Features')
    ax[1].set_xlabel('# of features used')
    ax[1].set_ylabel('Training time (s)')
    ax[1].grid()

    fig.savefig("{}/plots/p4_reduce_net/ica_performance.png".format(root_path))

if do_rp:
    print("########## Performing Covertype RP... ##########")
    c = 1
    interval = 1
    max_c = X_train_scaled.shape[1]
    # max_c = 3

    train_scores = []
    test_scores = []
    comp_count = []
    times = []
    done = False

    while c <= max_c:
        comp_count.append(c)
        print("Components: {}".format(c))

        start = time()

        grp = GRP(n_components=c).fit(X_train_scaled)
        X_train_reduced = grp.transform(X_train_scaled)
        X_test_reduced = grp.transform(X_test_scaled)

        nn = neural_network.MLPClassifier(max_iter=10000, hidden_layer_sizes=(10))
        nn.fit(X_train_reduced, y_train_hot)
        y_train_pred = nn.predict(X_train_reduced)
        y_test_pred = nn.predict(X_test_reduced)

        train_score = accuracy_score(y_train_hot, y_train_pred)
        test_score = accuracy_score(y_test_hot, y_test_pred)
        train_scores.append(train_score)
        test_scores.append(test_score)

        print("RP train accuracy with {} components: {}".format(c, train_score))
        print("RP test accuracy with {} components: {}".format(c, test_score))

        end = time()
        total_time = end - start
        times.append(total_time)
        print("Time elapsed: {}".format(total_time))

        c += interval
        interval += 1

        if done:
            break
        elif c >= max_c:
            c = max_c
            done = True

    fig, ax = plt.subplots(2, sharex=True)
    plt.subplots_adjust(hspace=0.1)

    ax[0].plot(comp_count, train_scores, label="train")
    ax[0].plot(comp_count, test_scores, label="test")

    # ax[0].set_xticks(range(1, max_c))
    ax[0].set_title('Time and Accuracy vs. Number of RP Features')
    # ax[0].set_xlabel('# of features used')
    ax[0].set_ylabel('Accuracy score')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(comp_count, times)

    ax[1].set_xticks(range(1, max_c, 5))
    # ax[1].set_title('Training Time vs. Number of Features')
    ax[1].set_xlabel('# of features used')
    ax[1].set_ylabel('Training time (s)')
    ax[1].grid()

    fig.savefig("{}/plots/p4_reduce_net/rp_performance.png".format(root_path))

if do_kpca:
    print("########## Performing Covertype KernelPCA... ##########")
    c = 1
    interval = 1
    max_c = X_train_scaled.shape[1]
    # max_c = 3

    train_scores = []
    test_scores = []
    comp_count = []
    times = []
    done = False

    while c <= max_c:
        comp_count.append(c)
        print("Components: {}".format(c))

        start = time()

        kpca = KernelPCA(n_components=c, kernel='rbf', fit_inverse_transform=True).fit(X_train_scaled)
        X_train_reduced = kpca.transform(X_train_scaled)
        X_test_reduced = kpca.transform(X_test_scaled)

        nn = neural_network.MLPClassifier(max_iter=10000, hidden_layer_sizes=(10))
        nn.fit(X_train_reduced, y_train_hot)
        y_train_pred = nn.predict(X_train_reduced)
        y_test_pred = nn.predict(X_test_reduced)

        train_score = accuracy_score(y_train_hot, y_train_pred)
        test_score = accuracy_score(y_test_hot, y_test_pred)
        train_scores.append(train_score)
        test_scores.append(test_score)

        print("KernelPCA train accuracy with {} components: {}".format(c, train_score))
        print("KernelPCA test accuracy with {} components: {}".format(c, test_score))

        end = time()
        total_time = end - start
        times.append(total_time)
        print("Time elapsed: {}".format(total_time))

        c += interval
        interval += 1

        if done:
            break
        elif c >= max_c:
            c = max_c
            done = True

    fig, ax = plt.subplots(2, sharex=True)
    plt.subplots_adjust(hspace=0.1)

    ax[0].plot(comp_count, train_scores, label="train")
    ax[0].plot(comp_count, test_scores, label="test")

    # ax[0].set_xticks(range(1, max_c))
    ax[0].set_title('Time and Accuracy vs. Number of KernelPCA Features')
    # ax[0].set_xlabel('# of features used')
    ax[0].set_ylabel('Accuracy score')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(comp_count, times)

    ax[1].set_xticks(range(1, max_c, 5))
    # ax[1].set_title('Training Time vs. Number of Features')
    ax[1].set_xlabel('# of features used')
    ax[1].set_ylabel('Training time (s)')
    ax[1].grid()

    fig.savefig("{}/plots/p4_reduce_net/kpca_performance.png".format(root_path))
