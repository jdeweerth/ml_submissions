import os
from time import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler  # , MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype


import mlrose_hiive as mlrose
from mlrose_hiive.runners import NNGSRunner
from mlrose_hiive import GeomDecay
import generate_plots as gp


root_path = os.getcwd()



# Load the Iris dataset
data = fetch_covtype()


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    train_size=5000,
                                                    test_size=2500,
                                                    shuffle=True,
                                                    random_state=3)

training_dist = (np.unique(y_train, return_counts=True))[1] / 5000
testing_dist = (np.unique(y_test, return_counts=True))[1] / 2500
print("training output distribution: {}".format(training_dist))
print("testing output distribution: {}".format(testing_dist))

# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

adult_x = X_train_scaled
adult_y = y_train_hot
adult_tst_x = X_test_scaled
adult_tst_y = y_test_hot

for nodes in [[10]]:

    print(nodes)
    start = time()

    nn_model1 = mlrose.NeuralNetwork(hidden_nodes=nodes, activation='relu',
                                     algorithm='gradient_descent',
                                     max_iters=3000,
                                     bias=True, is_classifier=True,
                                     learning_rate=0.0001, schedule=GeomDecay(1),
                                     early_stopping=False,
                                     clip_max=10, max_attempts=1000,
                                     random_state=124, curve=True)

    nn_model1.fit(adult_x, adult_y)

    # Predict labels for train set and assess accuracy
    adult_y_pred = nn_model1.predict(adult_x)

    adult_y_accuracy = accuracy_score(adult_y, adult_y_pred)
    adult_y_f1 = f1_score(adult_y, adult_y_pred, average='micro')

    print('Training accuracy: ', adult_y_accuracy)
    print('F1 score: ', adult_y_f1)

    # Predict labels for test set and assess accuracy
    adult_tst_y_pred = nn_model1.predict(adult_tst_x)

    adult_tst_y_accuracy = accuracy_score(adult_tst_y, adult_tst_y_pred)
    adult_tst_y_f1 = f1_score(adult_tst_y, adult_tst_y_pred, average='micro')

    print('Test accuracy: ', adult_tst_y_accuracy)
    print('F1 score: ', adult_tst_y_f1)

    fig_adult_lc = gp.plot_learning_curve(nn_model1,
                                          "Adult - learning curve",
                                          adult_x,
                                          adult_y, cv=3,
                                          train_sizes=np.linspace(0.03, 1.0, 12))
    fig_adult_lc.savefig(root_path + "/plots/nn/cover_lc_gd.png")

    print("Time elapsed: {}".format(time() - start))


for nodes in [[10]]:

    print(nodes)
    start = time()

    nn_model1 = mlrose.NeuralNetwork(hidden_nodes=nodes, activation='relu',
                                     algorithm='genetic_alg',
                                     max_iters=3000,
                                     bias=True, is_classifier=True,
                                     learning_rate=0.0001, schedule=GeomDecay(1),
                                     early_stopping=False,
                                     clip_max=10, max_attempts=1000,
                                     random_state=124, curve=True)

    nn_model1.fit(adult_x, adult_y)

    # Predict labels for train set and assess accuracy
    adult_y_pred = nn_model1.predict(adult_x)

    adult_y_accuracy = accuracy_score(adult_y, adult_y_pred)
    adult_y_f1 = f1_score(adult_y, adult_y_pred, average='micro')

    print('Training accuracy: ', adult_y_accuracy)
    print('F1 score: ', adult_y_f1)

    # Predict labels for test set and assess accuracy
    adult_tst_y_pred = nn_model1.predict(adult_tst_x)

    adult_tst_y_accuracy = accuracy_score(adult_tst_y, adult_tst_y_pred)
    adult_tst_y_f1 = f1_score(adult_tst_y, adult_tst_y_pred, average='micro')

    print('Test accuracy: ', adult_tst_y_accuracy)
    print('F1 score: ', adult_tst_y_f1)

    fig_adult_lc = gp.plot_learning_curve(nn_model1,
                                          "Adult - learning curve",
                                          adult_x,
                                          adult_y, cv=3,
                                          train_sizes=np.linspace(0.03, 1.0, 12))
    fig_adult_lc.savefig(root_path + "/plots/nn/cover_lc_ga.png")

    print("Time elapsed: {}".format(time() - start))

for nodes in [[10]]:

    print(nodes)
    start = time()

    nn_model1 = mlrose.NeuralNetwork(hidden_nodes=nodes, activation='relu',
                                     algorithm='simulated_annealing',
                                     max_iters=3000,
                                     bias=True, is_classifier=True,
                                     learning_rate=0.0001, schedule=GeomDecay(1),
                                     early_stopping=False,
                                     clip_max=10, max_attempts=1000,
                                     random_state=124, curve=True)

    nn_model1.fit(adult_x, adult_y)

    # Predict labels for train set and assess accuracy
    adult_y_pred = nn_model1.predict(adult_x)

    adult_y_accuracy = accuracy_score(adult_y, adult_y_pred)
    adult_y_f1 = f1_score(adult_y, adult_y_pred, average='micro')

    print('Training accuracy: ', adult_y_accuracy)
    print('F1 score: ', adult_y_f1)

    # Predict labels for test set and assess accuracy
    adult_tst_y_pred = nn_model1.predict(adult_tst_x)

    adult_tst_y_accuracy = accuracy_score(adult_tst_y, adult_tst_y_pred)
    adult_tst_y_f1 = f1_score(adult_tst_y, adult_tst_y_pred, average='micro')

    print('Test accuracy: ', adult_tst_y_accuracy)
    print('F1 score: ', adult_tst_y_f1)

    fig_adult_lc = gp.plot_learning_curve(nn_model1,
                                          "Adult - learning curve",
                                          adult_x,
                                          adult_y, cv=3,
                                          train_sizes=np.linspace(0.03, 1.0, 12))
    fig_adult_lc.savefig(root_path + "/plots/nn/cover_lc_sa.png")

    print("Time elapsed: {}".format(time() - start))

for nodes in [[10]]:

    print(nodes)
    start = time()

    nn_model1 = mlrose.NeuralNetwork(hidden_nodes=nodes, activation='relu',
                                     algorithm='random_hill_climb',
                                     max_iters=3000,
                                     bias=True, is_classifier=True,
                                     learning_rate=0.0001, schedule=GeomDecay(1),
                                     early_stopping=False,
                                     clip_max=10, max_attempts=1000,
                                     random_state=124, curve=True)

    nn_model1.fit(adult_x, adult_y)

    # Predict labels for train set and assess accuracy
    adult_y_pred = nn_model1.predict(adult_x)

    adult_y_accuracy = accuracy_score(adult_y, adult_y_pred)
    adult_y_f1 = f1_score(adult_y, adult_y_pred, average='micro')

    print('Training accuracy: ', adult_y_accuracy)
    print('F1 score: ', adult_y_f1)

    # Predict labels for test set and assess accuracy
    adult_tst_y_pred = nn_model1.predict(adult_tst_x)

    adult_tst_y_accuracy = accuracy_score(adult_tst_y, adult_tst_y_pred)
    adult_tst_y_f1 = f1_score(adult_tst_y, adult_tst_y_pred, average='micro')

    print('Test accuracy: ', adult_tst_y_accuracy)
    print('F1 score: ', adult_tst_y_f1)

    fig_adult_lc = gp.plot_learning_curve(nn_model1,
                                          "Adult - learning curve",
                                          adult_x,
                                          adult_y, cv=3,
                                          train_sizes=np.linspace(0.03, 1.0, 12))
    fig_adult_lc.savefig(root_path + "/plots/nn/cover_lc_rhc.png")

    print("Time elapsed: {}".format(time() - start))
