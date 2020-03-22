from time import time
import os

from copy import deepcopy

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn.metrics import mean_squared_error
from math import sqrt

from scipy.stats import kurtosis

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
X_train, X_test, y_train, y_test = train_test_split(cov_data.data, cov_data.target,
                                                    train_size=10000,
                                                    test_size=5000,
                                                    shuffle=True,
                                                    random_state=3)

# training_dist = (np.unique(y_train, return_counts=True))[1] / 10000
# testing_dist = (np.unique(y_test, return_counts=True))[1] / 5000
# print("training output distribution: {}".format(training_dist))
# print("testing output distribution: {}".format(testing_dist))

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
shop_copy_df = deepcopy(shop_df)
# shop_df = shop_df.drop(["OperatingSystems", "Browser", "PageValues", "Revenue",
#                         "Administrative_Duration", "Informational_Duration",
#                         "ProductRelated_Duration", "BounceRates", "ExitRates"],
#                        axis=1)
shop_df = shop_df.drop(["Revenue"], axis=1)

shop_df['Month'].replace({'Feb': 2, 'Mar': 3, 'May': 5, 'June': 6, 'Jul': 7,
                          'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12},
                         inplace=True)
shop_df['VisitorType'].replace({'Returning_Visitor': 0, 'New_Visitor': 1, 'Other': 2},
                               inplace=True)
shop_df['Weekend'].replace({False: 0, True: 1}, inplace=True)
# shop_df['Revenue'].replace({False: 0, True: 1}, inplace=True)


print("########## Performing Covertype PCA... ##########")
comps = 15
pca = PCA(n_components=comps).fit(X_train_scaled)
var_ratio = pca.explained_variance_ratio_
total_var_ratio = np.cumsum(var_ratio)
print(var_ratio)
print(total_var_ratio)

fig, ax1 = plt.subplots()

ax1.plot(range(1, comps + 1), var_ratio, label="Variance")
ax1.plot(range(1, comps + 1), total_var_ratio, label="Cumulative Variance")

ax1.set_xticks(range(1, comps + 1))
# ax1.set_ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
ax1.set_title('PCA - Variance explained by new components - Covertype Dataset')
ax1.set_xlabel('New feature number')
ax1.set_ylabel('Proportion of variance')
ax1.legend()
ax1.grid()

fig.savefig("{}/plots/p2_dim_reduction/pca/cover_scree.png".format(root_path))



pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_train_scaled)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1',
                                    'principal component 2'])
targetDf = pd.DataFrame(data=y_train, columns=['target'])
finalDf = pd.concat([principalDf, targetDf], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('PCA - 2 Principle Components - Covertype Dataset')
targets = range(7)
colors = ['y', 'g', 'b', 'c', 'm', 'r', 'orange']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c=color,
               s=5)
plt.legend(targets, markerscale=5)
ax.grid()
fig.savefig("{}/plots/p2_dim_reduction/pca/cover_pc_viz.png".format(root_path))

run_count = 1
runs = []
for r in range(run_count):
    c_error = []
    for c in range(1, X_train_scaled.shape[1] + 1):
        pca = PCA(n_components=c).fit(X_train_scaled)
        data_recon = np.dot(pca.transform(X_train_scaled), np.linalg.pinv(pca.components_.T))
        # data_error = np.sqrt((pd.DataFrame(data_recon) - pd.DataFrame(X_train_scaled)).apply(np.square).mean())
        rms = sqrt(mean_squared_error(data_recon, X_train_scaled))
        c_error.append(rms)
    runs.append(c_error)

runs_mean = np.mean(runs, axis=0)
runs_std = np.std(runs, axis=0)

fig, ax1 = plt.subplots()
ax1.plot(range(1, X_train_scaled.shape[1] + 1), runs_mean, label='mean')
# ax1.plot(range(1, X_train_scaled.shape[1] + 1), runs_mean + runs_std, label='mean + std')
# ax1.plot(range(1, X_train_scaled.shape[1] + 1), runs_mean - runs_std, label='mean - std')
ax1.grid()
# ax1.legend()

ax1.set_xlabel('Number of components used')
ax1.set_ylabel('RMSE')
ax1.set_title('PCA - Reconstruction Error - Covertype Dataset')

fig.savefig("{}/plots/p2_dim_reduction/pca/cover_error.png".format(root_path))



print("########## Performing Shopping PCA... ##########")
comps = 9
pca = PCA(n_components=comps).fit(shop_df)
var_ratio = pca.explained_variance_ratio_
total_var_ratio = np.cumsum(var_ratio)
print(var_ratio)
print(total_var_ratio)

fig, ax1 = plt.subplots()

ax1.plot(range(1, comps + 1), var_ratio, label="Variance")
ax1.plot(range(1, comps + 1), total_var_ratio, label="Cumulative Variance")

ax1.set_xticks(range(1, comps + 1))
# ax1.set_ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
ax1.set_title('PCA - Variance explained by new components - Shopping Dataset')
ax1.set_xlabel('New feature number')
ax1.set_ylabel('Proportion of variance')
ax1.legend()

fig.savefig("{}/plots/p2_dim_reduction/pca/shopping_scree.png".format(root_path))


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(shop_df)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1',
                                    'principal component 2'])
finalDf = pd.concat([principalDf, shop_copy_df['Revenue']], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('PCA - 2 Principle Components - Shopping Dataset')

# ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'], s=5)

targets = [True, False]
colors = ['b', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Revenue'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c=color,
               s=5)

# plt.legend(targets, markerscale=5)
ax.grid()
fig.savefig("{}/plots/p2_dim_reduction/pca/shopping_pc_viz.png".format(root_path))

run_count = 1
runs = []
for r in range(run_count):
    c_error = []
    for c in range(1, shop_df.shape[1] + 1):
        pca = PCA(n_components=c).fit(shop_df)
        data_recon = np.dot(pca.transform(shop_df), np.linalg.pinv(pca.components_.T))
        # data_error = np.sqrt((pd.DataFrame(data_recon) - pd.DataFrame(shop_df)).apply(np.square).mean())
        rms = sqrt(mean_squared_error(data_recon, shop_df))
        c_error.append(rms)
    runs.append(c_error)

runs_mean = np.mean(runs, axis=0)
runs_std = np.std(runs, axis=0)

fig, ax1 = plt.subplots()
ax1.plot(range(1, shop_df.shape[1] + 1), runs_mean, label='mean')
# ax1.plot(range(1, shop_df.shape[1] + 1), runs_mean + runs_std, label='mean + std')
# ax1.plot(range(1, shop_df.shape[1] + 1), runs_mean - runs_std, label='mean - std')
ax1.grid()
# ax1.legend()

ax1.set_xlabel('Number of components used')
ax1.set_ylabel('RMSE')
ax1.set_title('PCA - Reconstruction Error - Shopping Dataset')

fig.savefig("{}/plots/p2_dim_reduction/pca/shop_error.png".format(root_path))



print("########## Performing Covertype ICA... ##########")
X = X_train_scaled
run_count = 1
runs = []
for r in range(run_count):
    k_list = []
    for c in range(1, X.shape[1]):
        ica = FastICA(n_components=c).fit(shop_df)
        avg_kurtosis = np.average(abs(kurtosis(ica.components_, fisher=False)))
        k_list.append(avg_kurtosis)
        # print("Kurtosis with {} components: {}".format(c, avg_kurtosis))
    runs.append(k_list)

fig, ax1 = plt.subplots()
for run in runs:
    ax1.plot(range(1, X.shape[1]), run)
ax1.grid()

ax1.set_xlabel('Number of components')
ax1.set_ylabel('Kurtosis')
ax1.set_title('ICA - Kurtosis - Covertype Dataset')

fig.savefig("{}/plots/p2_dim_reduction/ica/cover_kurtosis.png".format(root_path))


error_list = []
for c in range(1, 51):
    ica = FastICA(n_components=c).fit(shop_df)
    data_fitted = pd.DataFrame(ica.fit_transform(X_train_scaled))
    data_recon = pd.DataFrame(ica.inverse_transform(data_fitted))
    # data_error = np.sqrt((data_recon - X_train_scaled).apply(np.square).mean())
    rms = sqrt(mean_squared_error(data_recon, X_train_scaled))
    error_list.append(rms)

fig, ax1 = plt.subplots()
ax1.bar(range(1, 51), error_list)

ax1.set_xlabel('Feature Count')
ax1.set_ylabel('RMSE')
ax1.set_title('ICA - Reconstruction Error - Covertype Dataset')

fig.savefig("{}/plots/p2_dim_reduction/ica/cover_error.png".format(root_path))



print("########## Performing Shopping ICA... ##########")

k_list = []
X = shop_df
for c in range(1, X.shape[1]):
    ica = FastICA(n_components=c).fit(shop_df)
    avg_kurtosis = np.average(abs(kurtosis(ica.components_, fisher=False)))
    k_list.append(avg_kurtosis)
    # print("Kurtosis with {} components: {}".format(c, avg_kurtosis))

fig, ax1 = plt.subplots()
ax1.plot(range(1, X.shape[1]), k_list)
ax1.grid()

ax1.set_xlabel('Number of components')
ax1.set_ylabel('Kurtosis')
ax1.set_title('ICA - Kurtosis - Shopping Dataset')

fig.savefig("{}/plots/p2_dim_reduction/ica/shop_kurtosis.png".format(root_path))

ica = FastICA(n_components=10)
data_fitted = pd.DataFrame(ica.fit_transform(shop_df.to_numpy()))
data_recon = pd.DataFrame(ica.inverse_transform(data_fitted))
data_error = np.sqrt((data_recon - shop_df.to_numpy()).apply(np.square).mean())
# rms = sqrt(mean_squared_error(data_recon, shop_df))

fig, ax1 = plt.subplots()
ax1.bar(range(1, shop_df.shape[1] + 1), data_error)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

ax1.set_xlabel('Feature')
ax1.set_ylabel('RMSE')
ax1.set_title('ICA - Reconstruction Error - Shopping Dataset')

fig.savefig("{}/plots/p2_dim_reduction/ica/shop_error.png".format(root_path))



print("########## Performing Covertype RP... ##########")

run_count = 25
runs = []
for r in range(run_count):
    c_error = []
    for c in range(1, X_train_scaled.shape[1] + 1):
        grp = GRP(n_components=c).fit(X_train_scaled)
        data_recon = np.dot(grp.transform(X_train_scaled), np.linalg.pinv(grp.components_.T))
        # data_error = np.sqrt((pd.DataFrame(data_recon) - pd.DataFrame(X_train_scaled)).apply(np.square).mean())
        rms = sqrt(mean_squared_error(data_recon, X_train_scaled))
        c_error.append(rms)
    runs.append(c_error)

runs_mean = np.mean(runs, axis=0)
runs_std = np.std(runs, axis=0)

fig, ax1 = plt.subplots()
ax1.plot(range(1, X_train_scaled.shape[1] + 1), runs_mean, label='mean')
ax1.plot(range(1, X_train_scaled.shape[1] + 1), runs_mean + runs_std, label='mean + std')
ax1.plot(range(1, X_train_scaled.shape[1] + 1), runs_mean - runs_std, label='mean - std')
ax1.grid()
ax1.legend()

ax1.set_xlabel('Number of components used')
ax1.set_ylabel('RMSE')
ax1.set_title('RP - Reconstruction Error - Covertype Dataset')

fig.savefig("{}/plots/p2_dim_reduction/rp/cover_error.png".format(root_path))



print("########## Performing Shopping RP... ##########")

run_count = 25
runs = []
for r in range(run_count):
    c_error = []
    for c in range(1, shop_df.shape[1] + 1):
        grp = GRP(n_components=c).fit(shop_df)
        data_recon = np.dot(grp.transform(shop_df), np.linalg.pinv(grp.components_.T))
        # data_error = np.sqrt((pd.DataFrame(data_recon) - pd.DataFrame(shop_df)).apply(np.square).mean())
        rms = sqrt(mean_squared_error(data_recon, shop_df))
        c_error.append(rms)
    runs.append(c_error)

runs_mean = np.mean(runs, axis=0)
runs_std = np.std(runs, axis=0)

fig, ax1 = plt.subplots()
ax1.plot(range(1, shop_df.shape[1] + 1), runs_mean, label='mean')
ax1.plot(range(1, shop_df.shape[1] + 1), runs_mean + runs_std, label='mean + std')
ax1.plot(range(1, shop_df.shape[1] + 1), runs_mean - runs_std, label='mean - std')
ax1.grid()
ax1.legend()

ax1.set_xlabel('Number of components used')
ax1.set_ylabel('RMSE')
ax1.set_title('RP - Reconstruction Error - Shopping Dataset')

fig.savefig("{}/plots/p2_dim_reduction/rp/shop_error.png".format(root_path))



print("########## Performing Covertype KernelPCA... ##########")

pca = KernelPCA(n_components=2, kernel='rbf')
principalComponents = pca.fit_transform(X_train_scaled)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1',
                                    'principal component 2'])
targetDf = pd.DataFrame(data=y_train, columns=['target'])
finalDf = pd.concat([principalDf, targetDf], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('KernelPCA RBF - 2 Principle Components - Covertype Dataset')
targets = range(7)
colors = ['y', 'g', 'b', 'c', 'm', 'r', 'orange']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c=color,
               s=5)
plt.legend(targets, markerscale=5)
ax.grid()
fig.savefig("{}/plots/p2_dim_reduction/kpca/cover_kpca_viz.png".format(root_path))

# run_count = 1
# runs = []
# for r in range(run_count):
#     c_error = []
#     for c in range(1, X_train_scaled.shape[1] + 1):
#         pca = KernelPCA(n_components=c, fit_inverse_transform=True).fit(X_train_scaled)
#         data_recon = pca.inverse_transform(X_train_scaled)
#         # data_error = np.sqrt((pd.DataFrame(data_recon) - pd.DataFrame(X_train_scaled)).apply(np.square).mean())
#         rms = sqrt(mean_squared_error(data_recon, X_train_scaled))
#         c_error.append(rms)
#     runs.append(c_error)

# runs_mean = np.mean(runs, axis=0)
# runs_std = np.std(runs, axis=0)

# fig, ax1 = plt.subplots()
# ax1.plot(range(1, X_train_scaled.shape[1] + 1), runs_mean, label='mean')
# # ax1.plot(range(1, X_train_scaled.shape[1] + 1), runs_mean + runs_std, label='mean + std')
# # ax1.plot(range(1, X_train_scaled.shape[1] + 1), runs_mean - runs_std, label='mean - std')
# ax1.grid()
# # ax1.legend()

# ax1.set_xlabel('Number of components used')
# ax1.set_ylabel('RMSE')
# ax1.set_title('KernelPCA - Reconstruction Error - Covertype Dataset')

# fig.savefig("{}/plots/p2_dim_reduction/kpca/cover_error.png".format(root_path))



print("########## Performing Shopping KernelPCA... ##########")

pca = KernelPCA(n_components=2, kernel='rbf')
principalComponents = pca.fit_transform(shop_df)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['principal component 1',
                                    'principal component 2'])
finalDf = pd.concat([principalDf, shop_copy_df['Revenue']], axis=1)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_title('KernelPCA RBF - 2 Principle Components - Shopping Dataset')

# ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'], s=5)

targets = [True, False]
colors = ['b', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Revenue'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c=color,
               s=5)

# plt.legend(targets, markerscale=5)
ax.grid()
fig.savefig("{}/plots/p2_dim_reduction/kpca/shopping_kpca_viz.png".format(root_path))

# run_count = 1
# runs = []
# for r in range(run_count):
#     c_error = []
#     for c in range(1, shop_df.shape[1] + 1):
#         pca = KernelPCA(n_components=c, fit_inverse_transform=True).fit(shop_df)
#         data_recon = pca.inverse_transform(shop_df)
#         # data_error = np.sqrt((pd.DataFrame(data_recon) - pd.DataFrame(shop_df)).apply(np.square).mean())
#         rms = sqrt(mean_squared_error(data_recon, shop_df))
#         c_error.append(rms)
#     runs.append(c_error)

# runs_mean = np.mean(runs, axis=0)
# runs_std = np.std(runs, axis=0)

# fig, ax1 = plt.subplots()
# ax1.plot(range(1, shop_df.shape[1] + 1), runs_mean, label='mean')
# # ax1.plot(range(1, shop_df.shape[1] + 1), runs_mean + runs_std, label='mean + std')
# # ax1.plot(range(1, shop_df.shape[1] + 1), runs_mean - runs_std, label='mean - std')
# ax1.grid()
# # ax1.legend()

# ax1.set_xlabel('Number of components used')
# ax1.set_ylabel('RMSE')
# ax1.set_title('KernelPCA - Reconstruction Error - Shopping Dataset')

# fig.savefig("{}/plots/p2_dim_reduction/kpca/shop_error.png".format(root_path))
