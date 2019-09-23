from time import time
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import neural_network
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

import generate_plots as gp

# EVIL CODE
import warnings
warnings.filterwarnings("ignore")

root_path = os.getcwd()

print("########## Importing Data... ##########")

# census data
adult_df = pd.read_csv("data/census_data/adult.data")
adult_df.dropna(inplace=True)
adult_test_df = pd.read_csv("data/census_data/adult.test")
adult_test_df.dropna(inplace=True)

# flag data
flag_df = pd.read_csv("data/flag_data/flag.data")
flag_df.dropna(inplace=True)

print("########## Splitting Data... ##########")
adult_x = adult_df.drop(['income'], axis=1)
# adult_x = adult_df
adult_y = adult_df['income']
adult_y = adult_y.to_frame()
adult_tst_x = adult_test_df.drop(['income'], axis=1)
# adult_tst_x = adult_test_df
adult_tst_y = adult_test_df['income']
adult_tst_y = adult_tst_y.to_frame()

flag_x = flag_df.drop(['religion'], axis=1)
flag_y = flag_df['religion']
flag_x, flag_tst_x, flag_y, flag_tst_y = train_test_split(flag_df,
                                                          flag_y,
                                                          test_size=0.33)

print("########## One-hot encoding... ##########")
categorical_feature_mask = (adult_x.dtypes == object)
categorical_cols = adult_x.columns[categorical_feature_mask].tolist()
column_mask = []
for column_name in list(adult_x.columns.values):
    column_mask.append(column_name in categorical_cols)

ohe = OneHotEncoder(categorical_features=column_mask, handle_unknown='ignore')
for col in categorical_cols:
    le = LabelEncoder()
    adult_x[col] = le.fit_transform(adult_x[col])
    adult_tst_x[col] = le.fit_transform(adult_tst_x[col])
adult_x = ohe.fit_transform(adult_x)
adult_tst_x = ohe.transform(adult_tst_x)


categorical_feature_mask = (flag_x.dtypes == object)
categorical_cols = flag_x.columns[categorical_feature_mask].tolist()
column_mask = []
for column_name in list(flag_x.columns.values):
    column_mask.append(column_name in categorical_cols)

ohe = OneHotEncoder(categorical_features=column_mask, handle_unknown='ignore')
for col in categorical_cols:
    le = LabelEncoder()
    flag_x[col] = le.fit_transform(flag_x[col])
    flag_tst_x[col] = le.fit_transform(flag_tst_x[col])
flag_x = ohe.fit_transform(flag_x)
flag_tst_x = ohe.transform(flag_tst_x)

print("########## Plotting Learning Curves... ##########")
nn_adult = neural_network.MLPClassifier(max_iter=10000)
nn_flag = neural_network.MLPClassifier(max_iter=1000)

start_time = time()

nn_adult_final = neural_network.MLPClassifier(max_iter=4000, hidden_layer_sizes=13, alpha=0.03)
nn_adult_final.fit(adult_x.todense(), adult_y.values.ravel())
adult_pred_y = nn_adult_final.predict(adult_tst_x.todense())
print("Adult neural network accuracy: {}".format(accuracy_score(adult_tst_y, adult_pred_y)))

print("Time elapsed: {}".format(time() - start_time))
start_time = time()

nn_flag_final = neural_network.MLPClassifier(max_iter=40000, hidden_layer_sizes=35, alpha=0.65)
nn_flag_final.fit(flag_x.todense(), flag_y.ravel())
flag_pred_y = nn_flag_final.predict(flag_tst_x.todense())
print("Flag neural network accuracy: {}".format(accuracy_score(flag_tst_y, flag_pred_y)))

print("Time elapsed: {}".format(time() - start_time))


fig_adult_lc = gp.plot_learning_curve(nn_adult,
                                      "Adult - learning curve",
                                      adult_x.todense(),
                                      adult_y.values.ravel(), cv=3,
                                      train_sizes=np.linspace(0.1, 1.0, 7))
fig_adult_lc.savefig(root_path + "/plots/nn/adult_lc.png")

fig_flag_lc = gp.plot_learning_curve(nn_flag,
                                     "Flag - learning curve",
                                     flag_x.todense(),
                                     flag_y.values.ravel(), cv=3,
                                     train_sizes=np.linspace(0.1, 1.0, 50))
fig_flag_lc.savefig(root_path + "/plots/nn/flag_lc.png")

print("########## Plotting alpha Validation Curves... ##########")
fig_adult_vc1 = gp.plot_validation_curve(nn_adult,
                                         "Adult - alpha Validation Curve",
                                         adult_x.todense(),
                                         adult_y.values.ravel(),
                                         param_name="alpha",
                                         param_range=np.linspace(0.01, 0.05, 5),
                                         cv=3)
fig_adult_vc1.savefig(root_path + "/plots/nn/alpha_adult_vc.png")

fig_flag_vc1 = gp.plot_validation_curve(nn_flag,
                                        "Flag - alpha Validation Curve",
                                        flag_x.todense(),
                                        flag_y.values.ravel(),
                                        param_name="alpha",
                                        param_range=10.0 ** -np.arange(0.1, 7, 0.1),
                                        cv=5)
fig_flag_vc1.savefig(root_path + "/plots/nn/alpha_flag_vc.png")

sizes = []
for x in range(1, 6):
    sizes.append((x,))
print(sizes)
print("########## Plotting hidden_layer_sizes Validation Curves... ##########")
fig_adult_vc2 = gp.plot_validation_curve(nn_adult,
                                         "Adult - hidden_layer_sizes Validation Curve",
                                         adult_x.todense(),
                                         adult_y.values.ravel(),
                                         param_name="hidden_layer_sizes",
                                         param_range=sizes,
                                         cv=5)
fig_adult_vc2.savefig(root_path + "/plots/nn/hidden_layer_sizes_adult_vc.png")

fig_flag_vc2 = gp.plot_validation_curve(nn_flag,
                                        "Flag - hidden_layer_sizes Validation Curve",
                                        flag_x.todense(),
                                        flag_y.values.ravel(),
                                        param_name="hidden_layer_sizes",
                                        param_range=sizes,
                                        cv=5)
fig_flag_vc2.savefig(root_path + "/plots/nn/hidden_layer_sizes_flag_vc.png")

fig_adult_iter = gp.plot_validation_curve(nn_adult,
                                          "Adult - Iterations Validation Curve",
                                          adult_x.todense(),
                                          adult_y.values.ravel(),
                                          param_name="max_iter",
                                          param_range=range(1, 10000, 2500),
                                          cv=3)
fig_adult_iter.savefig(root_path + "/plots/nn/iterations_adult_lc.png")

fig_flag_iter = gp.plot_validation_curve(nn_flag,
                                         "Flag - Iterations Validation Curve",
                                         flag_x.todense(),
                                         flag_y.values.ravel(),
                                         param_name="max_iter",
                                         param_range=range(1, 50000, 2500),
                                         cv=5)
fig_flag_iter.savefig(root_path + "/plots/nn/iterations_flag_lc.png")


print("Time elapsed: {}".format(time() - start_time))
