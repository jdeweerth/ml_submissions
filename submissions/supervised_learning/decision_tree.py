from time import time
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import tree
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
dt_adult = tree.DecisionTreeClassifier()
dt_flag = tree.DecisionTreeClassifier()

start_time = time()

dt_adult_final = tree.DecisionTreeClassifier(max_depth=7, max_leaf_nodes=5)
dt_adult_final.fit(adult_x, adult_y)
adult_pred_y = dt_adult_final.predict(adult_tst_x)
print("Adult decision tree accuracy: {}".format(accuracy_score(adult_tst_y, adult_pred_y)))

print("Time elapsed: {}".format(time() - start_time))
start_time = time()

dt_flag_final = tree.DecisionTreeClassifier()
dt_flag_final.fit(flag_x, flag_y)
flag_pred_y = dt_flag_final.predict(flag_tst_x)
print("Flag decision tree accuracy: {}".format(accuracy_score(flag_tst_y, flag_pred_y)))

print("Time elapsed: {}".format(time() - start_time))

fig_adult_lc = gp.plot_learning_curve(dt_adult,
                                      "adult learning curve",
                                      adult_x, adult_y, cv=3,
                                      train_sizes=np.linspace(0.1, 1.0, 20))
fig_adult_lc.savefig(root_path + "/plots/decision_tree/adult_lc.png")

fig_flag_lc = gp.plot_learning_curve(dt_flag,
                                     "flag learning curve",
                                     flag_x, flag_y, cv=3,
                                     train_sizes=np.linspace(0.1, 1.0, 20))
fig_flag_lc.savefig(root_path + "/plots/decision_tree/flag_lc.png")

print("########## Plotting Max Depth Validation Curves... ##########")
fig_adult_vc1 = gp.plot_validation_curve(dt_adult,
                                         "Max Depth Adult Validation Curve",
                                         adult_x, adult_y,
                                         param_name="max_depth",
                                         param_range=np.linspace(1, 50, 25),
                                         cv=5)
fig_adult_vc1.savefig(root_path +
                      "/plots/decision_tree/max_depth_adult_vc.png")

fig_flag_vc1 = gp.plot_validation_curve(dt_flag,
                                        "Max Depth Flag Validation Curve",
                                        flag_x, flag_y,
                                        param_name="max_depth",
                                        param_range=np.linspace(1, 50, 25),
                                        cv=5)
fig_flag_vc1.savefig(root_path +
                     "/plots/decision_tree/max_depth_flag_vc.png")

print("########## Plotting Max Leaf Nodes Validation Curves... ##########")
fig_adult_vc2 = gp.plot_validation_curve(dt_adult,
                                         "Max Depth Adult Validation Curve",
                                         adult_x, adult_y,
                                         param_name="max_leaf_nodes",
                                         param_range=range(2, 20),
                                         cv=5)
fig_adult_vc2.savefig(root_path +
                      "/plots/decision_tree/max_leaf_nodes_adult_vc.png")

fig_flag_vc2 = gp.plot_validation_curve(dt_flag,
                                        "Max Depth Flag Validation Curve",
                                        flag_x, flag_y,
                                        param_name="max_leaf_nodes",
                                        param_range=range(2, 25),
                                        cv=5)
fig_flag_vc2.savefig(root_path +
                     "/plots/decision_tree/max_leaf_nodes_flag_vc.png")

print("Time elapsed: {}".format(time() - start_time))
