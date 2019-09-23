---- PROJECT CODE ----
My code can be found in the following github repository:
https://github.com/jdeweerth/ml_submissions

The following files will read in data and generate the plots for their respective classifier:
decision_tree.py
knn.py
boost.py
neural_network.py
svm.py

The following file is a supporting file used to plot validation and learning curves, 
it has been adapted from scikit-learn examples:
generate_plots.py

---- PYTHON DEPENDENCIES ----
My code must all be executed with a Python 3.6.7 environment using the following packages:
cycler==0.10.0
joblib==0.13.2
kiwisolver==1.1.0
matplotlib==3.1.1
numpy==1.17.1
pandas==0.25.1
pyparsing==2.4.2
python-dateutil==2.8.0
pytz==2019.2
scikit-learn==0.21.3
scipy==1.3.1
six==1.12.0

---- DATA SETS ----
My data are included in the repository, but can also be found at:
https://archive.ics.uci.edu/ml/datasets/Census+Income
https://archive.ics.uci.edu/ml/datasets/Flags

The adult set was reduced in size and the test to train ratios were made 1:2 for both sets. 
In addition I added the column headers to the files to label the features.