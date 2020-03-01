---- PROJECT CODE ----
My code can be found in the following github repository:
https://github.com/jdeweerth/ml_submissions

The following files will read in data and generate the plots for their respective classifier:
4peaks_plots.py
knapsack_plots.py
flipflop_plots.py
nn_plots.py

The following file contains a modification to one of the mlrose_hiive package files, and must be used to replace the mlrose_hiive version at mlrose_hiive/neural/utils/weights.py:
weights.py

This last file is a supporting file used to help generate some of the plots used:
generate_plots.py

---- PYTHON DEPENDENCIES ----
My code must all be executed with a Python 3.6.7 environment using the following packages:
cycler==0.10.0
decorator==4.4.1
joblib==0.13.2
kiwisolver==1.1.0
matplotlib==3.1.1
mlrose-hiive==2.1.2
mlrose-reborn==2.0.0
networkx==2.4
numpy==1.17.1
pandas==0.25.1
pyparsing==2.4.2
python-dateutil==2.8.0
pytz==2019.2
scikit-learn==0.21.3
scipy==1.3.1
six==1.12.0

---- DATA SETS ----
My dataset is included in scikit-learn, but can also be found at:
https://archive.ics.uci.edu/ml/datasets/Covertype

I dramatically reduced the size of my dataset to 4000 and 2000 training and testing instances respectively. 