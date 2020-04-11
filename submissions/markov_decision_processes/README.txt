---- PROJECT CODE ----
My code can be found in the following github repository:
https://github.com/jdeweerth/ml_submissions

All algorithms/MDPs were based on the following two libraries:
https://github.com/hiive/hiivemdptoolbox
https://github.com/openai/gym

The following files will read in data and generate the plots for their respective MDP:
forest.py (for the Forest Management problem)
frozen_lake.py (for the 4x4 Frozen Lake problem)
frozen_lake_big.py (for the 8x8 Frozen Lake problem)

The following file contains a modification to one of the hiive/mdptoolbox package files, and must be used to replace the default version at hiive/mdptoolbox/mdp.py:
mdp.py

---- PYTHON DEPENDENCIES ----
My code must all be executed with a Python 3.6.7 environment using the following packages:
appnope          0.1.0  
backcall         0.1.0  
cloudpickle      1.3.0  
cycler           0.10.0 
decorator        4.4.1  
future           0.18.2 
gym              0.17.1 
ipython          7.13.0 
ipython-genutils 0.2.0  
jedi             0.16.0 
joblib           0.13.2 
kiwisolver       1.1.0  
matplotlib       3.1.1  
mdptoolbox-hiive 4.0.3.1
mlrose-hiive     2.1.2  
mlrose-reborn    2.0.0  
networkx         2.4    
numpy            1.17.1 
pandas           0.25.1 
parso            0.6.2  
pexpect          4.8.0  
pickleshare      0.7.5  
pip              20.0.2 
prompt-toolkit   3.0.5  
ptyprocess       0.6.0  
pydot            1.4.1  
pyglet           1.5.0  
Pygments         2.6.1  
pyparsing        2.4.2  
python-dateutil  2.8.0  
pytz             2019.2 
scikit-learn     0.21.3 
scipy            1.3.1  
setuptools       39.0.1 
six              1.12.0 
traitlets        4.3.3  
wcwidth          0.1.9  
