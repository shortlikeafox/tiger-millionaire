import pandas as pd
import numpy as np
from functions import *
import random
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.mixture import DPGMM

remove_fight_island = False


#Turn off warnings

import warnings
warnings.filterwarnings("ignore")

#Load models
#REMINDER: We are going to need to use 'eval' to get the models usable
with open("./data/models.csv", newline='') as f:
    reader = csv.reader(f)
    models = list(reader)
    
print(len(models))


###SELECT MODEL TO OPTIMIZE
#model_num = 19
