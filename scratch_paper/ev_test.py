# -*- coding: utf-8 -*-

import header as h
import pandas as pd
import numpy as np
import fs_definitions as fsd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

np.set_printoptions(suppress=True)

df = h.create_fight_df('../data/ufc-master.csv')
fs = 'c3d'
temp_df = fsd.create_prepped_df(fs, df)


print(df.shape)
print(temp_df.shape)
print(temp_df[['R_ev', 'R_ev_final', 'B_ev_final', 'Winner', 'label']])
y = temp_df['label']


#WE NEED TO STRIP OUT R_ev_final and B_ev_final 
#I THINK WE CAN DO THIS AND THEN SPLIT....

ev_df = temp_df[['B_ev_final', 'R_ev_final']]

temp_prepped_df = temp_df.drop(['Winner', 'label', 'R_ev_final',
                                'B_ev_final'], axis=1)

X = temp_prepped_df.values
X_ev = ev_df.values


print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state = 85)

X_train_ev, X_test_ev, y_train_ev, y_test_ev = train_test_split(X_ev, y, 
                                                    test_size=0.1,
                                                    random_state = 85)



print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

print('ev Testing shape:', X_test_ev.shape)

classifier = h.get_classifier(fs)

#print(classifier.get_params)

classifier.fit(X_train, y_train)

probs = classifier.predict_proba(X_test)
preds = classifier.predict(X_test)

print(probs.shape)
print(preds.shape)

preds = preds.reshape((len(preds),1))

#print(probabilities)

p_and_p = np.append(probs, preds, 1)
p_and_p = np.append(p_and_p, X_test_ev, 1)


print(p_and_p)

print(p_and_p.shape)

print(X_test.shape)


    
    


test_df = h.get_test_probs(df, 'c1', 85, .1)
    
print(test_df.head)