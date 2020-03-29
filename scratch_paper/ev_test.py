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



df = h.create_fight_df('../data/ufc-master.csv')
fs = 'c3d'
temp_df = fsd.create_prepped_df(fs, df)


print(df.shape)
print(temp_df.shape)

y = temp_df['label']

temp_prepped_df = temp_df.drop(['Winner', 'label'], axis=1)

X = temp_prepped_df.values

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state = 85)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

classifier = h.get_classifier(fs)

print(classifier.get_params)

classifier.fit(X_train, y_train)

probs = classifier.predict_proba(X_test)
preds = classifier.predict(X_test)

print(probs.shape)
print(preds.shape)

preds = preds.reshape((len(preds),1))

#print(probabilities)

p_and_p = np.append(probs, preds, 1)

print(p_and_p)

print(p_and_p.shape)
