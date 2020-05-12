# -*- coding: utf-8 -*-

import header as h
import fs_definitions as fsd
import numpy as np
import pandas as pd




fs = 'c1'
df = h.create_fight_df('../data/ufc-master.csv')
"""
#We want to figure out how to split on the date....


prepped_df = fsd.create_prepped_df(fs, df)


list_of_dates = (prepped_df['date_final'].unique())

list_of_dates = (np.flip(np.sort(list_of_dates)))

list_of_dates = list_of_dates[:50]

print(list_of_dates)

y = prepped_df[['label', 'date_final']]
ev_df = prepped_df[['date_final', 'B_ev_final', 'R_ev_final', 'Winner', 'label']]

prepped_df = prepped_df.drop(['Winner', 'label', 'R_ev_final',
                                'B_ev_final'], axis=1)



final_probs = None


for d in list_of_dates:
    X_test = prepped_df.loc[prepped_df['date_final'] == d]
    X_train = prepped_df.loc[prepped_df['date_final'] != d]
    y_test = y.loc[y['date_final'] == d]
    y_train = y.loc[y['date_final'] != d]
    X_test_ev = ev_df.loc[ev_df['date_final'] == d]
    X_train_ev = ev_df.loc[ev_df['date_final'] != d]
    


    #Remove the date
    X_test = X_test.drop('date_final', 1)
    X_train = X_train.drop('date_final', 1)
    y_train = y_train.drop('date_final', 1)
    y_test = y_test.drop('date_final', 1)
    print(X_test.shape)
    print(X_train.shape)
    print(y_test.shape)
    print(y_train.shape)
    print(X_test_ev.shape)
    print(X_train_ev.shape)
    print()
    print()
    
    classifier = h.get_classifier(fs)
    classifier.fit(X_train, y_train.values.ravel())
    probs = classifier.predict_proba(X_test)
    preds = classifier.predict(X_test)
    
    
    preds = preds.reshape((len(preds),1))
    X_test = np.append(X_test, probs,1)
    X_test = np.append(X_test, preds,1)
    X_test = np.append(X_test, X_test_ev,1)
    trash_df = prepped_df.drop(['date_final'], axis=1)
    colNamesArr = trash_df.columns.values
    colNamesArr = np.append(colNamesArr, ['B_prob', 'R_prob', 
                                          'preds', 'date_final',  
                                          'B_ev_final','R_ev_final', 
                                          'Winner', 'label'])
    final_df = pd.DataFrame(X_test)
    final_df.columns = colNamesArr    

    if (final_probs is not None):
        final_probs = final_probs.append(final_df)
    else:
        final_probs = final_df
"""
final_probs = h.get_recent_probs(df, fs)
print(final_probs.shape)
final_probs.to_csv('test_recent.csv')
results_df = h.get_bet_results(final_probs)
results_df.to_csv('recent_results.csv')