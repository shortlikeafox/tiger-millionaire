# -*- coding: utf-8 -*-

#INPUT: 
#df: The df to be evaluated
#m: The model to use
# labels: The labels
#odds: The odds
#min_ev: The minimum EV to place a bet
import numpy as np
def custom_cv_eval(df, m, labels, odds, min_ev=0):
    X = np.array(df)
    y = np.array(labels)
    odds = np.array(odds)
    running_total = 0
    count=1
    kf = KFold(n_splits=5, shuffle=True, random_state=75)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        odds_train, odds_test = odds[train_index], odds[test_index]
        #display(y_train)
        m.fit(X_train, y_train)
        probs=m.predict_proba(X_test)
        #print(probs)
        #We need to prep the dataframe to evaluate....
        #X_odds = X_test[['t1_odds', 't2_odds']]
        #print(X_test)
        #print(X_test[:, -1])
        #print(X_test[:, -2])
        X_odds = list(zip(odds_test[:, -2], odds_test[:, -1], probs[:, 0], probs[:, 1], y_test))
        ev_prepped_df = pd.DataFrame(X_odds, columns=['t1_odds', 't2_odds', 't1_prob', 't2_prob', 'winner'])
        #display(ev_prepped_df)
        #display(temp_df)
        #print(f"{count}: {get_ev_from_df(ev_prepped_df, print_stats = False)}")
        count=count+1
        running_total = running_total + get_ev_from_df(ev_prepped_df, print_stats = False, min_ev = min_ev)
        #display(ev_prepped_df)
    
    return running_total