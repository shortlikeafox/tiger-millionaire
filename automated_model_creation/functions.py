# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:26:01 2020

@author: matth
"""

import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
# -*- coding: utf-8 -*-

#Input: American Odds, and Probability of a Winning Bet
#Output: Bet EV based on a $100 bet
def get_bet_ev(odds, prob):
    if odds>0:
        return ((odds * prob) - (100 * (1-prob)) )
    else:
        return ((100 / abs(odds))*100*prob - (100 * (1-prob)))


#Input: American Odds
#Output: Profit on a successful bet
def get_bet_return(odds):
    if odds>0:
        return odds
    else:
        return (100 / abs(odds))*100


def get_ev_from_df(ev_df, print_stats = False, min_ev = 0, get_total=False):
    num_matches = 0
    num_bets = 0
    num_wins = 0
    num_losses= 0
    num_under= 0
    num_under_losses = 0
    num_under_wins = 0
    num_even = 0
    num_even_losses = 0
    num_even_wins = 0
    num_fav = 0
    num_fav_wins = 0
    num_fav_losses = 0
    profit = 0
    profit_per_bet = 0
    profit_per_match = 0    

    for index, row in ev_df.iterrows():
        num_matches = num_matches+1
        t1_bet_ev = get_bet_ev(row['t1_odds'], row['t1_prob'])
        #print(f"ODDS:{row['t1_odds']} PROB: {row['t1_prob']} EV: {t1_bet_ev}")
        t2_bet_ev = get_bet_ev(row['t2_odds'], row['t2_prob'])
        #print(f"ODDS:{row['t2_odds']} PROB: {row['t2_prob']} EV: {t2_bet_ev}")
        #print()
        
        t1_bet_return = get_bet_return(row['t1_odds'])
        t2_bet_return = get_bet_return(row['t2_odds'])
        
        
        if (t1_bet_ev > min_ev or t2_bet_ev > min_ev):
            num_bets = num_bets+1

            
        if t1_bet_ev > min_ev:
            if row['winner'] == 0:
                num_wins += 1
                profit = profit + t1_bet_return
                #print(t1_bet_return)
            elif row['winner'] == 1:
                num_losses += 1
                profit = profit - 100
            if (t1_bet_return > t2_bet_return):
                num_under += 1
                if row['winner'] == 0:
                    num_under_wins += 1
                elif row['winner'] == 1:
                    num_under_losses += 1
            elif (t1_bet_return < t2_bet_return):
                num_fav += 1
                if row['winner'] == 0:
                    num_fav_wins += 1
                elif row['winner'] == 1:
                    num_fav_losses += 1
            else:
                num_even += 1
                if row['winner'] == 0:
                    num_even_wins += 1
                elif row['winner'] == 1:
                    num_even_losses += 1

        if t2_bet_ev > min_ev:
            if row['winner'] == 1:
                num_wins += 1                    
                profit = profit + t2_bet_return
            elif row['winner'] == 0:
                num_losses += 1
                profit = profit - 100
            if (t2_bet_return > t1_bet_return):
                num_under += 1
                if row['winner'] == 1:
                    num_under_wins += 1
                elif row['winner'] == 0:
                    num_under_losses += 1
            elif (t2_bet_return < t1_bet_return):
                num_fav += 1
                if row['winner'] == 1:
                    num_fav_wins += 1
                elif row['winner'] == 0:
                    num_fav_losses += 1
            else:
                num_even += 1
                if row['winner'] == 1:
                    num_even_wins += 1
                elif row['winner'] == 0:
                    num_even_losses += 1
            
    if num_bets > 0:
        profit_per_bet = profit / num_bets
    else:
        profit_per_bet = 0
    if num_matches > 0:
        profit_per_match = profit / num_matches
    else:
        profit_per_match = 0
        
    if print_stats:
        print(f"""
          Number of matches: {num_matches}
          Number of bets: {num_bets}
          Number of winning bets: {num_wins}
          Number of losing bets: {num_losses}
          Number of underdog bets: {num_under}
          Number of underdog wins: {num_under_wins}
          Number of underdog losses: {num_under_losses}
          Number of Favorite bets: {num_fav}
          Number of favorite wins: {num_fav_wins}
          Number of favorite losses: {num_fav_losses}
          Number of even bets: {num_even}
          Number of even wins: {num_even_wins}
          Number of even losses: {num_even_losses}
          Profit: {profit}
          Profit per bet: {profit_per_bet}
          Profit per match: {profit_per_match}
          
          """)
    if (get_total):
        #print(f"# Matches: {num_matches}, # Bets: {num_bets} # Wins: {num_wins}")
        return(profit)
    else:
        return (profit_per_bet)




#INPUT: 
#df: The df to be evaluated
#m: The model to use
# labels: The labels
#odds: The odds
#min_ev: The minimum EV to place a bet
def custom_cv_eval(df, m, labels, odds, min_ev=0, verbose=False, get_total=False):
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
        running_total = running_total + get_ev_from_df(ev_prepped_df, print_stats = verbose, min_ev = min_ev, get_total=get_total)
        #display(ev_prepped_df)
    
    return running_total