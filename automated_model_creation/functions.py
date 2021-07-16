# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:26:01 2020

@author: matth
"""

import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import NuSVC
from sklearn.svm import SVC
#from sklearn.mixture import DPGMM
# -*- coding: utf-8 -*-


def get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev = 0, verbose=False, get_total=True):
    df_sel = input_df[input_features]
    df_sel = df_sel.dropna()
    df_sel = pd.get_dummies(df_sel)
    labels_sel = input_labels[input_labels.index.isin(df_sel.index)]
    odds_sel = odds_input[odds_input.index.isin(df_sel.index)] 
    if len(odds_sel.columns) == 6:
        best_score = custom_cv_eval_mov(df_sel, input_model, labels_sel, odds_sel, min_ev = min_ev, verbose=verbose, 
                                get_total=get_total)        
    else:
        best_score = custom_cv_eval(df_sel, input_model, labels_sel, odds_sel, min_ev = min_ev, verbose=verbose, 
                                get_total=get_total)

    
    return best_score


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






#    get_ev_from_df_mov(odds_test, probs, labels_test, label_list, print_stats = True, min_ev = input_ev, get_total=True)

def get_ev_from_df_mov(df_odds, probs, labels, label_list, probs_label_list, print_stats = False, min_ev = 0, get_total=True):
    probs_label_list = [int(a) for a in probs_label_list]
    #labels = [int(a) for a in labels]
    
    
    df_odds.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)
    score = 0
    #print(df_odds)
    for i in range(len(df_odds)):
        #print(i)
        #        df_temp_odds = df_odds.iloc[[i, :]]
        #print(df_odds.iloc[[i]])
        for l in range(len(probs[i])):
            #print(f"{label_list[probs_label_list[l]]}: {probs[i][l]}")
            temp_odds = (df_odds.loc[[i]])[label_list[probs_label_list[l]]][i]
            #print((temp_odds))
            bet_ev = get_bet_ev(temp_odds, probs[i][l])
            #print(bet_ev)
            if bet_ev > min_ev:
                #print(l)
                if labels[i] == probs_label_list[l]:
                    #print(f"{int(labels[i])} {probs_label_list[l]}")
                    score = score + get_bet_return(temp_odds)
                    #print(f"Winning Bet. New Score: {score}")
                else:
                    score = score - 100
                    #print(f"Losing Bet.  New Score: {score}")
                    
            #print()
            
            
            
        #print(f"Result: {label_list[int(labels[i])]} ({int(labels[i])})")
    print("Real Score: " + str(score))
    return(score)


def get_ev_from_df(ev_df, print_stats = False, min_ev = 0, get_total=True):
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
def custom_cv_eval(df, m, labels, odds, min_ev=0, verbose=False, get_total=True):
    #If we have less than 5 samples we are going to break the split.
    if len(df) < 5:
        return 0
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
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(X_train)
        scaled_test = scaler.transform(X_test)
        
        m.fit(scaled_train, y_train)
        probs=m.predict_proba(scaled_test)
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






def get_ev_for_optimize_mov(df_odds, probs, labels,  print_stats = False, min_ev = 0, get_total=True):
        
    score = 0
    #print(df_odds)
    for i in range(len(df_odds)):
        #print(i)
        #        df_temp_odds = df_odds.iloc[[i, :]]
        #print()
        #print()
        #print(df_odds[i])
        for l in range(len(probs[i])):
            temp_odds = (df_odds[i][l])
            #print((temp_odds))
            bet_ev = get_bet_ev(temp_odds, probs[i][l])
            #print(bet_ev)
            if bet_ev > min_ev:
                #print(l)
                if labels[i] == l:
                    #print(f"{int(labels[i])} {l}")
                    score = score + get_bet_return(temp_odds)
                    #print(f"Winning Bet. New Score: {score}")
                else:
                    score = score - 100
                    #print(f"Losing Bet.  New Score: {score}")
                    
            #print()
            
            
            
        #print(f"Result: {labels[i]}")
    return(score)



def custom_cv_eval_mov(df, m, labels, odds, min_ev=0, verbose=False, get_total=True):
    #If we have less than 5 samples we are going to break the split.
    #print("HI")
    if len(df) < 5:
        return 0
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
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(X_train)
        scaled_test = scaler.transform(X_test)
        
        m.fit(scaled_train, y_train)
        probs=m.predict_proba(scaled_test)
        #print(probs)
        #We need to prep the dataframe to evaluate....
        #X_odds = X_test[['t1_odds', 't2_odds']]
        #print(X_test)
        #print(X_test[:, -1])
        #print(X_test[:, -2])
        #X_odds = list(zip(odds_test[:, -2], odds_test[:, -1], probs[:, 0], probs[:, 1], y_test))
        #ev_prepped_df = pd.DataFrame(X_odds, columns=['t1_odds', 't2_odds', 't1_prob', 't2_prob', 'winner'])
        #display(ev_prepped_df)
        #display(temp_df)
        #print(f"{count}: {get_ev_from_df(ev_prepped_df, print_stats = False)}")
        count=count+1
        running_total = running_total + get_ev_for_optimize_mov(odds_test, probs, y_test,  min_ev= min_ev, get_total=get_total )

        #display(ev_prepped_df)
    
    return running_total



#We split off the labels and the odds.  Now we can rewrite the function
#INPUT
#pos_features: The list of possible features
#m: The model
#cur_features: The list of current features
#scale: Does the data need to be scaled?  
def get_best_features(pos_features, m, df, cur_features, labels, odds, scale=False, min_ev=0):
    best_feature = ''
        
    #If there are no current features...
    if len(cur_features) == 0:
        best_score = -10000
    else:
        df_sel = df[cur_features]
        df_sel = df_sel.dropna()
        df_sel = pd.get_dummies(df_sel)
        #OK we need to filter the labels and odds based off of the indices
        labels_sel = labels[labels.index.isin(df_sel.index)]
        odds_sel = odds[odds.index.isin(df_sel.index)]        
        best_score = custom_cv_eval(df_sel, m, labels_sel, odds_sel, min_ev=min_ev)
        
    best_feature = ""
    
    print(f"Current best score is: {best_score}")
    #Go thru every feature and test it...
    for f in pos_features:
        #If f is not a current feature
        if f not in cur_features:
            new_features = [f] + cur_features
            df_sel = df[new_features]
            df_sel = df_sel.dropna()
            df_sel = pd.get_dummies(df_sel)
            #display(df_sel)
            #OK we need to filter the labels and odds based off of the indices
            labels_sel = labels[labels.index.isin(df_sel.index)]
            odds_sel = odds[odds.index.isin(df_sel.index)]
            new_score = custom_cv_eval(df_sel, m, labels_sel, odds_sel, min_ev=min_ev)
            #print(f"{len(df_sel)} {len(labels_sel)} {len(odds_sel)}")
            if new_score > best_score:
                print(f"Feature: {f} Score: {new_score}")
                best_score = new_score
                best_feature = f
    if best_feature != "":
        print(f"The best feature was {best_feature}.  It scored {best_score}")
        cur_features = [best_feature] + cur_features
        #Keep running until we don't improve
        return(get_best_features(pos_features, m, df, cur_features, labels, odds, scale, min_ev=min_ev))
    else:
        print("NO IMPROVEMENT")
        print(f"FINAL BEST SCORE: {best_score}")
        return cur_features                
                
    return []




def get_best_features_mov(pos_features, m, df, cur_features, labels, odds, label_list, scale=False, min_ev=0):
    best_feature = ''
        
    #If there are no current features...
    if len(cur_features) == 0:
        best_score = -1000000
    else:
        df_sel = df[cur_features]
        df_sel = df_sel.dropna()
        df_sel = pd.get_dummies(df_sel)
        #OK we need to filter the labels and odds based off of the indices
        labels_sel = labels[labels.index.isin(df_sel.index)]
        odds_sel = odds[odds.index.isin(df_sel.index)]        
        labels_sel = labels_sel.dropna()
        odds_sel = odds_sel[odds_sel.index.isin(labels_sel.index)]     
        df_sel = df_sel[df_sel.index.isin(labels_sel.index)] 
        best_score = custom_cv_eval_mov(df_sel, m, labels_sel, odds_sel, min_ev=min_ev)
        
        
    best_feature = ""
    
    print(f"Current best score is: {best_score}")
    #Go thru every feature and test it...
    for f in pos_features:
        #If f is not a current feature
        if f not in cur_features:
            new_features = [f] + cur_features
            df_sel = df[new_features]
            df_sel = df_sel.dropna()
            df_sel = pd.get_dummies(df_sel)
            #display(df_sel)
            #OK we need to filter the labels and odds based off of the indices
            labels_sel = labels[labels.index.isin(df_sel.index)]
            odds_sel = odds[odds.index.isin(df_sel.index)]
            labels_sel = labels_sel.dropna()
            odds_sel = odds_sel[odds_sel.index.isin(labels_sel.index)]     
            df_sel = df_sel[df_sel.index.isin(labels_sel.index)] 
            
            
            new_score = custom_cv_eval_mov(df_sel, m, labels_sel, odds_sel, min_ev=min_ev)
            #print(f"{len(df_sel)} {len(labels_sel)} {len(odds_sel)}")
            if new_score > best_score:
                print(f"Feature: {f} Score: {new_score}")
                best_score = new_score
                best_feature = f
    if best_feature != "":
        print(f"The best feature was {best_feature}.  It scored {best_score}")
        cur_features = [best_feature] + cur_features
        #Keep running until we don't improve
        return(get_best_features_mov(pos_features, m, df, cur_features, labels, odds, label_list,  scale, min_ev=min_ev))
    else:
        print("NO IMPROVEMENT")
        print(f"FINAL BEST SCORE: {best_score}")
        return cur_features                
                
    return []




def tune_LogisticRegression(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. penalty ('l1' or 'l2')
    #2. tol (original_value, original_value * 1.2, original_value * 0.8, rand(0, 10)
    #3. random_state = 75
    #4. solver = 'newton-cg', 'lbfgs', 'sag', 'saga'    
    ###############################################################################################################
    print()
    print()
    print("Starting New Run for LogisticRegression")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    

    
    penalty = ['l1', 'l2', 'none']
    solver = ['newton-cg', 'lbfgs', 'sag']
    tol = [input_model.tol, input_model.tol * 1.2, input_model.tol * .8, random.random() * 10 ]
    for s in solver:
        score = -10000
        for p in penalty:
            for t in tol:
                if ((s == 'newton-cg') & (p == 'l1')) |\
                ((s == 'lbfgs') & (p == 'l1')) |\
                ((s == 'sag') & (p == 'l1')):

                    pass
                else:
                    test_model = LogisticRegression(solver = s, penalty = p, tol=t, random_state=75, max_iter=50000)
                    score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
                    if score > best_score:
                        best_score = score
                        output_model = test_model
                        
                        print()
                        print("NEW BEST SCORE")
                        print("solver:", s, 
                              "penalty:", p,
                              "tol:", t,
                              "Best Score:", best_score)        
                        print()
                        print()
                    else:
                        pass
                        print("solver:", s, 
                              "penalty:", p,
                              "tol:", t,
                              "Score:", score)                                                       
    return(output_model)

def tune_DecisionTreeClassifier(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. criterion ('gini', 'entropy')
    #2. splitter ('random', 'best')
    #3. max_depth ('none', IF A NUMBER EXISTS +1, -1, random, else 2 RANDOM INTS 1->100)
    #4. min_samples_leaf(n-1, 0,  n+1)
    #5. max_leaf_nodes:('none', n+1, n-1, OR 4 random numbers)
    ###############################################################################################################
    print()
    print()
    print("Starting New Run for DecisionTree")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    

    criterion = ['gini', 'entropy']
    splitter = ['random', 'best']
    if input_model.max_depth == None:
        max_depth = [None, random.randrange(100)+1, random.randrange(100)+1]
    else:
        max_depth = [input_model.max_depth, input_model.max_depth - 1, input_model.max_depth + 1, random.randrange(100)+1]
        max_depth = [i for i in max_depth if i > 0]

    min_samples_leaf = [input_model.min_samples_leaf, input_model.min_samples_leaf *1.01,
                         input_model.min_samples_leaf*0.99]
    min_samples_leaf = [i for i in min_samples_leaf if i > 0]    
    if ((input_model.max_leaf_nodes == None) or (input_model.max_leaf_nodes == 1)):
        max_leaf_nodes = [None, random.randrange(1000)+1, random.randrange(1000)+1]
    else:
        max_leaf_nodes = [input_model.max_leaf_nodes, input_model.max_leaf_nodes - 1, 
                     input_model.max_leaf_nodes + 1, random.randrange(1000)+1]
        max_leaf_nodes = [i for i in max_leaf_nodes if i > 0]
        
    for l in max_leaf_nodes:
        for sam in min_samples_leaf:
            for m in max_depth:
                for c in criterion:
                    for s in splitter:
                        test_model = DecisionTreeClassifier(criterion = c, splitter = s, max_depth = m,
                                                            min_samples_leaf=sam, max_leaf_nodes = l, random_state=75)
                        score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
                        if score > best_score:
                            best_score = score
                            output_model = test_model
                            print()
                            print("NEW BEST SCORE")
                            
                            print("Criterion:", c, "splitter:", s, "max_depth:", m, 
                                  "min_samples_leaf:", sam, "max_leaf_nodes:", l, best_score)        
                            print()
                        else:
                            pass
                            print("Criterion:", c, "splitter:", s, "max_depth:", m, 
                                  "min_samples_leaf:", sam, "max_leaf_nodes:", l, score)        
                            
                                        
    
    return output_model

def tune_RandomForestClassifier(input_model, input_features, input_df, input_labels, odds_input, min_ev=0, tested_hps = []):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. criterion ('gini', 'entropy')
    #2. max_features ('auto', 'sqrt', 'log2')
    #3. max_depth ('none', IF A NUMBER EXISTS +2, -2, ELSE 2 RANDOM INTS 1->100)
    #4. min_samples_leaf(n-2, 0, n+2)
    #5. max_leaf_nodes:('none', n+2, n-2, OR 2 random numbers)
    #6. n_estimators: (n, n+2, n-2)
    ###############################################################################################################    
    print()
    print()
    print("Starting New Run for RandomForestClassifier")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)        
    #1. criterion ('gini', 'entropy')
    criterion = ['gini', 'entropy']
    #2. max_features ('auto', 'log2')
    max_features = ['auto', 'log2', None]
    #3. max_depth ('none', IF A NUMBER EXISTS +2, +4, -2, -4 ELSE 4 RANDOM INTS 1->100)
    if input_model.max_depth == None:
        max_depth = [None, random.randrange(100)+1]
    else:
        max_depth = [input_model.max_depth, random.randrange(100)+1]
        max_depth = [i for i in max_depth if i > 0]
    #4. min_samples_leaf(n-1, n-2, 0,  n+1, n+2)
    min_samples_leaf = [input_model.min_samples_leaf, input_model.min_samples_leaf*1.01, input_model.min_samples_leaf*0.99]
    min_samples_leaf = [i for i in min_samples_leaf if i > 0]
    
    #5. max_leaf_nodes:('none', n+1, n+2, n-1, n-2, OR 4 random numbers)
    if ((input_model.max_leaf_nodes == None) or (input_model.max_leaf_nodes == 1)):
        max_leaf_nodes = [None, random.randrange(1000)+1]
    else:
        max_leaf_nodes = [input_model.max_leaf_nodes, random.randrange(1000)+1]
        max_leaf_nodes = [i for i in max_leaf_nodes if i > 1 ]
    n_estimators = [input_model.n_estimators, random.randrange(200)+1]
    n_estimators = [i for i in n_estimators if i > 0]
    
    
    
    for n in n_estimators:
        for ml in max_leaf_nodes:
            for ms in min_samples_leaf:
                for md in max_depth:
                    for mf in max_features:
                        for c in criterion:
                            if (len(tested_hps) == 6) and (n in tested_hps[0]) and (ml in tested_hps[1]) and (ms in tested_hps[2]) and \
                                (md in tested_hps[3]) and (mf in tested_hps[4]) and (c in tested_hps[5]): 
                                print("PASS.  We have already tested this.")
                            else:

                                test_model = RandomForestClassifier(n_estimators = n, max_leaf_nodes = ml, 
                                                                    min_samples_leaf = ms,
                                                                    max_depth = md, criterion = c, 
                                                                    max_features = mf, 
                                                                    n_jobs = -1,
                                                                    random_state=75)
                                #score = random.random()
                                score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
                                if score > best_score:
                                    best_score = score
                                    output_model = test_model
                                    print()
                                    print("NEW BEST SCORE")
                                    print("Criterion:", c, "max_features:", mf, "max_depth:", md, "min_samples_leaf:", ms,
                                          "max_leaf_nodes:", ml, "n_estimators", n, best_score)        
                                    print()
                                    print()
                                else:
                                    pass
                                    print("Criterion:", c, "max_features:", mf, "max_depth:", md, "min_samples_leaf:", ms,
                                          "max_leaf_nodes:", ml, "n_estimators", n, score)        

    new_hps = [n_estimators, max_leaf_nodes, min_samples_leaf, max_depth, max_features, criterion]
    return output_model, new_hps





def tune_ExtraTreeClassifier(input_model, input_features, input_df, input_labels, odds_input, min_ev=0, tested_hps = []):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. criterion ('gini', 'entropy')
    #2. max_features ('auto', 'sqrt', 'log2')
    #3. max_depth ('none', IF A NUMBER EXISTS +2, -2, ELSE 2 RANDOM INTS 1->100)
    #4. min_samples_leaf(n-2, 0, n+2)
    #5. max_leaf_nodes:('none', n+2, n-2, OR 2 random numbers)
    #6. n_estimators: (n, n+2, n-2)
    ###############################################################################################################    
    print()
    print()
    print("Starting New Run for RandomForestClassifier")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)        
    #1. criterion ('gini', 'entropy')
    criterion = ['gini', 'entropy']
    #2. max_features ('auto', 'log2')
    max_features = ['auto', 'log2', None]
    #3. max_depth ('none', IF A NUMBER EXISTS +2, +4, -2, -4 ELSE 4 RANDOM INTS 1->100)
    if input_model.max_depth == None:
        max_depth = [None, random.randrange(100)+1]
    else:
        max_depth = [input_model.max_depth, random.randrange(100)+1]
        max_depth = [i for i in max_depth if i > 0]
    #4. min_samples_leaf(n-1, n-2, 0,  n+1, n+2)
    min_samples_leaf = [input_model.min_samples_leaf, input_model.min_samples_leaf*1.01, input_model.min_samples_leaf*0.99]
    min_samples_leaf = [i for i in min_samples_leaf if i > 0]
    
    #5. max_leaf_nodes:('none', n+1, n+2, n-1, n-2, OR 4 random numbers)
    if ((input_model.max_leaf_nodes == None) or (input_model.max_leaf_nodes == 1)):
        max_leaf_nodes = [None, random.randrange(1000)+1]
    else:
        max_leaf_nodes = [input_model.max_leaf_nodes, random.randrange(1000)+1]
        max_leaf_nodes = [i for i in max_leaf_nodes if i > 1 ]
    
    
    
    for ml in max_leaf_nodes:
        for ms in min_samples_leaf:
            for md in max_depth:
                for mf in max_features:
                    for c in criterion:
                        if (len(tested_hps) == 6) and  (ml in tested_hps[1]) and (ms in tested_hps[2]) and \
                            (md in tested_hps[3]) and (mf in tested_hps[4]) and (c in tested_hps[5]): 
                            print("PASS.  We have already tested this.")
                        else:

                            test_model = RandomForestClassifier( 
                                                                min_samples_leaf = ms,
                                                                max_depth = md, criterion = c, 
                                                                max_features = mf, 
                                                                random_state=75)
                            #score = random.random()
                            score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
                            if score > best_score:
                                best_score = score
                                output_model = test_model
                                print()
                                print("NEW BEST SCORE")
                                print("Criterion:", c, "max_features:", mf, "max_depth:", md, "min_samples_leaf:", ms,
                                      "max_leaf_nodes:", ml,  best_score)        
                                print()
                                print()
                            else:
                                pass
                                print("Criterion:", c, "max_features:", mf, "max_depth:", md, "min_samples_leaf:", ms,
                                      "max_leaf_nodes:", ml,  score)        

    new_hps = [max_leaf_nodes, min_samples_leaf, max_depth, max_features, criterion]
    return output_model, new_hps






def tune_GradientBoostingClassifier(input_model, input_features, input_df, input_labels, odds_input, min_ev=0, tested_hps = []):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. criterion ('friedman_mse', 'mse', 'mae')
    #2. loss ('deviance', 'exponential')
    #3. n_estimators (n, n+1, n-1)
    #4. learning_rate (learning_rate, learning_rate *1.1, learning_rate*.9)
    #5. min_samples_leaf: (n, n-1, n+1)
    #6. max_depth: (n, n+1, n-1)
    #7. max_features: (None, 'auto', 'sqrt', 'log2')
    #8. max_leaf_nodes: (None, n+1, n-1, OR 2 random numbers)
    #9. tol (n, n*1.1, n*.9)
    ###############################################################################################################  
    print()
    print()
    print("Starting New Run")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)
    
    #1. criterion ('friedman_mse', 'mse', 'mae')
    criterion = ['friedman_mse']
    
    #2. loss ('deviance', 'exponential')
    loss = ['deviance']

    #3. n_estimators (n, n+1, n-1)
    n_estimators = [input_model.n_estimators, random.randrange(200)+1]
    n_estimators = [i for i in n_estimators if i > 0]    
    
    #4. learning_rate (learning_rate, learning_rate *1.1, learning_rate*.9)
    learning_rate = [input_model.learning_rate]
    
    #5. min_samples_leaf: (n, n-1, n+1)
    min_samples_leaf = [input_model.min_samples_leaf, input_model.min_samples_leaf*0.99, input_model.min_samples_leaf*1.01]
    min_samples_leaf = [i for i in min_samples_leaf if i > 0]

    #6. max_depth: (n, n+1, n-1)
    if input_model.max_depth == None:
        max_depth = [None, random.randrange(100)+1]
    else:
        max_depth = [input_model.max_depth, random.randrange(100)+1]
        max_depth = [i for i in max_depth if i > 0]
        
    #7. max_features: (None, 'auto', 'sqrt', 'log2')
    max_features = ['sqrt', 'log2', None]

    #8. max_leaf_nodes: (None, n+1, n-1, OR 2 random numbers)
    if input_model.max_leaf_nodes == None:
        max_leaf_nodes = [None, random.randrange(1000)+1]
    else:
        max_leaf_nodes = [input_model.max_leaf_nodes, random.randrange(1000)+1]
        max_leaf_nodes = [i for i in max_leaf_nodes if i > 0]

    #9. tol (n, n*1.1, n*.9)
    tol = [input_model.tol, random.random()]
            
    print(len(tol) * len(max_leaf_nodes) * len(max_features) * len(max_depth) * len(min_samples_leaf) * len(learning_rate) * len(n_estimators) * len(loss) * len(criterion))    
        
        
    for t in tol:
        for ml in max_leaf_nodes:    
            for mf in max_features:
                for md in max_depth:
                    for ms in min_samples_leaf:
                        for lr in learning_rate:
                            for n in n_estimators:
                                for l in loss:
                                    for c in criterion:
                                        test_model = GradientBoostingClassifier(n_estimators = n, 
                                                                                learning_rate = lr,
                                                                                criterion = c,
                                                                                min_samples_leaf = ms,
                                                                                max_depth = md,
                                                                                loss = l, 
                                                                                max_features = mf,
                                                                                max_leaf_nodes = ml,
                                                                                tol = t,
                                                                                random_state=75)
                                        score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
                                        
                                        if score > best_score:
                                            best_score = score
                                            output_model = test_model
                                            print()
                                            print("NEW BEST SCORE")
                                            print("Criterion:", c,
                                                  "n_estimators:", n,
                                                  "Loss:", l,
                                                  "Learning Rate:", lr,
                                                  "Min Samples/Leaf:", ms,
                                                  "Max Depth:", md,
                                                  "Max Features:", mf,
                                                  "Max Leaf Nodes:", ml,
                                                  "tol:", t,
                                                  "Best Score:", best_score)        
                                            print()
                                            print()
                                        else:
                                            pass
                                            print("Criterion:", c,
                                                  "n_estimators:", n,                          
                                                  "Loss:", l, 
                                                  "Learning Rate:", lr,
                                                  "Min Samples/Leaf:", ms,
                                                  "Max Depth:", md,
                                                  "Max Features:", mf,
                                                  "Max Leaf Nodes:", ml,
                                                  "tol:", t,
                                                  "Score:", score)        

    
    return(output_model)

def tune_GaussianNB(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. var_smoothing (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6)
    ###############################################################################################################  
    print()
    print()
    print("Starting New Run for GaussianNB")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    
    
    var_smoothing = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    
    for v in var_smoothing:
        test_model = GaussianNB(var_smoothing = v)
        score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
        if score > best_score:
            best_score = score
            output_model = test_model
            print()
            print("NEW BEST SCORE")
            print("var_smoothing:", v, 
                  "Best Score:", best_score)        
            print()
            print()
        else:
            pass
            print("var_smoothing:", v, 
                  "Score:", score)            
    return output_model


def tune_MLPClassifier(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. var_smoothing (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6)
    ###############################################################################################################  
    print()
    print()
    print("Starting New Run for MLPClassifier")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    
    
    hidden_layer_sizes = [input_model.hidden_layer_sizes, (100,), (10,5), (6,), (1,),(5,), (15,),(25,)]
    
    for h in hidden_layer_sizes:
        test_model = MLPClassifier(hidden_layer_sizes=h)
        score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
        if score > best_score:
            best_score = score
            output_model = test_model
            print()
            print("NEW BEST SCORE")
            print("hidden_layer_sizes:", h, 
                  "Best Score:", best_score)        
            print()
            print()
        else:
            pass
            print("hidden_layer_sizes:", h, 
                  "Score:", score)        
        
    
    return output_model


def tune_RadiusNeighborsClassifier(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. weights ('uniform' or 'distance')
    #2. p (1 or 2)
    ###############################################################################################################
    print()
    print()
    print("Starting New Run for RadiusNeighborsClassifier")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    

    
    weights = ['uniform', 'distance']
    p = [1,2]
    for w in weights:
        score = -10000
        for p1 in p:
            test_model = RadiusNeighborsClassifier(weights = w, p=p1, outlier_label='most_frequent')
            score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
            if score > best_score:
                best_score = score
                output_model = test_model
                
                print()
                print("NEW BEST SCORE")
                print("weights:", w, 
                      "p:", p1,
                      "Best Score:", best_score)        
                print()
                print()
            else:
                pass
                print("weights:", w, 
                      "p:", p1,
                      "Score:", score)        
            
    return(output_model)

def tune_KNeighborsClassifier(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. n_neighbors (1,2,3,4,5,6,7,8,9,10)
    #2. weights ('uniform', 'distance')
    ###############################################################################################################
    print()
    print()
    print("Starting New Run for KNeighborsClassifier")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    

    
    weights = ['uniform', 'distance']
    n_neighbors = [1,2,3,4,5,6,7,8,9,10]
    for w in weights:
        score = -10000
        for n in n_neighbors:
            test_model = KNeighborsClassifier(weights = w, n_neighbors=n)
            score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
            if score > best_score:
                best_score = score
                output_model = test_model
                
                print()
                print("NEW BEST SCORE")
                print("weights:", w, 
                      "n_neighbors:", n,
                      "Best Score:", best_score)        
                print()
                print()
            else:
                pass
                print("weights:", w, 
                      "n_neighbors:", n,
                      "Score:", score)        
                
    return(output_model)


def tune_SGDClassifier(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. penalty ('l1', 'l2')
    #2. alpha (input_model.alpha, input_model.alpha*.9, input_model.alpha*1.1)
    #3. loss ('modified_huber', 'log')
    ###############################################################################################################
    print()
    print()
    print("Starting New Run for SGDClassifier")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    

    
    penalty = ['l1', 'l2']
    alpha = [input_model.alpha, input_model.alpha*.9, input_model.alpha*1.1]
    loss = ['modified_huber', 'log']
    for l in loss:
        for p in penalty:
            score = -10000
            for a in alpha:
                test_model = SGDClassifier(loss = l, penalty = p, alpha = a, random_state=75)
                score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
                if score > best_score:
                    best_score = score
                    output_model = test_model
                    
                    print()
                    print("NEW BEST SCORE")
                    print("loss: ", l,
                          "penalty: ", p, 
                          "alpha: ", a,
                          "Best Score: ", best_score)        
                    print()
                    print()
                else:
                    pass
                    print("loss: ", l,
                          "penalty: ", p, 
                          "alpha: ", a,
                          "Score: ", score)        
                
    return(output_model)


def tune_BaggingClassifier(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. base_estimator ('GaussianNB()', 'DecisionTreeClassifier()', LogisticRegression(), RadiusNeighborsClassifier())
    #2. bootstrap(True, False)
    #3. n_estimators(input_model.n_estimators, input_model.n_estimators+3, input_model.n_estimators-3)
    ###############################################################################################################
    print()
    print()
    print("Starting New Run for BaggingClassifier")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    

    
    base_estimator = [GaussianNB(), DecisionTreeClassifier(random_state=75), LogisticRegression(random_state=75)]
    bootstrap = [True, False]
    n_estimators = [input_model.n_estimators, input_model.n_estimators+3, input_model.n_estimators-3]
    for be in base_estimator:
        for bs in bootstrap:
            for e in n_estimators:
                if e > 0:
                    test_model = BaggingClassifier(base_estimator = be, bootstrap = bs, n_estimators = e, random_state=75)
                    score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
                    if score > best_score:
                        best_score = score
                        output_model = test_model
                        
                        print()
                        print("NEW BEST SCORE")
                        print("base_estimator: ", be,
                              "bootstrap: ", bs, 
                              "n_estimators: ", e,
                              "Best Score: ", best_score)        
                        print()
                        print()
                    else:
                        pass
                        print("base_estimator: ", be,
                              "bootstrap: ", bs, 
                              "n_estimators: ", e,
                               "Score: ", score)        
                
    return(output_model)




def tune_LinearDiscriminantAnalysis(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. solver ['svd', 'lsqr', 'eigen']
    #2. tol [n, n*1.1, n*.9]
    ###############################################################################################################
    print()
    print()
    print("Starting New Run for LinearDiscriminantAnalysis")
    print(input_model)
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    

    solver = ['svd', 'lsqr']
    tol = [input_model.tol, input_model.tol*1.1, input_model.tol*0.9]

    for s in solver:
        for t in tol:
            test_model = LinearDiscriminantAnalysis(solver = s, tol = t)
            score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
            if score > best_score:
                best_score = score
                output_model = test_model
                
                print()
                print("NEW BEST SCORE")
                print("solver: ", s,
                      "tol: ", t, 
                      "Best Score: ", best_score)        
                print()
                print()
            else:
                pass
                print("solver: ", s,
                      "tol: ", t, 
                       "Score: ", score)        
    print("output model: " + str(output_model))
    return(output_model)



def tune_NuSVC(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. nu = [input_model.nu, input_model.nu*1.1, input_model.nu*0.9, random.random()]
    #2. tol = [input_model.tol, input_model.nu*1.1, input_model.nu*0.9, random.random()]

    ###############################################################################################################
    print()
    print()
    print("Starting New Run for NuSVC")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    

    
    nu = [input_model.nu, input_model.nu*1.1, input_model.nu*0.9, random.random()]    
    tol = [input_model.tol, input_model.tol*1.1, input_model.tol*0.9, random.random()]    
    for n in nu:
        for t in tol:
            test_model = NuSVC(nu=n, tol=t, probability=True, random_state=75)
            score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
            if score > best_score:
                best_score = score
                output_model = test_model
                
                print()
                print("NEW BEST SCORE")
                print("nu: ", n,
                      "tol: ", t, 
                      "Best Score: ", best_score)        
                print()
                print()
            else:
                pass
                print("nu: ", n,
                      "tol: ", t, 
                       "Score: ", score)        
                
    return(output_model)




def tune_SVC(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. C = [input_model.nu, input_model.nu*1.1, input_model.nu*0.9, random.random()*1000]
    #2. tol = [input_model.tol, input_model.nu*1.1, input_model.nu*0.9, random.random()]

    ###############################################################################################################
    print()
    print()
    print("Starting New Run for SVC")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    

    
    C = [input_model.C, input_model.C*1.1, input_model.C*0.9, random.random()*1000]    
    tol = [input_model.tol, input_model.tol*1.1, input_model.tol*0.9, random.random()]    
    for n in C:
        for t in tol:
            test_model = SVC(C=n, tol=t, probability=True, random_state=75)
            score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
            if score > best_score:
                best_score = score
                output_model = test_model
                
                print()
                print("NEW BEST SCORE")
                print("C: ", n,
                      "tol: ", t, 
                      "Best Score: ", best_score)        
                print()
                print()
            else:
                pass
                print("C: ", n,
                      "tol: ", t, 
                       "Score: ", score)        
                
    return(output_model)


"""
def tune_DPGMM(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    ###############################################################################################################
    #Parameters we are going to fine-tune:
    #1. covariance_type = ['spherical', 'tied', 'diag', 'full']
    #2. tol = [input_model.tol, input_model.tol*1.1, input_model.tol*0.9, random.random()]

    ###############################################################################################################
    print()
    print()
    print("Starting New Run for DPGMM")
    print()
    print()
    output_model = input_model
    best_score = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=min_ev)
    print("Previous Best Score:", best_score)    

    
    covariance_type = ['spherical', 'tied', 'diag', 'full']
    tol = [input_model.tol, input_model.tol*1.1, input_model.tol*0.9, random.random()]    
    for n in covariance_type:
        for t in tol:
            test_model = DPGMM(covariance_type = n, tol = t)
            score = get_ev(input_df, test_model, input_features, input_labels, odds_input, min_ev=min_ev)
            if score > best_score:
                best_score = score
                output_model = test_model
                
                print()
                print("NEW BEST SCORE")
                print("Covariance Type: ", n,
                      "tol: ", t, 
                      "Best Score: ", best_score)        
                print()
                print()
            else:
                pass
                print("Covariance Type: ", n,
                      "tol: ", t, 
                       "Score: ", score)        
                
    return(output_model)
"""



#Let's just hold for now
def tune_hyperparameters_mov(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    return(input_model)




def tune_hyperparameters(input_model, input_features, input_df, input_labels, odds_input, min_ev=0):
    best_model = input_model
    keep_going = True

    """
    if isinstance(input_model, DPGMM):
        while(keep_going):
            pos_model = (tune_DPGMM(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model
    """

    if isinstance(input_model, SVC):
        while(keep_going):
            pos_model = (tune_SVC(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model



    elif isinstance(input_model, NuSVC):
        while(keep_going):
            pos_model = (tune_NuSVC(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model



    elif isinstance(input_model, LinearDiscriminantAnalysis):
        while(keep_going):
            pos_model = (tune_LinearDiscriminantAnalysis(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
                print("pos model: " + str(pos_model))
                print(" output_model: " + str(output_model))
            else:
                best_model = pos_model


    elif isinstance(input_model, BaggingClassifier):
        while(keep_going):
            pos_model = (tune_BaggingClassifier(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model
    
    
    elif isinstance(input_model, SGDClassifier):
        while(keep_going):
            pos_model = (tune_SGDClassifier(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model

    
    elif isinstance(input_model, LogisticRegression):
        while(keep_going):
            pos_model = (tune_LogisticRegression(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model
                
    elif isinstance(input_model, DecisionTreeClassifier):
        while(keep_going):
            pos_model = (tune_DecisionTreeClassifier(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model            
                
    elif isinstance(input_model, RandomForestClassifier):
        tested_hps = []
        while(keep_going):
            pos_model, tested_hps = (tune_RandomForestClassifier(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev, tested_hps=tested_hps))
            print(tested_hps)
            print(len(tested_hps))
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model    

    elif isinstance(input_model, ExtraTreeClassifier):
        tested_hps = []
        while(keep_going):
            pos_model, tested_hps = (tune_ExtraTreeClassifier(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev, tested_hps=tested_hps))
            print(tested_hps)
            print(len(tested_hps))
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model    



                                
    elif isinstance(input_model, GradientBoostingClassifier):
        print("HI")
        while(keep_going):
            pos_model = (tune_GradientBoostingClassifier(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model                    
                
    elif isinstance(input_model, MLPClassifier):
        print("MLPClassifier")
        while(keep_going):
            pos_model = (tune_MLPClassifier(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model                    
                
                
    elif isinstance(input_model, GaussianNB):
        while(keep_going):
            pos_model = (tune_GaussianNB(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model    

    elif isinstance(input_model, RadiusNeighborsClassifier):
        while(keep_going):
            pos_model = (tune_RadiusNeighborsClassifier(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model    
                
    elif isinstance(input_model, KNeighborsClassifier):
        while(keep_going):
            pos_model = (tune_KNeighborsClassifier(best_model, input_features, input_df, input_labels, odds_input, min_ev=min_ev))
            if str(pos_model) == str(best_model):  #Direct comparisons don't seem to work....
                keep_going = False
                output_model = best_model
            else:
                best_model = pos_model    
                
                
                
                
    else:
        output_model = input_model
    print("Real output model: " + str(output_model))
    return(output_model)                


#Let's just hold....
def tune_ev_mov(input_model, input_features, input_df, input_labels, odds_input, verbose=False):
    return(0)



def tune_ev(input_model, input_features, input_df, input_labels, odds_input, verbose=False):
    best_ev = -100000
    best_pos = -1
    for temp_ev in range(50):
        pos_ev = get_ev(input_df, input_model, input_features, input_labels, odds_input, min_ev=temp_ev, verbose=verbose,
                       get_total=True)
        print(temp_ev, pos_ev)
        if pos_ev > best_ev:
            best_ev = pos_ev
            best_pos = temp_ev
    return best_pos
    
def remove_to_improve_mov(cur_features, m, df, labels, odds, scale=False, min_ev = 0):
    number_of_features = len(cur_features)
    df_sel = df[cur_features]
    df_sel = df_sel.dropna()
    df_sel = pd.get_dummies(df_sel)
    labels_sel = labels[labels.index.isin(df_sel.index)]
    odds_sel = odds[odds.index.isin(df_sel.index)]        
    orig_score = custom_cv_eval_mov(df_sel, m, labels_sel, odds_sel, get_total=True, min_ev = min_ev)    
    best_features = cur_features
    best_score = orig_score
    print(f"The original score is {orig_score}")
    if number_of_features == 0:
        return []
    if number_of_features == 1:
        return cur_features
    
    for z in range(number_of_features):
        temp_feature = df.columns[z]
        temp_features = cur_features.copy()
        #Remove a feature
        del temp_features[z]
        df_sel = df[temp_features]
        df_sel = df_sel.dropna()
        df_sel = pd.get_dummies(df_sel)
        labels_sel = labels[labels.index.isin(df_sel.index)]
        odds_sel = odds[odds.index.isin(df_sel.index)]        
        temp_score = custom_cv_eval_mov(df_sel, m, labels_sel, odds_sel, get_total=True, min_ev = min_ev)
        if temp_score > best_score:
            best_features = temp_features
            best_score = temp_score
            print(f"NEW BEST FEATURE SET WITH: " + temp_feature + " REMOVED " + str(best_score))
        
        #Get a score
    if best_features != cur_features:
        return remove_to_improve_mov(best_features, m, df, labels, odds, scale, min_ev)
    else:
        return best_features    
    
def remove_to_improve(cur_features, m, df, labels, odds, scale=False, min_ev = 0):
    #If the list is empty we can just return it without doing anything
    number_of_features = len(cur_features)
    df_sel = df[cur_features]
    df_sel = df_sel.dropna()
    df_sel = pd.get_dummies(df_sel)
    labels_sel = labels[labels.index.isin(df_sel.index)]
    odds_sel = odds[odds.index.isin(df_sel.index)]        
    orig_score = custom_cv_eval(df_sel, m, labels_sel, odds_sel, get_total=True, min_ev = min_ev)
    #print(orig_score)
    best_features = cur_features
    best_score = orig_score
    print(f"The original score is {orig_score}")
    if number_of_features == 0:
        return []
    if number_of_features == 1:
        return cur_features
    
    for z in range(number_of_features):
        temp_features = cur_features.copy()
        #Remove a feature
        del temp_features[z]
        df_sel = df[temp_features]
        df_sel = df_sel.dropna()
        df_sel = pd.get_dummies(df_sel)
        labels_sel = labels[labels.index.isin(df_sel.index)]
        odds_sel = odds[odds.index.isin(df_sel.index)]        
        temp_score = custom_cv_eval(df_sel, m, labels_sel, odds_sel, get_total=True, min_ev = min_ev)
        if temp_score > best_score:
            best_features = temp_features
            best_score = temp_score
            print(f"NEW BEST FEATURE SET")
            print(best_features)
            print(best_score)
        else:
            print("Score: ", temp_score)
        
        #Get a score
    if best_features != cur_features:
        return remove_to_improve(best_features, m, df, labels, odds, scale, min_ev)
    else:
        return best_features    
    
    
    
    
    
def evaluate_model(input_model, input_features, input_ev, train_df, train_labels, train_odds, test_df, test_labels,
                  test_odds, verbose=True):
    model_score = 0
    
    df_train = train_df[input_features].copy()
    df_test = test_df[input_features].copy()
    df_train = df_train.dropna()
    df_test = df_test.dropna()
        
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)
    df_train, df_test = df_train.align(df_test, join='left', axis=1)    #Ensures both sets are dummified the same
    df_test = df_test.fillna(0)

    #LOOK AT get_ev and prepare the labels and odds
    
    labels_train = train_labels[train_labels.index.isin(df_train.index)]
    odds_train = train_odds[train_odds.index.isin(df_train.index)] 
    labels_test = test_labels[test_labels.index.isin(df_test.index)]
    odds_test = test_odds[test_odds.index.isin(df_test.index)] 
    
    
    
    display(df_train.shape)
    display(labels_train.shape)
    display(odds_train.shape)
    display(df_test.shape)
    display(labels_test.shape)
    display(odds_test.shape)
    
    
    
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(df_train)
    
    input_model.fit(scaled_train, labels_train)

    scaled_test = scaler.transform(df_test)
    
    probs = input_model.predict_proba(scaled_test)

    
    odds_test = np.array(odds_test)    
    
    
    prepped_test = list(zip(odds_test[:, -2], odds_test[:, -1], probs[:, 0], probs[:, 1], labels_test))
    ev_prepped_df = pd.DataFrame(prepped_test, columns=['t1_odds', 't2_odds', 't1_prob', 't2_prob', 'winner'])
    
    display(ev_prepped_df)
    
    #display(df_test)
    #display(df_test)
    model_score = get_ev_from_df(ev_prepped_df, print_stats = True, min_ev = input_ev, get_total=True)
    
    return(model_score)    


def evaluate_model_mov(input_model, input_features, input_ev, train_df, train_labels, train_odds, test_df, test_labels,
                  test_odds, label_list, verbose=True):
    model_score = 0
    
    df_train = train_df[input_features].copy()
    df_test = test_df[input_features].copy()
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)
    df_train, df_test = df_train.align(df_test, join='left', axis=1)    #Ensures both sets are dummified the same
    df_test = df_test.fillna(0)

    labels_train = train_labels[train_labels.index.isin(df_train.index)]
    odds_train = train_odds[train_odds.index.isin(df_train.index)] 
    labels_test = test_labels[test_labels.index.isin(df_test.index)]
    odds_test = test_odds[test_odds.index.isin(df_test.index)] 
    
    odds_train = odds_train.dropna()
    odds_test = odds_test.dropna()
    
    df_train = df_train[df_train.index.isin(odds_train.index)]
    df_test = df_test[df_test.index.isin(odds_test.index)]
    
    labels_train = labels_train[labels_train.index.isin(odds_train.index)]
    labels_test = labels_test[labels_test.index.isin(odds_test.index)]    
    
    

    if verbose:
        display(df_train.shape)
        display(labels_train.shape)
        display(odds_train.shape)
        display(df_test.shape)
        display(labels_test.shape)
        display(odds_test.shape)

    print(labels_train)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(df_train)
    
    input_model.fit(scaled_train, labels_train)

    scaled_test = scaler.transform(df_test)
    
    probs = input_model.predict_proba(scaled_test)
    model_score = get_ev_from_df_mov(odds_test, probs, labels_test, label_list, input_model.classes_, print_stats = True, min_ev = input_ev, get_total=True)


    return(model_score)