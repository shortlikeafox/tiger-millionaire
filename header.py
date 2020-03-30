# -*- coding: utf-8 -*-

#Contains Global Variables and Functions for Tiger Millionaire
import pandas as pd
import math
import numpy as np
#from pyspark.ml.feature import oneHotEncoder
import fs_definitions as fsd
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from random import randint




print("Header imported")
#Constants that we will need


MASTER_CSV_FILE = "data/ufc-master.csv" #The master csv
EVENT_CSV_FILE = "data/event.csv" #An alternate csv, one use if for 
                                    #upcoming events
CLASSIFICATION_RESULTS_CSV = ""  #The "grades" of each model
FEATURES = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c1d', 'c2d', 'c3d', 
            'c4d', 'c5d']  #Codes for current models


def encode_and_bind(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Works to one-hot-encode a feature without deleting any
    columns
    Parameters
    ----------
    df : pd.DataFrame
        The original dataframe
    feature : str
        The feature to be run through the one-hot-encoder

    Returns
    -------
    The dataframe with the new features encoded

    """
    dummies = pd.get_dummies(df[[feature]])
    res = pd.concat([df, dummies], axis=1)
    return res


def create_fight_df(csv_location: str) -> pd.DataFrame:
    """
    

    Parameters
    ----------
    csv_location : str
        The location of the csv to be used.  Will most likely either be
        MASTER_CSV_FILE or EVENT_CSV_FILE

    Returns
    -------
    df cleaned of bad data and nulls

    """
    
    df = pd.read_csv(csv_location)  #read the csv into a DataFrame
    
    #Filter out all fights that end in a draw
    df = df[df['Winner'] != 'Draw']
    

    #Certain features cannot be null for the program to work
    #correctly
    df = df[df['R_fighter'].notna()]
    df = df[df['B_fighter'].notna()]
    df = df[df['location'].notna()]
    df = df[df['B_Stance'].notna()]
    df = df[df['R_Stance'].notna()]
    df = df[df['Winner'].notna()]
    df = df[df['weight_class'].notna()]
    df = df[df['R_odds'].notna()]
    df = df[df['B_odds'].notna()]


    #Well.... we need to create a label column


    #We need to encode the winner as the label...
    df["Winner"] = df["Winner"].astype('category')
    df["label"] = df["Winner"].cat.codes
    
    df['date'] = pd.to_datetime(df['date'])

        
    return df

def create_master_df():
    """
    Returns the DataFrame associated with MASTER_CSV_FILE

    Returns
    -------
    the DataFrame associated with MASTER_CSV_FILE

    """
    return create_fight_df(MASTER_CSV_FILE)


def split_event(event_date: str, df: pd.DataFrame) -> [pd.DataFrame,\
    pd.DataFrame]:
    """
    Splits a fight DataFrame.  Removes all fights from a current date and
    places them in a separate DataFrame

    Parameters
    ----------
    event_date : str
        The date to be removed and placed into its own DataFrame
    df : pd.DataFrame
        The master DataFrame we wish to use

    Returns
    -------
    A tuple of the two dataframes (master_df, event_df)

    """
    event_df = df[df['date'] == event_date]
    master_df = df[df['date'] != event_date]
    
    return (master_df, event_df)

def get_classifier(fs: str):
    """
    Returns the proper classifier with hyperparameters

    Parameters
    ----------
    fs : str
        The feature set

    Returns
    -------
    The classifier

    """
    if fs =='c1':
        the_classifier = LogisticRegression(solver='liblinear', 
                                            max_iter = 10000)
        
    if fs =='c2':
        the_classifier = LogisticRegression(solver='liblinear',
                                            max_iter=10000)
    
    if fs=='c3':
        the_classifier = LogisticRegression(solver='liblinear',
                                            max_iter=10000,
                                            fit_intercept=False)
    
    if fs=='c4':
        the_classifier=RandomForestClassifier(max_leaf_nodes=64,
                                              max_depth=5,
                                              min_samples_split=6)
        
    if fs=='c5':
        the_classifier=LogisticRegression()
        
    if fs=='c6':
        the_classifier=GradientBoostingClassifier(max_leaf_nodes=28,
                                                  max_depth=7, 
                                                  n_estimators=20)
        
    if fs=='c1d':
        the_classifier=LogisticRegression(tol=1.0e-3, solver='liblinear')
                                                  
    if fs=='c2d':
        the_classifier=LogisticRegression(max_iter=10000, solver='liblinear')
        
    if fs=='c3d':
        the_classifier=RandomForestClassifier(max_leaf_nodes=40,
                                              max_depth=5,
                                              min_samples_split=6)
        
    if fs=='c4d':
        the_classifier=GradientBoostingClassifier()
        
    if fs=='c5d':
        the_classifier=RandomForestClassifier(criterion='entropy', 
                                              max_leaf_nodes=34, 
                                              max_depth=5,
                                              n_estimators=25)
        
    return(the_classifier)
    
def get_test_probs(df: pd.DataFrame, fs: str, seed: int
                   , split: float)-> pd.DataFrame:
    """
    Returns a DataFrame that includes classification results    

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame we will be analyzing (Probably the master dataframe)
        
    fs: feature set
        The feature set

    seed : int
        A seed for the split

    split : float
        the split amount

    Returns
    -------
    Returns a dataframe that has been put through a classifier function.
    The dataframe will include the probabilities and predictions.

    """
    print("\n\n******************************************************\n\n")
    print("STARTING get_test_probs")


    print("\n\n******************************************************")
    
    """
    1. Create Prepped DF
    2. Create alternate DF that only includes odds
    3. Remove certain characteristics from Prepped DF
    4. Split both DFs
    4.5: Create classifier
    5. Run classification
    6. Append odds and probs and preds and winner and label to df
    7. Return it.
    """
    
    #1. create prepped df
    prepped_df = fsd.create_prepped_df(fs, df)

    #2. Create alternate DF that only includes odds
    ev_df = prepped_df[['B_ev_final', 'R_ev_final', 'Winner', 'label']]

    #3. Remove certain characteristics from Prepped DF
    y = prepped_df['label']
    prepped_df = prepped_df.drop(['Winner', 'label', 'R_ev_final',
                                'B_ev_final'], axis=1)
    X = prepped_df.values
    X_ev = ev_df.values
    
    #4. Split both DFs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split,
                                                    random_state = seed)

    X_train_ev, X_test_ev, y_train_ev, y_test_ev = train_test_split(X_ev, y, 
                                                    test_size=split,
                                                    random_state = seed)

    #4.5 Create classifier
    classifier = get_classifier(fs)
    
    #5. Run classification
        
    classifier.fit(X_train, y_train)

    probs = classifier.predict_proba(X_test)
    preds = classifier.predict(X_test)
    #p_and_p = np.append(probs, preds, 1)

    #6. Append odds and probs and preds and winner and label to df
    preds = preds.reshape((len(preds),1))
    X_test = np.append(X_test, probs,1)
    X_test = np.append(X_test, preds,1)
    X_test = np.append(X_test, X_test_ev,1)
    colNamesArr = prepped_df.columns.values
    colNamesArr = np.append(colNamesArr, ['B_prob', 'R_prob', 'preds', 
                                          'B_ev_final','R_ev_final', 'Winner',
                                          'label'])
    print(X_test.shape)
    print(colNamesArr.shape)
    final_df = pd.DataFrame(X_test)
    final_df.columns = colNamesArr    
    
    
    return final_df

def get_fighter_ev(prob: float, ev: float) -> float:
    """
    

    Parameters
    ----------
    prob : float
        The probability of a fighter winning
    ev : float
        The ev if this fighter wins

    Returns
    -------
    float
        Returns the EV of a bet given a hypothetical $100 bet

    """
    return_ev = (prob * ev) - ((1-prob) * 100)
    return(return_ev)

def get_bet_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes as input the prepared dataframe with classification information from
    get_test_probs.  Returns a DataFrame with information about how the
    bets did

    Parameters
    ----------
    df : pd.DataFrame
        Prepared input dataframe

    Returns
    -------
    pd.DataFrame
        DataFrame with information about how the bets did

    """
    num_fights = 0
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
    profit_per_fight = 0
    
    #Iterate through the dataframe....
    for index, row in df.iterrows():
        #We need to deal with the above variables one-by-one
        num_fights = num_fights+1
        
        R_bet_ev = get_fighter_ev(row['R_prob'], row['R_ev_final'])
        B_bet_ev = get_fighter_ev(row['B_prob'], row['B_ev_final'])
        if (R_bet_ev > 0 or B_bet_ev > 0):
            num_bets = num_bets+1
            
        if R_bet_ev > 0:
            if row['Winner'] == 'Red':
                num_wins += 1
                profit = profit + row['R_ev_final']
            elif row['Winner'] == 'Blue':
                num_losses += 1
                profit = profit - 100
            if (row['R_ev_final'] > row['B_ev_final']):
                num_under += 1
                if row['Winner'] == 'Red':
                    num_under_wins += 1
                elif row['Winner'] == 'Blue':
                    num_under_losses += 1
            elif (row['R_ev_final'] < row['B_ev_final']):
                num_fav += 1
                if row['Winner'] == 'Red':
                    num_fav_wins += 1
                elif row['Winner'] == 'Blue':
                    num_fav_losses += 1
            else:
                num_even += 1
                if row['Winner'] == 'Red':
                    num_even_wins += 1
                elif row['Winner'] == 'Blue':
                    num_even_losses += 1
                
                
        if B_bet_ev > 0:
            if row['Winner'] == 'Blue':
                num_wins += 1
                profit = profit + row['B_ev_final']
            elif row['Winner'] == 'Red':
                num_losses += 1
                profit = profit - 100
            if (row['B_ev_final'] > row['R_ev_final']):
                num_under += 1
                if row['Winner'] == 'Blue':
                    num_under_wins += 1
                elif row['Winner'] == 'Red':
                    num_under_losses += 1
            elif (row['B_ev_final'] < row['R_ev_final']):
                num_fav += 1
                if row['Winner'] == 'Blue':
                    num_fav_wins += 1
                elif row['Winner'] == 'Red':
                    num_fav_losses += 1
            else:
                num_even += 1
                if row['Winner'] == 'Blue':
                    num_even_wins += 1
                elif row['Winner'] == 'Red':
                    num_even_losses += 1
    
    profit_per_bet = profit / num_bets
    profit_per_fight = profit / num_fights
    
    print(f"""
          Number of fights: {num_fights}
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
          Profit per fight: {profit_per_fight}
          
          """)
    
    return_df = pd.DataFrame(data=([[num_fights, num_bets, num_wins,
                                       num_losses, num_under,
                                       num_under_losses, num_under_wins,
                                       num_even, num_even_losses,
                                       num_even_wins, num_fav, 
                                       num_fav_wins, num_fav_losses, 
                                       profit, profit_per_bet,
                                       profit_per_fight]]))
    print(return_df.shape)
    return_df.columns= ['num_fights', 'num_bets', 'num_wins',
                             'num_losses', 'num_under', 'num_under_losses',
                             'num_under_wins', 'num_even', 'num_even_losses',
                             'num_even_wins', 'num_fav', 'num_fav_wins', 
                             'num_fav_losses', 'profit', 'profit_per_bet',
                             'profit_per_fight']

    return(return_df)

                              

def get_bet_results_multiple(df: pd.DataFrame, n: int
                             , fs: str) -> pd.DataFrame:
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        The master dataframe
    n : int
        How many times to run each model

    Returns
    -------
    DataFrame with information about how the bets did.  It runs
    get_bet_results multiple times and stores the results in a DataFrame

    """
    all_results_df = None
    for _ in range(n):
        itr_df = get_test_probs(df,fs,randint(1, 1000000),0.1)
        single_result_df = get_bet_results(itr_df)

        if (all_results_df is not None):
            all_results_df = all_results_df.append(single_result_df)
        else:
            all_results_df = single_result_df
    
    return(all_results_df)