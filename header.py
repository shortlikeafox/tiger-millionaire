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