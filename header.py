# -*- coding: utf-8 -*-

#Contains Global Variables and Functions for Tiger Millionaire
import pandas as pd
import math
import fs_definitions

print("Header imported")
#Constants that we will need

MASTER_CSV_FILE = "data/ufc-master.csv" #The master csv
EVENT_CSV_FILE = "data/event.csv" #An alternate csv, one use if for 
                                    #upcoming events
CLASSIFICATION_RESULTS_CSV = ""  #The "grades" of each model
FEATURES = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c1d', 'c2d', 'c3d', 
            'c4d', 'c5d']  #Codes for current models


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




    
    return df
    