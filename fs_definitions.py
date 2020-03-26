# -*- coding: utf-8 -*-
import pandas as pd


print("fs_definitions imported")



def create_prepped_df(fs: str, df: pd.DataFrame ):
    """Method that returns a dataframe that fits a feature set definition.
    
    
    Parameters
    ----------
    fs : str
        The code for the feature set. i.e. 'c1', c3d', etc...
    df : pd.DataFrame
        The dataframe we are filtering

    Returns
    -------
    df: The filtered dataframe.  This INCLUDES the label column
    

    """
    
    if fs =='c1':
        df = df[['title_bout', 'no_of_rounds', 'B_current_lose_streak',
                 'B_current_win_streak', 'B_longest_win_streak', 'B_losses', 
                 'B_total_rounds_fought', 'B_total_title_bouts', 
                 'B_win_by_Decision_Majority', 'B_win_by_Decision_Split',
                 'B_win_by_Decision_Unanimous', 'B_win_by_KO/TKO',
                 'B_win_by_Submission', 'B_win_by_TKO_Doctor_Stoppage',
                 'B_wins', 'B_Height_cms', 'B_Reach_cms', 
                 'R_current_lose_streak', 'R_current_win_streak', 
                 'R_longest_win_streak', 'R_losses', 
                 'R_total_rounds_fought', 'R_total_title_bouts', 
                 'R_win_by_Decision_Majority', 'R_win_by_Decision_Split',
                 'R_win_by_Decision_Unanimous', 'R_win_by_KO/TKO',
                 'R_win_by_Submission', 'R_win_by_TKO_Doctor_Stoppage',
                 'R_wins', 'R_Height_cms', 'R_Reach_cms', 'B_age', 'R_age',
                 'B_avg_SIG_STR_pct', 'R_avg_SIG_STR_pct',
                 'Winner', 'label']]
    
    
    if fs == 'c5':
        df = df[['constant_1', 'Winner', 'label']]
    
    
    if fs == 'c5d':
        df = df[['B_avg_SIG_STR_pct', 'B_ev', 'R_total_rounds_fought',
                      'R_win_by_Decision_Split', 'R_win_by_Decision_Majority',
                      'Winner', 'label']]
    
    
    
    
    #Remove all rows with null values
    df = df.dropna(how='any', axis=0)
    
    
    return df