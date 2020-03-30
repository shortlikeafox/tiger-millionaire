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
    
    column_names = "" #The columns that need to be encoded
    
    #We will use this to make copies fo the evaluation columns....
    df_copy = df[['R_ev', 'B_ev']].copy()
    
    
    if fs =='c1':
        #The list of features that are included in this model
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
                 'B_avg_SIG_STR_pct', 'R_avg_SIG_STR_pct', 'R_fighter', 
                 'B_fighter', 'location', 'B_Stance', 'R_Stance', 
                 'weight_class', 'lose_streak_dif', 'win_streak_dif',
                 'longest_win_streak_dif', 'win_dif', 'loss_dif', 
                 'total_round_dif', 'total_title_bout_dif', 'ko_dif', 
                 'sub_dif', 'height_dif', 'reach_dif', 'age_dif', 
                 'sig_str_dif', 'country', 'Winner', 'label']]
        
        #Used so 'get_dummies' uses the proper columns
        column_names = ['R_fighter', 'B_fighter', 'location', 'B_Stance',
                        'R_Stance', 'weight_class', 'country']
    
    
    if fs == 'c2':
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
                 'B_avg_SIG_STR_pct', 'R_avg_SIG_STR_pct', 'R_fighter',
                 'B_fighter', 'location', 'B_Stance', 'R_Stance', 
                 'weight_class', 'gender', 'country', 'Winner', 'label']]
        #Used so 'get_dummies' uses the proper columns
        column_names = ['R_fighter', 'B_fighter', 'location', 'B_Stance',
                        'R_Stance', 'weight_class', 'country', 'gender']
        
    
    if fs == 'c3':
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
                 'B_avg_SIG_STR_pct', 'R_avg_SIG_STR_pct', 'location', 
                 'B_Stance', 'R_Stance', 'weight_class', 'Winner', 'label']]
        #Used so 'get_dummies' uses the proper columns
        column_names = ['location', 'B_Stance', 'R_Stance', 'weight_class']
    
    
    if fs == 'c4':
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
                 'B_avg_SIG_STR_pct', 'R_avg_SIG_STR_pct', 'B_Stance', 
                 'R_Stance', 'weight_class', 'R_odds', 'B_odds', 'Winner', 
                 'label']]
        #Used so 'get_dummies' uses the proper columns
        column_names = ['B_Stance', 'R_Stance', 'weight_class']
        
    
    if fs == 'c5':
        df = df[['constant_1', 'Winner', 'label']]

    if fs == 'c6':
        df = df[['B_avg_SIG_STR_pct', 'B_ev', 'R_total_rounds_fought',
                 'R_win_by_Decision_Split', 'R_win_by_Decision_Majority',
                 'R_current_win_streak', 'gender', 'Winner', 'label']]
        column_names = ['gender']

    if fs == 'c1d':
        df = df[['B_age', 'R_age', 'loss_dif', 'age_dif', 'B_odds', 'R_odds',
                 'gender', 'Winner', 'label']]    

        column_names = ['gender']

    if fs =='c2d':
        #The list of features that are included in this model
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
                 'B_avg_SIG_STR_pct', 'R_avg_SIG_STR_pct', 'R_fighter', 
                 'B_fighter', 'location', 'B_Stance', 'R_Stance', 
                 'weight_class', 'lose_streak_dif', 'win_streak_dif',
                 'longest_win_streak_dif', 'win_dif', 'loss_dif', 
                 'total_round_dif', 'total_title_bout_dif', 'ko_dif', 
                 'sub_dif', 'height_dif', 'reach_dif', 'age_dif', 
                 'sig_str_dif',  'Winner', 'label']]        
        #Used so 'get_dummies' uses the proper columns
        column_names = ['R_fighter', 'B_fighter', 'location', 'B_Stance',
                        'R_Stance', 'weight_class']
        
        
    if fs == 'c3d':
        #The list of features that are included in this model
        df = df[['B_age', 'R_age', 'loss_dif', 'reach_dif', 'age_dif', 
                 'B_odds', 'R_odds', 'gender', 'R_ev', 'Winner', 'label']]        
        column_names = ['gender']

    if fs =='c4d':
        #The list of features that are included in this model
        df = df[['no_of_rounds','B_current_lose_streak', 
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
                 'B_avg_SIG_STR_pct', 'R_avg_SIG_STR_pct','location', 
                 'B_Stance', 'R_Stance', 'weight_class', 'lose_streak_dif', 
                 'win_streak_dif', 'longest_win_streak_dif', 'win_dif', 
                 'loss_dif', 'total_round_dif', 'total_title_bout_dif', 
                 'ko_dif', 'sub_dif', 'height_dif', 'reach_dif', 'age_dif', 
                 'sig_str_dif',  'Winner', 'label']]
        
        #Used so 'get_dummies' uses the proper columns
        column_names = ['location', 'B_Stance', 'R_Stance', 'weight_class']

    if fs == 'c5d':
        df = df[['B_avg_SIG_STR_pct', 'B_ev', 'R_total_rounds_fought',
                      'R_win_by_Decision_Split', 'R_win_by_Decision_Majority',
                      'Winner', 'label']]



    #The test model is where we can test different features
    if fs =='test':
        df = df[['gender', 'Winner', 'label']]
        column_names = ['gender']
    
    #These are going to be used for evauluation purposes
    df = df.assign(B_ev_final = df_copy['B_ev'])
    df = df.assign(R_ev_final = df_copy['R_ev'])
    
    
    #Remove all rows with null values
    df = df.dropna(how='any', axis=0)
    
    #Convert dummy variables
    if (column_names):
        df = pd.get_dummies(df, columns=column_names )

    
    return df