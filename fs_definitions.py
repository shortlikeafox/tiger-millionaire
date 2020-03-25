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
    df: The filtered dataframe

    """
    
    
    
    return df