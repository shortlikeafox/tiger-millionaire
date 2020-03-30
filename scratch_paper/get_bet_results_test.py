# -*- coding: utf-8 -*-

import header as h
import pandas as pd
import numpy as np
import fs_definitions as fsd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
import seaborn as sns
import matplotlib.pyplot as plt

fs = 'c2'

df = h.create_fight_df('../data/ufc-master.csv')
old_df = df
df = h.get_test_probs(df,fs,75,0.1)

df_results = h.get_bet_results(df)

print(df_results)



results = (h.get_bet_results_multiple(old_df, 10, fs))

print(results.shape)

print(results)

results.to_csv('results.csv')