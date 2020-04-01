# -*- coding: utf-8 -*-

#We will test stuff here...
import header as h
import pandas as pd
import numpy as np
import fs_definitions as fsd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


pd.set_option('display.max_rows', 500) #Used for debugging

#df = h.create_fight_df(h.MASTER_CSV_FILE)
#print(df.head)
#print(df.describe())
#print(len(df))
#print(df.head)
#print(df.dtypes)
#OK... Now we need to create some dataframes....
df = h.create_master_df()
#print(df.head)
#print(len(df))
#adding a comment








temp_df = fsd.create_prepped_df('c1', df)

#print(temp_df)

X = temp_df.iloc[:, :-2].values
y = temp_df.iloc[:, -1:].values

#print(X)

#print(y)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state = 85)

y_test = np.ravel(y_test)
y_train = np.ravel(y_train)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

rf = RandomForestClassifier(criterion='entropy',
                            max_leaf_nodes=34,
                            max_depth=5,
                            n_estimators=25
                            )

rf.fit(X_train, y_train)

predictions = rf.predict(X_test)

errors = abs(predictions - y_test)
total_errors = (sum(errors))
print(1 -total_errors / len(y_test))

#print(X_train.dtypes)





df.to_csv('testfile.csv')
