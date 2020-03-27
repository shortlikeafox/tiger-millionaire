# -*- coding: utf-8 -*-

#We will test stuff here...
import header as h
import pandas as pd
import numpy as np
import fs_definitions as fsd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

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

y = temp_df['label']

temp_prepped_df = temp_df.drop(['Winner', 'label'], axis=1)

X = temp_prepped_df.values

#y = temp_df.iloc[:, -1:].values



print(temp_df['label'].value_counts())

sns.countplot(x='Winner',data=temp_df, palette='hls')

%matplotlib inline
pd.crosstab(temp_df.title_bout,temp_df.Winner).plot(kind='bar')
plt.title("Title bout vs. winner")



#print(X)

#print(y)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state = 75)

y_test = np.ravel(y_test)
y_train = np.ravel(y_train)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

lr = LogisticRegression(max_iter=10000)

lr.fit(X_train, y_train)

predictions = lr.predict(X_test)

errors = abs(predictions - y_test)
total_errors = (sum(errors))
print(1 -total_errors / len(y_test))



print(f"In the test set 0 wins {len(y_test) - sum(y_test)}")
print(f"In the test set 1 wins {sum(y_test)}")
print(f"I predict 0 to win {len(predictions) - sum(predictions)}")
print(f"I predict 1 to win {sum(predictions)}")
#print(X_train.dtypes)





#df.to_csv('testfile.csv')
