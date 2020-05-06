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
df = h.create_fight_df('../data/ufc-master.csv')
print(df.dtypes)
#print(df.head)
#print(len(df))
#adding a comment


fs = 'c3'

temp_df = fsd.create_prepped_df(fs, df)

y = temp_df['label']

temp_prepped_df = temp_df.drop(['Winner', 'label'], axis=1)

X = temp_prepped_df.values

#y = temp_df.iloc[:, -1:].values



print(temp_df['label'].value_counts())
"""
sns.countplot(x='Winner',data=temp_df, palette='hls')

%matplotlib inline
pd.crosstab(temp_df.title_bout,temp_df.Winner).plot(kind='bar')
plt.title("Title bout vs. winner")

"""

#print(X)

#print(y)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state = 85)

y_test = np.ravel(y_test)
y_train = np.ravel(y_train)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

lr = h.get_classifier(fs)


print(f"The type of classifier is {lr.get_params}")

print(y_train)

lr.fit(X_train, y_train)

predictions = lr.predict(X_test)

errors = abs(predictions - y_test)
total_errors = (sum(errors))

print(f"The number of errors is {sum(errors)}")
print(f"That means I am getting {len(predictions) - sum(errors)} right\
      out of {len(predictions)} for a {(len(predictions) - sum(errors)) / (len(predictions))}")


print(f"In the test set 0 wins {len(y_test) - sum(y_test)}")
print(f"In the test set 1 wins {sum(y_test)}")
print(f"I predict 0 to win {len(predictions) - sum(predictions)}")
print(f"I predict 1 to win {sum(predictions)}")
#print(X_train.dtypes)


"""
df = h.create_master_df()
print(df.dtypes)


#print(df.head)
print(len(df))

date_df = df['date']
print(date_df.dtypes)


master_df, event_df = h.split_event("02-29-2020", df)

print(len(master_df))
print(len(event_df))

print(event_df['Winner'])
print(event_df['label'])
"""

red_favorite = df[df['R_ev'] < df['B_ev']]
blue_favorite = df[df['R_ev'] > df['B_ev']]


print(f"Red is the favorite {len(red_favorite)} times.  Blue is the",
      f"favorite {len(blue_favorite)} times")
print(len(red_favorite) + len(blue_favorite))
print(len(df))
even_amount = len(df) - (len(red_favorite) + len(blue_favorite))
objects = ('Red', 'Blue', 'Even')
y_pos = np.arange(len(objects))
performance=[len(red_favorite), len(blue_favorite), even_amount]
plt.bar(y_pos, performance, align='center', alpha=1, 
        color=['red', 'blue', 'purple'] )
plt.xticks(y_pos, objects)
plt.ylabel('Number of Fights')
plt.title('Corner of Favorite Fighter')

#df.to_csv('testfile.csv')
