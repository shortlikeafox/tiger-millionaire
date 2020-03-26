# -*- coding: utf-8 -*-

#We will test stuff here...
import header as h

import pandas as pd

pd.set_option('display.max_rows', 500) #Used for debugging


#df = h.create_fight_df(h.MASTER_CSV_FILE)

#print(df.head)


#print(df.describe())


#print(len(df))

#print(df.head)


#OK... Now we need to create some dataframes....

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

df.to_csv('testfile.csv')

df2 = h.create_master_df()
#adding a comment