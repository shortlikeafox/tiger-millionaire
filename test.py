# -*- coding: utf-8 -*-

#We will test stuff here...
import header as h

import pandas as pd

pd.set_option('display.max_rows', 500) #Used for debugging


df = h.create_fight_df(h.MASTER_CSV_FILE)

print(df.head)


print(df.describe())


print(len(df))

#print(df.head)

#print(df.dtypes)

#OK... Now we need to create some dataframes....

