# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:15:29 2020

@author: matth
"""

#Here we are actually going to scrape the info and make a dataframe.
#Let's just grab everything, because we can....


import os
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime
import pickle

"""
Each stat has a red and blue component, unless otherwise stated:
    
#Non-RED/BLUE stats
7. Red Fighter
8. Blue Fighter\
9. Date (we need to get this from the outside list)


#RED/BLUE stats
1. KD
2. Sig Str Landed
3. Sig Str Attempted
4. Sig Str %
5. Total Strikes Attempted
6. Total Strikes Landed
7. TD Attempted
8. TD Landed
9 TD %
10. Sub Attempt
11. Pass
12. Rev
13. Sig Strikes Head Attempted
14. Sig STrikes Head Landed
15. Sig Strikes Body Attempted
16. Sig Strikes Body Landed
17. Sig Strikes Leg Attempted
18. Sig Strikes Leg Landed
19. Distance Attempted
20. Distance Landed
21. Cliinch Attempted
22. Clinch Landed
23. Ground Attempted
24. Ground Landed


"""
#Grab the datelist
with open ('date_list', 'rb') as fp:
    datelist = pickle.load(fp)

#Create date dictionary
temp_event_list = []
temp_date_list = []
for n in range(len(datelist)):
    temp_event_list.append(datelist[n][0])
    temp_date_list.append(datelist[n][1])

date_dict = dict(zip(temp_event_list, temp_date_list))


bout_pages = os.listdir('bout_webpages')
df = []
#print(bout_pages)

for z in range(len(bout_pages)):
    print(z)
    #print("HI")
    b = bout_pages[z]
    html=open('bout_webpages/' + b)
    bs=BeautifulSoup(html, 'html.parser')
    
    print(b)

    #7. Red Fighter
    #8. Blue Fighter

    pos_names = bs.find_all('a', {'style': 'color:#B10101'})  
    if len(pos_names) > 1:
        r_fighter = pos_names[0].get_text().strip()
        b_fighter = pos_names[1].get_text().strip()        
    else:
        print("uh-oh")
        r_fighter = ''
        b_fighter = ''
    #9. Date
    date = (date_dict[b.split('--')[1][:-5]])

    
    
    #RED/BLUE stats
    #"b-fight-details__table-col"
    pos_vals = bs.find_all('td', {'class': 'b-fight-details__table-col'})
    #print(len(pos_vals))
    count=0
    for p in pos_vals:
    #1. KD
        pos_feature = p.find_all('p', {'class': 'b-fight-details__table-text'})
        if count==1: 
            r_kd = pos_feature[0].get_text().strip()
            b_kd = pos_feature[1].get_text().strip()
            #print(f"Blue KD: {b_kd}; red kd: {r_kd}")
    #2. Sig Str Landed
    #3. Sig Str Attempted
        if count==2:
            R_sig_str_landed = pos_feature[0].get_text().strip().split()[0]
            B_sig_str_landed = pos_feature[1].get_text().strip().split()[0]
            R_sig_str_attempted = pos_feature[0].get_text().strip().split()[2]
            B_sig_str_attempted = pos_feature[1].get_text().strip().split()[2]

    #4. Sig Str %
        if count==3:
            #We need to remove the percent sign and add a '.'
            R_sig_str_pct = '.' + pos_feature[0].get_text().strip()[:-1]
            B_sig_str_pct = '.' + pos_feature[1].get_text().strip()[:-1]
            #We have a problem with 100% being represented as 10%...
            if R_sig_str_pct == '.100':
                R_sig_str_pct = '1'
            if B_sig_str_pct == '.100':
                B_sig_str_pct = '1'
            
    #5. Total Strikes Attempted
    #6. Total Strikes Landed
        if count==4:
            R_str_landed = pos_feature[0].get_text().strip().split()[0]
            B_str_landed = pos_feature[1].get_text().strip().split()[0]
            R_str_attempted = pos_feature[0].get_text().strip().split()[2]
            B_str_attempted = pos_feature[1].get_text().strip().split()[2]

    #7. TD Attempted
    #8. TD Landed
        if count==5:
            R_td_landed = pos_feature[0].get_text().strip().split()[0]
            B_td_landed = pos_feature[1].get_text().strip().split()[0]
            R_td_attempted = pos_feature[0].get_text().strip().split()[2]
            B_td_attempted = pos_feature[1].get_text().strip().split()[2]
            
    #9 TD %
        if count==6:
            #We need to remove the percent sign and add a '.'
            R_td_pct = '.' + pos_feature[0].get_text().strip()[:-1]
            B_td_pct = '.' + pos_feature[1].get_text().strip()[:-1]
            if R_td_pct == '.100':
                R_td_pct = '1'
            if B_td_pct == '.100':
                B_td_pct = '1'

    #10. Sub Attempt
        if count==7:
            R_sub = pos_feature[0].get_text().strip()
            B_sub = pos_feature[1].get_text().strip()
            
    #11. Pass
        if count==8:   
            R_pass = pos_feature[0].get_text().strip()
            B_pass = pos_feature[1].get_text().strip()

    #12. Rev.
        if count==9:
            R_rev = pos_feature[0].get_text().strip()
            B_rev = pos_feature[1].get_text().strip()
 
            
        #print(count)
        #print(p)
#        print()
        count=count+1

    df.append(pd.DataFrame([[r_fighter, b_fighter, date, r_kd, b_kd,
                        R_sig_str_landed, B_sig_str_landed,
                        R_sig_str_attempted, B_sig_str_attempted,
                        R_sig_str_pct, B_sig_str_pct,
                        R_str_landed, B_str_landed,
                        R_str_attempted, B_str_attempted,
                        R_td_landed, B_td_landed,
                        R_td_attempted, B_td_attempted,
                        R_td_pct, B_td_pct,
                        R_sub, B_sub,
                        R_pass, B_pass,
                        R_rev, B_rev
                        
                        
                        ]], 
                           columns=['R_fighter', 'B_fighter', 
                                    'date', 'R_kd_bout', 'B_kd_bout',
                                    'R_sig_str_landed_bout',
                                    'B_sig_str_landed_bout',
                                    'R_sig_str_attempted_bout',
                                    'B_sig_str_attempted_bout',
                                    'R_sig_str_pct_bout',
                                    'B_sig_str_pct_bout',
                                    'R_tot_str_landed_bout',
                                    'B_tot_str_landed_bout',
                                    'R_tot_str_attempted_bout',
                                    'B_tot_str_attempted_bout',
                                    'R_td_landed_bout',
                                    'B_td_landed_bout',
                                    'R_td_attempted_bout',
                                    'B_td_attempted_bout',
                                    'R_td_pct_bout',
                                    'B_td_pct_bout',
                                    'R_sub_attempts_bout',
                                    'B_sub_attempts_bout',
                                    'R_pass_bout', 'B_pass_bout',
                                    'R_rev_bout', 'B_rev_bout'
                                    
                                    
                                    
                                    ]))


total_df = pd.concat(df)

total_df.to_csv('bout_info.csv', index=False)    
