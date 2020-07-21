# -*- coding: utf-8 -*-
import os
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime

"""
Created on Tue Jul 14 10:26:46 2020

@author: matth
"""


#The goal here to to use the event webpages to create a dataframe of finishes.
#We will then merge this with the big dataframe.

"""
What we need:
    R_fighter
    B_fighter
    date
    finish
"""

#Let's get a list of all the files in the event_webpages folder.

event_pages = os.listdir('event_webpages')
df = []

for z in range(len(event_pages)):
#for z in range(1):
    e = event_pages[z]
    
    html=open('event_webpages/' + e)
    bs=BeautifulSoup(html, 'html.parser')
    
    ##Let's get the date first....
    
    date_raw = bs.find_all('li', {'class':'b-list__box-list-item'})
    child_count=0
    for dr in date_raw:
        temp_count=0
        for child in dr.children:
            #print(child.string)
            #print(temp_count, child_count)
            if ((temp_count == 2) & (child_count == 0)):
                raw_date = (child.string.strip())
            if ((temp_count == 2) & (child_count == 1)):
                location = (child.string.strip())
                
            temp_count = temp_count+1
    
        child_count = child_count+1
    
    
    formatted_date = datetime.strptime(raw_date, "%B %d, %Y")
    date_datetime = formatted_date
    
    #The pound sign removes the leading 0.
    formatted_date=(formatted_date.strftime("%#m/%e/%Y"))
    
    
    
    
    
    fights = bs.find_all('td', {'class':'b-fight-details__table-col l-page_align_left'})
    
        
    f_count = 0
    fighters_raw = []
    
    
    #Each fight is split into 3 cells
    #The first has information about the fighters.  Their names and links
    #The 2nd has the weight class of the fight
    #The 3rd is junk
    for f in fights:
        
        if f_count%3 == 0 :
            fighters_raw.append(f)
        f_count=f_count+1
    
    
    red_fighter_list = []
    blue_fighter_list = []
    
    for f in fighters_raw:
        temp_fighters = f.find_all('p')
        temp_links = f.find_all('a')
        #print("Red fighter: ", temp_fighters[0].get_text().strip())
        #print("Red Link: ", temp_links[0].attrs['href'])
        #print("Blue Fighter:", temp_fighters[1].get_text().strip())
        #print("Blue Link: ", temp_links[1].attrs['href'])
        red_fighter_list.append(temp_fighters[0].get_text().strip())
        blue_fighter_list.append(temp_fighters[1].get_text().strip())
    
    
    #print(red_fighter_list)
    #print(blue_fighter_list)
    
    round_list = []
    time_list = []
    
    temp_list = bs.find_all('td', {'class':'b-fight-details__table-col'})
    
    print(len(temp_list))
    
    count = 0
    for t in temp_list:
        #print(f"COUNT: {count}")
        #print(t)
    
        if (count) % 10 == 8:
            #There are 2 paragraphs here.  One with the finish.  The other with the 
            print(f"ROUND: {t.get_text().strip()}")
            #print(count)
            #print(t)
            round_list.append(t.get_text().strip())
            #finish_details_list.append(temp_finish_list[1].get_text().strip())
        elif (count) % 10 == 9:
            time_list.append(t.get_text().strip())
        count = count+1
        
        
        
    print(len(red_fighter_list))
    print(len(blue_fighter_list))
    print(len(round_list))
    print(len(time_list))
    df.append(pd.DataFrame(list(zip(red_fighter_list, blue_fighter_list, round_list, 
                               time_list)), columns=['R_fighter', 'B_fighter',
                                                               'round', 'time']))
        
    df[z]['date'] = formatted_date                                                           
    #display(df[z]) 

total_df = pd.concat(df)

total_df.to_csv('finish_times.csv', index=False)