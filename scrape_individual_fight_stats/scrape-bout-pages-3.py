# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:05:23 2020

@author: matth
"""

import os
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime
import pickle

event_pages = os.listdir('../scrape_all_events/event_webpages')
date_list = []

for z in range(len(event_pages)):
    e = event_pages[z]
    
    html=open('../scrape_all_events/event_webpages/' + e)
    bs=BeautifulSoup(html, 'html.parser')

    #print(bs)    
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
    
    date_list.append([e, formatted_date])

#print(date_list)


with open('date_list', 'wb') as fp:
    pickle.dump(date_list, fp)
    
with open ('date_list', 'rb') as fp:
    itemlist = pickle.load(fp)

print(itemlist)    


#Here we are going to create a date list.


