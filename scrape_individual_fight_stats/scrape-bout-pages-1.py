# -*- coding: utf-8 -*-

import os
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime
import pickle


"""
Created on Sun Jul 26 14:14:34 2020

@author: matth
"""


"""The goal is to get these stats:
    
    SLpM, 
    StrikeAcc, 
    SApM, 
    Strike defense, 
    takedowns, 
    takedown attempts, 
    takedownacc, 
    takedown_defense, 
    and sub attempts
 
    
for each fight.  These are to be used for DraftKings things. 
"""
#First we need to scrape each individual-bout page

def make_sane_filename(filename):
    return("".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip())


event_pages = os.listdir('../scrape_all_events/event_webpages')


#print(event_pages)
    
link_list = []
for z in range(len(event_pages)):
    e = event_pages[z]
    
    html=open('../scrape_all_events/event_webpages/' + e)
    bs=BeautifulSoup(html, 'html.parser')

    bout_page_links_raw = bs.find_all('a', {'class':'b-flag b-flag_style_green'})

    count = 0
    for l in bout_page_links_raw:
        link_list.append([l.attrs['href'],
                          (str(count) + '--' + e)])
        count = count+1

#print(link_list)

    #print(bs)
    
with open('all_events_list', 'wb') as fp:
    pickle.dump(link_list, fp)
    
with open ('all_events_list', 'rb') as fp:
    itemlist = pickle.load(fp)

print(itemlist)    
    