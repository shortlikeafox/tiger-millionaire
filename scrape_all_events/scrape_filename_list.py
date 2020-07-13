# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:37:12 2020

@author: matth
"""

import pandas as pd
from urllib.request import urlopen, Request
import urllib.request
from bs4 import BeautifulSoup
from datetime import datetime
import pickle

def make_sane_filename(filename):
    return("".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip())

#First let's get a list of all of the event links....
link_list = []

for z in range(23):
    z  = z+1
    page_name = 'http://ufcstats.com/statistics/events/completed?page=' + str(z)
    print(page_name)
    req = Request(page_name, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(req).read()
    bs=BeautifulSoup(html, 'html.parser')
    
    #All the links seem to be in a table.... let's see if we can figure it out
    bs_filtered = bs.find('tbody')
    #print(bs_filtered)
    
    #Now we just grab the links?
    links = bs_filtered.find_all('a')
    
    
    
    for l in links:
        #print((links[0]))
        link_list.append([l.attrs['href'],
                          make_sane_filename(l.get_text().strip())])
    
    
    #print(link_list)

with open('all_events_list', 'wb') as fp:
    pickle.dump(link_list, fp)
    
with open ('all_events_list', 'rb') as fp:
    itemlist = pickle.load(fp)

print(itemlist)    

#display(bs)