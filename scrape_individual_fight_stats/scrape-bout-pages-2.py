# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:27:05 2020

@author: matth
"""

import os
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime
import pickle

#OK We have the list of links.  We just need to scrape and save those pages

with open ('all_events_list', 'rb') as fp:
    filename_list = pickle.load(fp)

print(len(filename_list))

for f in filename_list:
    #Let's scrape!!
    html = urlopen(f[0])
    bs = BeautifulSoup(html.read(), 'html.parser')
    with open(f'bout_webpages/{f[1]}.html', "w", encoding='utf-8') as file:
        file.write(str(bs))