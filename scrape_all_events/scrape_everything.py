import pandas as pd
from urllib.request import urlopen, Request
import urllib.request
from bs4 import BeautifulSoup
from datetime import datetime
import pickle


with open ('all_events_list', 'rb') as fp:
    filename_list = pickle.load(fp)

print(len(filename_list))

for f in filename_list:
    #Let's scrape!!
    html = urlopen(f[0])
    bs = BeautifulSoup(html.read(), 'html.parser')
    with open(f'event_webpages/{f[1]}.html', "w", encoding='utf-8') as file:
        file.write(str(bs))