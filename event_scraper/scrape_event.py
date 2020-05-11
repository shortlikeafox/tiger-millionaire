# -*- coding: utf-8 -*-
import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
from datetime import datetime

#We are going to automate the data gathering for upcoming UFC events.
#First let's create an empty DataFrame.

#We are going to get the column list from the master-dataframe.  This
#will guarantee that they are the same.

temp_df = pd.read_csv("../data/ufc-master.csv")

column_list = temp_df.columns

#Now let's build an empty dataframe with the columns
df = pd.DataFrame(columns=column_list)



#OK.  Now we have a receptacle for the data.  That's the hard part
#right?


#There is an event page and individual fighter pages.  
#First let's grab all the information we can off of the event page
#then we can move to the individual fighter pages.



#REVERT ME!!!
#####################################################################
#html=urlopen('http://ufcstats.com/event-details/5f8e00c27b7e7410')
#bs=BeautifulSoup(html, 'html.parser')
######################################################################
#So we aren't constantly scraping let's save the file.  This will have
#to be reverted before we go live.
#with open("temp.html", "w", encoding='utf-8') as file:
#    file.write(str(bs))



#REMOVE ME############################################################
fight_file=open('temp.html', "r")
    
bs=BeautifulSoup(fight_file.read(), 'html.parser')
#######################################################################





fights = bs.find_all('td', {'class':'b-fight-details__table-col l-page_align_left'})
#print (len(fights))

f_count = 0
fighters_raw = []
weight_classes_raw = []


#Each fight is split into 3 cells
#The first has information about the fighters.  Their names and links
#The 2nd has the weight class of the fight
#The 3rd is junk
for f in fights:
    if f_count%3 == 0 :
        fighters_raw.append(f)
    if f_count%3 == 1:
        weight_classes_raw.append(f)

    f_count=f_count+1

#These lists will contain the fighter and a link
red_fighter_list = []
blue_fighter_list = []

for f in fighters_raw:
    temp_fighters = f.find_all('p')
    temp_links = f.find_all('a')
    #print("Red fighter: ", temp_fighters[0].get_text().strip())
    #print("Red Link: ", temp_links[0].attrs['href'])
    #print("Blue Fighter:", temp_fighters[1].get_text().strip())
    #print("Blue Link: ", temp_links[1].attrs['href'])
    red_fighter_list.append([temp_fighters[0].get_text().strip(),
                             temp_links[0].attrs['href']])
    blue_fighter_list.append([temp_fighters[1].get_text().strip(),
                             temp_links[1].attrs['href']])
    
#print(bs)

#print(red_fighter_list[0][1])
#print(blue_fighter_list)


###################################################################
#Insert R_fighter and B_fighter
#Let's start entering data into the dataframe!
for i in range(len(red_fighter_list)):
    df_temp = pd.DataFrame({'R_fighter': red_fighter_list[i][0],
                           'B_fighter': blue_fighter_list[i][0]},
                           index=[i]) 
    #print(df_temp)
    #print(df_temp)
    df = pd.concat([df, df_temp])
    
#display(df)

##################################################################
#Let's get the date, and location
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

#The pound sign removes the leading 0.
formatted_date=(formatted_date.strftime("%#m/%e/%Y"))
df['date'] = formatted_date
df['location'] = location


#################################################################
#Let's get the country
split_location = location.split(',')
country = split_location[len(split_location)-1]
#print(country.strip())
country=country.strip()
df['country'] = country


##################################################################
#Set the winner to Red, because it's how the coding works.  THis
#will be updated when the event is complete
df['Winner']='Red'



#################################################################
#Set weight class
weight_classes = bs.find_all('p', {'class':'b-fight-details__table-text'})
weight_class_list = []
temp_count=0
for wc in weight_classes:
    #print(temp_count)
    if((temp_count+5)%10==0):
        weight_class_list.append(wc.get_text().strip())
    temp_count += 1

df['weight_class']=weight_class_list



##################################################################
#Set title_bout
#THIS NEEDS TO BE UPDATED WHEN WE HAVE AN ACTUAL TITLE FIGHT
#IT HAS TO DO WITH AN IMAGE NEXT TO THE WEIGHT CLASS.  SO WE CAN
#TIE THIS INTO HOW WE DETERMINE THE WEIGHT CLASS
number_of_fights = len(weight_class_list)
title_fight_list = []
for z in range(number_of_fights):
    title_fight_list.append(False)

df['title_bout'] = title_fight_list



##################################################################
#Set Gender... We can use the weight_class_list for this
#How this works is we look at the weight class name.  If the first
#word is "Women's" we are dealing with a FEMALE fight.  Otherwise
#MALE
gender_list = []
for wc in weight_class_list:
    if wc.split(' ')[0] == "Women's":
        gender_list.append('FEMALE')
    else:
        gender_list.append('MALE')

df['gender'] = gender_list




##################################################################
#Determine the number of rounds.  First check for title fight.
#All title fights are 5 rounds.  The main event is also 5 rounds.

round_list = []
for z in range(number_of_fights):
    if(title_fight_list==True):
        round_list.append(5)
    else:
        round_list.append(3)
        
round_list[0] = 5

#print(round_list)

df['no_of_rounds'] = round_list




#################################################################
#Now we need access to the fighter pages!

#First let's save them all so we don't have to constantly access them


#REVERT BEFORE GOING LIVE
"""
red_count = 0
for f in red_fighter_list:
    print(f[1][7:])
    
    html= urlopen(f[1])
    bs = BeautifulSoup(html.read(), 'html.parser')
    with open(f'fighter_pages/r{red_count}.html', "w", encoding='utf-8') as file:
        file.write(str(bs))
 

    red_count+=1

blue_count = 0
for f in blue_fighter_list:
    print(f[1][7:])
    
    html= urlopen(f[1])
    bs = BeautifulSoup(html.read(), 'html.parser')
    with open(f'fighter_pages/b{blue_count}.html', "w", encoding='utf-8') as file:
        file.write(str(bs))
 

    blue_count+=1
"""

#Find the current lose and win streaks
blue_fighter_win_streak = []
blue_fighter_lose_streak = []
red_fighter_win_streak = []
red_fighter_lose_streak = []
blue_draw_list = []
red_draw_list = []
blue_strike_list = []
blue_strike_acc_list = []
sub_list = []
td_list = []

z = 0
for z in range(number_of_fights):
    b_fighter_file=open(f'fighter_pages/b{z}.html', "r")
    blue_soup=BeautifulSoup(b_fighter_file.read(), 'html.parser')
    blue_results_raw = blue_soup.find_all('i',{'class':'b-flag__text'})
    win_streak = 0
    lose_streak =0
    draw_count=0
    end_streak = False #Set to true when the streak is over
    for r in blue_results_raw:
        r=r.get_text()
        if r=='draw':
            draw_count+=1        
        if end_streak == False:
            if r=='next': #Usually the first line.  Just skip
                pass
            elif r=='win':
                if (win_streak>0):
                    win_streak+=1
                elif(win_streak==0 and lose_streak==0):
                    win_streak+=1
                else:
                    end_streak = True
            elif r=='loss':
                if (lose_streak>0):
                    lose_streak+=1
                elif(win_streak==0 and lose_streak==0):
                    lose_streak+=1
                else:
                    end_streak=True
                    
        #print(r)
    #print(f"Win Streak: {win_streak}. Lose streak: {lose_streak}")
    blue_fighter_win_streak.append(win_streak)
    blue_fighter_lose_streak.append(lose_streak)
    blue_draw_list.append(draw_count)
    
    r_fighter_file=open(f'fighter_pages/r{z}.html', "r")
    red_soup=BeautifulSoup(r_fighter_file.read(), 'html.parser')
    red_results_raw = red_soup.find_all('i',{'class':'b-flag__text'})
    win_streak = 0
    lose_streak =0
    draw_count=0
    end_streak = False #Set to true when the streak is over
    for r in red_results_raw:
        r=r.get_text()
        if r=='draw':
            draw_count+=1
        if end_streak == False:
            if r=='next': #Usually the first line.  Just skip
                pass
            elif r=='win':
                if (win_streak>0):
                    win_streak+=1
                elif(win_streak==0 and lose_streak==0):
                    win_streak+=1
                else:
                    end_streak = True
            elif r=='loss':
                if (lose_streak>0):
                    lose_streak+=1
                elif(win_streak==0 and lose_streak==0):
                    lose_streak+=1
                else:
                    end_streak=True
                    
        #print(r)
    #print(f"Win Streak: {win_streak}. Lose streak: {lose_streak}")
    red_fighter_win_streak.append(win_streak)
    red_fighter_lose_streak.append(lose_streak)
    red_draw_list.append(draw_count)
    
    ###################################################################
    #onto some data we do not need to calculate
    #Sig Strikes Landed: {SLpM}
    #Sig Strikes Percent {Str. Acc}
    blue_strikes_raw = blue_soup.find_all('li',
                            {'class':'b-list__box-list-item b-list__box-list-item_type_block'})
    #print()
    #print()
    #print()
    s_count = 0
    for s in blue_strikes_raw:
        if s_count == 5:
            blue_strikes = str(s)
            blue_strikes = blue_strikes.split('</i>')
            blue_strikes = blue_strikes[1]
            #print(temp)
            #There is a tag at the end we need to strip
            blue_strikes = blue_strikes[:-5]
            blue_strikes=blue_strikes.strip()
            #print(blue_strikes.strip())
            blue_strike_list.append(blue_strikes)
            #print(s)   
        if s_count == 6:
            blue_str_acc = str(s)
            blue_str_acc = blue_str_acc.split('</i>')
            blue_str_acc = blue_str_acc[1]
            #print(temp)
            #There is a tag at the end we need to strip
            blue_str_acc = blue_str_acc[:-5]
            blue_str_acc=blue_str_acc.strip()
            #print(blue_strikes.strip())
            blue_strike_acc_list.append('.'+blue_str_acc[:-1])
            #print(s)   
        else:
            #I think we can get the value without caring too
            #much what it is..... This should save some coding
            isolate_stat = str(s)
            isolate_stat = isolate_stat.split('</i>')
            isolate_stat = isolate_stat[1]
            isolate_stat = isolate_stat[:-5]
            isolate_stat = isolate_stat.strip()
            if s_count == 13:
                sub_list.append(isolate_stat)
            if s_count == 10:
                td_list.append(isolate_stat)
                


        print(s_count)
        print(s)
        s_count+=1
    #print()
    #print()
    #print()

#Here we add all the lists to the dataframe    
df['B_current_win_streak'] = blue_fighter_win_streak
df['B_current_lose_streak'] = blue_fighter_lose_streak
df['R_current_win_streak'] = red_fighter_win_streak
df['R_current_lose_streak'] = red_fighter_lose_streak

#Draws
df['R_draw'] = red_draw_list
df['B_draw'] = blue_draw_list
df['B_avg_SIG_STR_landed'] = blue_strike_list
df['B_avg_SIG_STR_pct'] = blue_strike_acc_list
df['B_avg_SUB_ATT'] = sub_list
df['B_avg_TD_landed'] = td_list
#print(blue_strike_acc_list)
#print(sub_list)



df.to_csv('scraped_event.csv', index=False)


