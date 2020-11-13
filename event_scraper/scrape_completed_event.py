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
html=urlopen('http://ufcstats.com/event-details/3bc27ec15facbcf3')
bs=BeautifulSoup(html, 'html.parser')
######################################################################
#So we aren't constantly scraping let's save the file.  This will have
#to be reverted before we go live.
#with open("temp.html", "w", encoding='utf-8') as file:
#    file.write(str(bs))



#REMOVE ME############################################################
#fight_file=open('temp.html', "r")
    
#bs=BeautifulSoup(fight_file.read(), 'html.parser')
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
weight_class_list = []

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


for w in weight_classes_raw:
    temp_wc = w.find_all('p')
    weight_class_list.append(temp_wc[0].get_text().strip())
#print(weight_class_list)
 
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
date_datetime = formatted_date

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
#weight_classes = bs.find_all('p', {'class':'b-fight-details__table-text'})
#weight_class_list = []
#temp_count=0
#for wc in weight_classes:
#    #print(temp_count)
#    if((temp_count+5)%10==0):
#        weight_class_list.append(wc.get_text().strip())
#    temp_count += 1
#
#print(weight_class_list)

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


#Find the current lose and win streaks
blue_fighter_win_streak = []
blue_fighter_lose_streak = []
red_fighter_win_streak = []
red_fighter_lose_streak = []
blue_draw_list = []
red_draw_list = []
blue_strike_list = []
red_strike_list = []
blue_strike_acc_list = []
red_strike_acc_list = []
sub_list = []
td_list = []
red_sub_list = []
red_td_list = []
td_acc_list = []
red_td_acc_list = []
red_fighter_longest_win_streak = []
blue_fighter_longest_win_streak = []
blue_total_losses = []
red_total_losses = []
blue_total_rounds = []
red_total_rounds = []
blue_title_bouts = []
red_title_bouts = []
blue_total_maj_dec = []
red_total_maj_dec = []
blue_total_split_dec = []
red_total_split_dec = []
blue_total_un_dec = []
red_total_un_dec = []
blue_total_ko = []
red_total_ko = []
blue_total_sub = []
red_total_sub = []
blue_total_wins = []
red_total_wins = []
stance_list = []
height_list = []
reach_list = []
weight_list = []
red_stance_list = []
red_height_list = []
red_reach_list = []
red_weight_list = []
blue_age_list = []
red_age_list = []

z = 0


for z in range(number_of_fights):
    print("new fight")
    b_fighter_file=open(f'fighter_pages/b{z}.html', "r")
    blue_soup=BeautifulSoup(b_fighter_file.read(), 'html.parser')
    blue_results_raw = blue_soup.find_all('i',{'class':'b-flag__text'})
    blue_rounds_raw = blue_soup.find_all('p', {'class':'b-fight-details__table-text'})
    
    ################################################################
    #Blue Total rounds fought
    #Round totals are on 21, 38, 55, 72... etc...
    #So that is (count - 4) % 17 = 0
    temp_count=0
    round_count = 0
    #print(len(blue_rounds_raw))
    count = 0
    for b in blue_rounds_raw:
        #print(count)
        #count = count+1
        #print(b.get_text())
        if (temp_count>20):
            if (temp_count -21) % 17 == 0:
                #print(b.get_text().strip())
                round_raw = b.get_text()
                print(round_raw)
                round_stripped = round_raw.strip()
                round_count+=int(round_stripped)
                #print(round_count)
        temp_count+=1
    blue_total_rounds.append(round_count)
    ################################################################

    ###############################################################
    #Blue total title bouts.  We are looking for 'belt.png'
    title_bout_count = 0
    
    #print(blue_soup)
    title_bout_count = str(blue_soup).count('belt.png')
    #print(title_bout_count)
    #If the upcoming fight is a title bout we need to subtract 1
    if(df.iloc[z]['title_bout']):
        title_bout_count -= 1
    blue_title_bouts.append(title_bout_count)
        
    
    
    
    
    
    ###############################################################








    ################################################################
    #Determine the type of win for BLUE
    temp_count = 0
    for b in blue_rounds_raw:
        #print(temp_count)
        #print(b.get_text())
        temp_count+=1
        
    #OK so it lists win or loss at 6, 23, 40...etc....
    #it lists type of win at 19, 36, 53, ....etc...
    
    
    temp_count=0
    dec_maj_count = 0
    dec_split_count = 0
    dec_un_count = 0
    ko_count = 0
    sub_count = 0
    win_flag = False #Set to true when we have a win

    for b in blue_rounds_raw:
        if (temp_count>5):
            if (temp_count-6) % 17 == 0:
                #We have a win
                if(b.get_text().strip()) == 'win':
                    win_flag = True
                else:
                    win_flag = False
        #Now we are going to look at the win_flag.  If it's
        #true we can tally the method
            if (temp_count-19) % 17 == 0:
                if (win_flag == True):
                    if(b.get_text().strip())=='M-DEC':
                        dec_maj_count += 1
                    elif(b.get_text().strip())=='S-DEC':
                        dec_split_count +=1
                    elif(b.get_text().strip())=='U-DEC':
                        dec_un_count += 1
                    elif(b.get_text().strip())=='KO/TKO':
                        ko_count += 1
                    elif(b.get_text().strip())=='SUB':
                        sub_count += 1
                    
        temp_count+=1
    
    blue_total_maj_dec.append(dec_maj_count)
    blue_total_split_dec.append(dec_split_count)    
    blue_total_un_dec.append(dec_un_count)
    blue_total_ko.append(ko_count)
    blue_total_sub.append(sub_count)
            #if (temp_count - 4) % 17 == 0:
    #            #print(b.get_text().strip())
    #            round_raw = b.get_text()
    #            round_stripped = round_raw.strip()
    #            round_count+=int(round_stripped)
    #            #print(round_count)
    #    temp_count+=1
    #blue_total_rounds.append(round_count)
    ################################################################












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

    #Let's get the longest win streak
    #We can get total losses here too
    longest_win_streak = 0
    temp_win_streak = 0
    total_losses=0
    total_wins=0
    for b in blue_results_raw:
        b=b.get_text()
        if b=='draw':
            if temp_win_streak > longest_win_streak:
                longest_win_streak = temp_win_streak
            temp_win_streak = 0
        if b=='win':
            temp_win_streak += 1
            total_wins+=1
        elif b=='loss':
            temp_win_streak = 0
            total_losses+=1
        if temp_win_streak > longest_win_streak:
            longest_win_streak = temp_win_streak

                    
        #print(r)
    #print(f"Win Streak: {win_streak}. Lose streak: {lose_streak}")
    blue_fighter_win_streak.append(win_streak)
    blue_fighter_lose_streak.append(lose_streak)
    blue_draw_list.append(draw_count)
    blue_fighter_longest_win_streak.append(longest_win_streak)
    blue_total_losses.append(total_losses)
    blue_total_wins.append(total_wins)
    
    r_fighter_file=open(f'fighter_pages/r{z}.html', "r")
    red_soup=BeautifulSoup(r_fighter_file.read(), 'html.parser')
    red_results_raw = red_soup.find_all('i',{'class':'b-flag__text'})
    red_rounds_raw = red_soup.find_all('p', {'class':'b-fight-details__table-text'})
 
    ################################################################
    #Red Total rounds fought
    #Round totals are on 21, 38, 55, 72... etc...
    #So that is (count - 4) % 17 = 0
    temp_count=0
    round_count = 0
    #print(len(blue_rounds_raw))
    for r in red_rounds_raw:
        if (temp_count>20):
            if (temp_count -21) % 17 == 0:
                #print(b.get_text().strip())
                round_raw = r.get_text()
                round_stripped = round_raw.strip()
                round_count+=int(round_stripped)
                #print(round_count)
                #print(round_count)
        temp_count+=1
    red_total_rounds.append(round_count)
    ################################################################


    ###############################################################
    #Red total title bouts.  We are looking for 'belt.png'
    title_bout_count = 0
    
    #print(blue_soup)
    title_bout_count = str(red_soup).count('belt.png')
    #print(title_bout_count)
    #If the upcoming fight is a title bout we need to subtract 1
    if(df.iloc[z]['title_bout']):
        title_bout_count -= 1
    red_title_bouts.append(title_bout_count)    
    
    ###############################################################




    ################################################################
    #Determine the type of win for BLUE
    temp_count = 0
    #OK so it lists win or loss at 6, 23, 40...etc....
    #it lists type of win at 19, 36, 53, ....etc...
    
    
    temp_count=0
    dec_maj_count = 0
    dec_split_count = 0
    dec_un_count = 0
    ko_count = 0
    sub_count = 0
    win_flag = False #Set to true when we have a win

    for r in red_rounds_raw:
        if (temp_count>5):
            if (temp_count-6) % 17 == 0:
                #We have a win
                if(r.get_text().strip()) == 'win':
                    win_flag = True
                else:
                    win_flag = False
        #Now we are going to look at the win_flag.  If it's
        #true we can tally the method
            if (temp_count-19) % 17 == 0:
                if (win_flag == True):
                    if(r.get_text().strip())=='M-DEC':
                        dec_maj_count += 1
                    elif(r.get_text().strip())=='S-DEC':
                        dec_split_count +=1
                    elif(r.get_text().strip())=='U-DEC':
                        dec_un_count += 1
                    elif(r.get_text().strip())=='KO/TKO':
                        ko_count += 1
                    elif(r.get_text().strip())=='SUB':
                        sub_count += 1
                    
        temp_count+=1
    
    red_total_maj_dec.append(dec_maj_count)
    red_total_split_dec.append(dec_split_count)    
    red_total_un_dec.append(dec_un_count)
    red_total_ko.append(ko_count)
    red_total_sub.append(sub_count)
            #if (temp_count - 4) % 17 == 0:
    #            #print(b.get_text().strip())
    #            round_raw = b.get_text()
    #            round_stripped = round_raw.strip()
    #            round_count+=int(round_stripped)
    #            #print(round_count)
    #    temp_count+=1
    #blue_total_rounds.append(round_count)
    ################################################################







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
                    
    #Let's get the longest win streak
    #We can get loss count in this loop too
    total_losses = 0
    total_wins = 0
    longest_win_streak = 0
    temp_win_streak = 0
    for r in red_results_raw:
        r=r.get_text()
        if r=='draw':
            if temp_win_streak > longest_win_streak:
                longest_win_streak = temp_win_streak
            temp_win_streak = 0
        if r=='win':
            temp_win_streak += 1
            total_wins+=1
        elif r=='loss':
            temp_win_streak = 0
            total_losses+=1
        if temp_win_streak > longest_win_streak:
            longest_win_streak = temp_win_streak
                    
        #print(r)
    #print(f"Win Streak: {win_streak}. Lose streak: {lose_streak}")
    red_fighter_win_streak.append(win_streak)
    red_fighter_lose_streak.append(lose_streak)
    red_draw_list.append(draw_count)
    red_fighter_longest_win_streak.append(longest_win_streak)
    red_total_losses.append(total_losses)
    red_total_wins.append(total_wins)
    
    ###################################################################
    #onto some data we do not need to calculate
    #Sig Strikes Landed: {SLpM}
    #Sig Strikes Percent {Str. Acc}
    blue_strikes_raw = blue_soup.find_all('li',
                            {'class':'b-list__box-list-item b-list__box-list-item_type_block'})

    red_strikes_raw = red_soup.find_all('li',
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
            if s_count == 11:   #td_accuracy
                #We need to remove the percent sign
                isolate_stat = isolate_stat[:-1]
                #We need to convert to decimal
                isolate_stat = float(isolate_stat) / 100
                td_acc_list.append(isolate_stat)
            if s_count ==3:
                #Stance
                stance_list.append(isolate_stat)
            if s_count == 0:
                #Height
                #print(isolate_stat)
                #We need to split into feet and inches and
                #convert to cm....
                isolate_stat = isolate_stat.replace("'", "")
                isolate_stat = isolate_stat.replace('"', '')
                height_tuple = isolate_stat.split(" ")
                if isolate_stat == ('--'):
                    total_inches = 0
                else:
                    total_inches = int(height_tuple[0])*12 + int(height_tuple[1])
                height_in_cm = total_inches * 2.54
                #print(height_tuple)
                #print(total_inches)
                #print(height_in_cm)
                height_list.append(height_in_cm)
            if s_count == 2:
                #Reach
                isolate_stat = isolate_stat.replace('"', '')
                if isolate_stat == ('--'):
                    reach_in_cm = height_in_cm
                else:
                    reach_in_cm = int(isolate_stat) * 2.54

                reach_list.append(reach_in_cm)
            if s_count == 1:
                #weight
                #print(isolate_stat)
                isolate_stat = isolate_stat.replace(" lbs.", '')
                #print(isolate_stat)
                weight_list.append(isolate_stat)
            if s_count == 4:
                #Age
                #print(isolate_stat)
                #print(formatted_date)
                if isolate_stat == '--':
                    age = 30
                else:
                    birth_date = datetime.strptime(isolate_stat, "%b %d, %Y")
                    age = date_datetime.year - birth_date.year - ((date_datetime.month, date_datetime.day) < (birth_date.month, birth_date.day))
                
                blue_age_list.append(age)
        #print(s_count)
        #print(s)
        s_count+=1
    #print()
    #print()
    #print()



    s_count = 0
    for s in red_strikes_raw:
        if s_count == 5:
            red_strikes = str(s)
            red_strikes = red_strikes.split('</i>')
            red_strikes = red_strikes[1]
            #print(temp)
            #There is a tag at the end we need to strip
            red_strikes = red_strikes[:-5]
            red_strikes=red_strikes.strip()
            #print(blue_strikes.strip())
            red_strike_list.append(red_strikes)
            #print(len(red_strike_list))
        if s_count == 6:
            red_str_acc = str(s)
            red_str_acc = red_str_acc.split('</i>')
            red_str_acc = red_str_acc[1]
            #print(temp)
            #There is a tag at the end we need to strip
            red_str_acc = red_str_acc[:-5]
            red_str_acc=red_str_acc.strip()
            #print(blue_strikes.strip())
            red_strike_acc_list.append('.'+red_str_acc[:-1])
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
                red_sub_list.append(isolate_stat)
            if s_count == 10:
                red_td_list.append(isolate_stat)
            if s_count == 11:   #td_accuracy
                #We need to remove the percent sign
                isolate_stat = isolate_stat[:-1]
                #We need to convert to decimal
                isolate_stat = float(isolate_stat) / 100
                red_td_acc_list.append(isolate_stat)
            if s_count ==3:
                #Stance
                red_stance_list.append(isolate_stat)
            if s_count == 0:
                #Height
                #print(isolate_stat)
                #We need to split into feet and inches and
                #convert to cm....
                isolate_stat = isolate_stat.replace("'", "")
                isolate_stat = isolate_stat.replace('"', '')
                height_tuple = isolate_stat.split(" ")
                total_inches = int(height_tuple[0])*12 + int(height_tuple[1])
                height_in_cm = total_inches * 2.54
                #print(height_tuple)
                #print(total_inches)
                #print(height_in_cm)
                red_height_list.append(height_in_cm)
            if s_count == 2:
                #Reach
                isolate_stat = isolate_stat.replace('"', '')
                if isolate_stat == '--':
                    reach_in_cm = height_in_cm
                else:
                    reach_in_cm = int(isolate_stat) * 2.54
                red_reach_list.append(reach_in_cm)
            if s_count == 1:
                #weight
                #print(isolate_stat)
                isolate_stat = isolate_stat.replace(" lbs.", '')
                #print(isolate_stat)
                red_weight_list.append(isolate_stat)
            if s_count == 4:
                #Age
                if isolate_stat == '--':
                    age = 30
                else:
                    birth_date = datetime.strptime(isolate_stat, "%b %d, %Y")
                    age = date_datetime.year - birth_date.year - ((date_datetime.month, date_datetime.day) < (birth_date.month, birth_date.day))
                red_age_list.append(age)


        s_count+=1











#Here we add all the lists to the dataframe    
df['B_current_win_streak'] = blue_fighter_win_streak
df['B_current_lose_streak'] = blue_fighter_lose_streak
df['R_current_win_streak'] = red_fighter_win_streak
df['R_current_lose_streak'] = red_fighter_lose_streak
df['R_longest_win_streak'] = red_fighter_longest_win_streak
df['B_longest_win_streak'] = blue_fighter_longest_win_streak
df['B_losses'] = blue_total_losses
df['R_losses'] = red_total_losses
df['B_total_rounds_fought'] = blue_total_rounds
df['R_total_rounds_fought'] = red_total_rounds
df['B_total_title_bouts'] = blue_title_bouts
df['R_total_title_bouts'] = red_title_bouts
df['B_win_by_Decision_Majority'] = blue_total_maj_dec
df['B_win_by_Decision_Split'] = blue_total_split_dec
df['B_win_by_Decision_Unanimous'] = blue_total_un_dec
df['B_win_by_KO/TKO'] = blue_total_ko
df['B_win_by_Submission'] = blue_total_sub
df['B_win_by_TKO_Doctor_Stoppage'] = 0
df['B_wins'] = blue_total_wins
df['R_wins'] = red_total_wins
df['R_win_by_Decision_Majority'] = red_total_maj_dec
df['R_win_by_Decision_Split'] = red_total_split_dec
df['R_win_by_Decision_Unanimous'] = red_total_un_dec
df['R_win_by_KO/TKO'] = red_total_ko
df['R_win_by_Submission'] = red_total_sub
df['R_win_by_TKO_Doctor_Stoppage'] = 0

df['B_Reach_cms'] = reach_list
df['B_Weight_lbs'] = weight_list
df['R_Reach_cms'] = red_reach_list
df['R_Weight_lbs'] = red_weight_list



#Draws
df['R_draw'] = red_draw_list
df['B_draw'] = blue_draw_list
df['B_avg_SIG_STR_landed'] = blue_strike_list
df['R_avg_SIG_STR_landed'] = red_strike_list
df['B_avg_SIG_STR_pct'] = blue_strike_acc_list
df['R_avg_SIG_STR_pct'] = red_strike_acc_list
df['B_avg_SUB_ATT'] = sub_list
df['B_avg_TD_landed'] = td_list
df['R_avg_SUB_ATT'] = red_sub_list
df['R_avg_TD_landed'] = red_td_list

df['B_avg_TD_pct'] = td_acc_list
df['R_avg_TD_pct'] = red_td_acc_list


df['B_Stance'] = stance_list
df['B_Height_cms'] = height_list
df['R_Stance'] = red_stance_list
df['R_Height_cms'] = red_height_list

df['B_age'] = blue_age_list
df['R_age'] = red_age_list

#Differences!!!
df['win_streak_dif'] = df['B_current_win_streak'] - df['R_current_win_streak']
df['lose_streak_dif'] = df['B_current_lose_streak'] - df['R_current_lose_streak']
df['longest_win_streak_dif'] = df['B_longest_win_streak'] - df['R_longest_win_streak']
df['win_dif'] = df['B_wins'] - df['R_wins']
df['loss_dif'] = df['B_losses'] - df['R_losses']
df['total_round_dif'] = df['B_total_rounds_fought'] - df['R_total_rounds_fought']
df['total_title_bout_dif'] = df['B_total_title_bouts'] - df['R_total_title_bouts']
df['ko_dif'] = df['B_win_by_KO/TKO'] - df['R_win_by_KO/TKO']
df['sub_dif'] = df['B_win_by_Submission'] - df['R_win_by_Submission']
df['height_dif'] = df['B_Height_cms'] - df['R_Height_cms']
df['reach_dif'] = df['B_Reach_cms'] - df['R_Reach_cms']
df['sig_str_dif'] = df['B_avg_SIG_STR_landed'].astype(float) - df['R_avg_SIG_STR_landed'].astype(float)
df['avg_sub_att_dif'] = df['B_avg_SUB_ATT'].astype(float) - df['R_avg_SUB_ATT'].astype(float)
df['avg_td_dif'] = df['B_avg_TD_landed'].astype(float) - df['R_avg_TD_landed'].astype(float)
df['empty_arena'] = 1
df['constant_1'] = 1
df['age_dif'] = df['B_age'] - df['R_age']
#print(blue_strike_acc_list)
#print(sub_list)









df.to_csv('scraped_event.csv', index=False)


