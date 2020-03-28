# -*- coding: utf-8 -*-
#Here we will visualize the models.  
#We are interested in looking at the confusion matrices

import header as h
import pandas as pd
import numpy as np
import fs_definitions as fsd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

df = h.create_master_df()   #Create the master df

for fs in h.FEATURES:
    
    temp_df = fsd.create_prepped_df(fs, df) #create the prepped df
    y = temp_df['label'] #Set 'y' to the winner value
    
    #Remove the winner and label from the features
    temp_prepped_df = temp_df.drop(['Winner', 'label'], axis=1)
    
    X = temp_prepped_df.values #Set x to the values
    
    #class_names = temp_df.Winner #This should match to the winners?
    class_names = ['Blue', 'Red']
    
    print(class_names)
    
    print(X.shape)
    print(y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state = 85)
    
    print('Training Features Shape:', X_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', X_test.shape)
    print('Testing Labels Shape:', y_test.shape)
    
    
    classifier = h.get_classifier(fs)
    
    classifier.fit(X_train, y_train)
    
    #Plot confusion matrix
    titles_options= [(f"{fs} Confusion matrix", None),
                     ("Normalized confusion matrix", 'true')]
    
    title = f"{fs} Confusion matrix"
    normalize=None
    
    
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize,
                                 values_format='.5g'
                                 )
    disp.ax_.set_title(title)
    plt.rcParams.update({'font.size': 16})
    print(title)
    print(disp.confusion_matrix)
    plt.grid(False)
    plt.show()
    
    predictions = classifier.predict(X_test)
    
    errors = abs(predictions - y_test)
    total_errors = (sum(errors))
    
    
    print(f"The number of errors is {sum(errors)}")
    print(f"That means I am getting {len(predictions) - sum(errors)} right\
          out of {len(predictions)} for a {(len(predictions) - sum(errors)) / (len(predictions))}")
    
    
    print(f"In the test set 0 wins {len(y_test) - sum(y_test)}")
    print(f"In the test set 1 wins {sum(y_test)}")
    print(f"I predict 0 to win {len(predictions) - sum(predictions)}")
    print(f"I predict 1 to win {sum(predictions)}")
    
    cm = confusion_matrix(predictions, y_test)
    
    tp = cm[0][0] 
    tn = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    
    total = tp + tn + fp + fn
    
    print(f"tp for {fs}: {tp}")
    print(f"tn: {tn}")
    print(f"fp: {fp}")
    print(f"fn: {fn}")
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp)
    
    #***I think that True Positive Rate may be the indicator of a good
    #model....
    
    true_positive = tp / (tp + fn)
    
    print(f"The precision is: {precision}")
    print(f"The accuracy is {accuracy}")
    print(f"The prevalence of blue is {(tp + fn) / total}")
    print(f"The true_positive rate for {fs} is {true_positive}")