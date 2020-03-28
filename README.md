# tiger-millionaire
## UFC Fight Predictor

Tiger Millionaire (T-M) is a project whose goal is to use machine learning models to predict the winner of Ultimate Fighting Championship (UFC) fights. Not only will it predict winners, it will use the odds to predict profitable bets. The project will be deemed to be a success if we can create a model that is able to return a profit.  

***
This is a work in progress.  This readme will be updated as work progresses.

***
### Initial Steps

* The original data set comes from [WarrierRajeev's Kaggle dataset](https://www.kaggle.com/rajeevw/ufcdata/)
* I have modified this data to add extra features and remove some rows
* The models predict who will win: 'Blue' or 'Red'.  The red fighter is generally the favorite.

![Red = 2629, Blue = 1459, Draw = 79](images/ss2.PNG)

* Because of this I predict models that have a high true positive rate with respect to blue could be the most profitable.  Being able to predict underdog winners means more return per winning bet.
* My best model currently has a true positive rate of .57.

![TP = 97, FN = 73, TN = 198, FP = 49](images/ss1.PNG)

### Acknowledgements

* This project is built off of WarrierRajeev's initial work: (https://github.com/WarrierRajeev/UFC-Predictions)
