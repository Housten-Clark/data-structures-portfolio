# In what situation does going for it on fourth down increase an NFL team's chances of winning the game?
> October 4th, 2026

**Problem Definition**   

Fourth down conversions are some of the most exciting and tense moment in football. Seeing your team convert in a high pressure situation to win the game is a testament to show how hype football can be. However, is going for it on 4th actually worth it? In which situation can we predict a conversion on 4th down? In this project I hope to answer this question. Using two machine learning models to find the best situation to go for it on fourth down.   

**Data description**  
The main dataset I used for this project is the nflready api, hosted at [this link](https://github.com/nflverse/nflreadpy). I am looking at games in the past 10 years, from the 2015-2025 seasons. The whole data set is far to large and is not needed for most of the project. So we only pull out the columns that we need. These being:  
- season
- game id
- description, shows what happened in the play
- down
- play type
- fourth down converted
- fourth down failed
- yards to go, shows yards to the first down marker
- yardline 100, shows yards to the opponents goal line
- quarter
- game seconds remaining
- score differential
- possession team timeouts remaining
- win percentage
- shotgun, is it a shotgun play?

  **Data cleaning**
  First off, we need to gather up only the plays we care about. Fourth downs that were not a kick or punt. This is done with the following code:  
  `fourth = pbp.loc[(pbp["down"] == 4) & (pbp["play_type"].isin(["run", "pass"]))]`   
   `print(f"All 4th-down run/pass plays: {len(fourth):,}")`
  
  The we will filter for plays that have a clear result of converted, or failed.
`fourth = fourth.loc[(fourth["fourth_down_converted"] + fourth["fourth_down_failed"]) == 1]`  
 `print(f"With a clear converted/failed result: {len(fourth):,}")`

   Finally, we get our target variable    
`fourth = fourth.copy()`  
 `fourth["converted"] = fourth["fourth_down_converted"].astype(int)`

This gave us a total of 6,625 fourth down plays to work with. Next we choose variables to be used before the snap, to see if we can predict when a fourth down is "worth it" to a team. Using other variables like yards gained, EPA, and win probability added would be data leakage. As these variables would not be available to our model before the snap. To do this we use the following code:  
`  feature_cols = [
    "ydstogo", 
    "yardline_100", 
    "qtr", 
    "game_seconds_remaining", 
    "score_differential", 
    "posteam_timeouts_remaining", 
    "wp", 
    "shotgun",
]  
df = fourth[["season", "game_id", "desc", "converted"] + feature_cols].copy()  
df.head()`  

Next we see if there are any empty columns:  
`df.isna().sum()`  

This returned that there were no empty columns, but to be safe we still drop all empty columns.  
`df = df.dropna().reset_index(drop=True)`  

**Visualizations**  
Before we start creating any models, we have to look at a few things. 

First off, lets look at our class balance.  

<p align="center" style="text-align:center;">
    <img width="472" height="372" alt="image" src="https://github.com/user-attachments/assets/dfd58df5-2ff5-482d-8736-4a7648224bc2" />  

Looks like our classes are pretty balanced, with a conversion rate of 52.2%.  

Next we're going to get a feel for 4th down conversions in the NFL. Looking at conversions by how many yards to the first down marker, field position, and when teams went for it the most by year.  

<p align="center" style="text-align:center;">
 <img width="691" height="430" alt="image" src="https://github.com/user-attachments/assets/2b06adc4-c4b8-4c59-98e1-3a0ffe71e134" />  

<p align="center" style="text-align:center;">
 <img width="691" height="430" alt="image" src="https://github.com/user-attachments/assets/1fa17a86-aa93-4feb-b817-91a28af9baf1" />  

 <p align="center" style="text-align:center;">
 <img width="700" height="430" alt="image" src="https://github.com/user-attachments/assets/3c826af7-00f5-45a5-a200-90f6352bdc95" />    

 These graphs show what we assume is true in the NFL. Teams convert more when they are close to the first down, and close to the opponents goaline. Notably, teams start going for it more and more as the seasons progress. There is about a 6% increase from 2017-2021.  

**Prepping for the models**

Before creating any machine learning models, we need to prepare our data for them. First, we must split our data. Usually you would split on a percentage of 80/20, however for this project I split by season. We train on the older seasons (2015-2022) and test on the newest seasons (2023-2024). This mimics predicting the future, and it makes sure plays from the same game never end up in both the training and testing data. The data is split with this code:

```
train_df = df.loc[df["season"] <= 2022]
test_df = df.loc[df["season"] >= 2023]
```

Then we create our test and training variables with this:

```
X_trn = train_df[feature_cols]
y_trn = train_df["converted"]
X_tst = test_df[feature_cols]
y_tst = test_df["converted"]
```

Finally, we check if any game slipped into both of the datasets:

```
print("Games in both sets:", len(set(train_df["game_id"]) & set(test_df["game_id"])))
```

This results in zero games in both datasets, meaning we are good to go.

Next I create a baseline to test our models against later. A baseline is the score to beat. Our baseline is a "model" that always guesses the most common outcome from the training data. The baseline model always predicts converted, and gives us a baseline accuracy of 0.545. Meaning it correctly predicts a fourth down will be converted about 55% of the time. We are trying to beat this score with our models.

**Logistic Regression**

Logistic regression works best when features are on a similar scale, so we standardize them first. We learn the scaling (mean and spread) from the training data only and then apply it to the test data, so no test information leaks in and spoils the results.

```
scaler = StandardScaler()
X_trn_scaled = scaler.fit_transform(X_trn)   # learn the scaling from training data only
X_tst_scaled = scaler.transform(X_tst)       # apply the same scaling to the test data
```

Then we run our logistic regression, and get the following result.

```text
              precision    recall  f1-score   support

      Failed       0.67      0.46      0.54       750
   Converted       0.64      0.81      0.72       898

    accuracy                           0.65      1648
   macro avg       0.65      0.63      0.63      1648
weighted avg       0.65      0.65      0.64      1648
```

The model correctly predicted a situation would lead to a fourth down conversion 64% of the time. Of all the plays, the model caught 81% of the ones that were actually converted. The model also predicted failed correctly 67% of the time, and only caught 46% of the failed plays.

**Decision Tree**

Next we create a decision tree to find which situation is best to go for it on fourth down. We start by creating a decision tree that is not tuned yet.

```
# A basic tree with no tuning
tree_default = DecisionTreeClassifier(random_state=42)
tree_default.fit(X_trn, y_trn)

print(f"Tree depth: {tree_default.get_depth()}")
print(f"Training accuracy: {tree_default.score(X_trn, y_trn):.3f}")
print(f"Testing accuracy:  {tree_default.score(X_tst, y_tst):.3f}")
```

This gave a training accuracy of 1.00, and testing accuracy of .542. Meaning we have overfit. The tree memorized the data rather than seeking patterns. Let's find a better tree depth.  

We can't pick the depth by looking at the test set (that would be cheating). Instead we validate inside the training data, for each of the last three training seasons (2020, 2021, 2022), we train on the seasons before it and check accuracy on that season. Then we average the three.  

<p align="center" style="text-align:center;">
 <img width="700" height="430" alt="image" src="https://github.com/user-attachments/assets/f2a78508-d23d-4458-b447-759398044b67" />  

 This graph shows that we should use a depth of 3
