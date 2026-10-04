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

## Prepping for the Models

Before creating any machine learning models, we need to prepare our data for them.

First, we must split our data. Usually, you would split the data using a percentage such as 80/20. However, for this project, I split the data by season. We train on the older seasons (**2015–2022**) and test on the newest seasons (**2023–2024**).

This mimics predicting the future, and it makes sure plays from the same game never end up in both the training and testing data.

The data is split with this code:

```python
train_df = df.loc[df["season"] <= 2022]
test_df = df.loc[df["season"] >= 2023]
```

Then we create our training and testing variables:

```python
X_trn = train_df[feature_cols]
y_trn = train_df["converted"]

X_tst = test_df[feature_cols]
y_tst = test_df["converted"]
```

Finally, we check if any games accidentally ended up in both datasets:

```python
print("Games in both sets:", len(set(train_df["game_id"]) & set(test_df["game_id"])))
```

This results in **zero games in both datasets**, meaning we are good to go.

### Baseline

Next, I create a baseline to test our models against later.

A baseline is the score to beat. Our baseline is a "model" that always guesses the most common outcome from the training data.

The baseline model always predicts **converted**, giving us a baseline accuracy of **0.545**. This means it correctly predicts a fourth-down conversion about **55% of the time**.

We are trying to beat this score with our machine learning models.

---

## Logistic Regression

Logistic regression works best when features are on a similar scale, so we standardize them first.

We learn the scaling (mean and spread) from the training data only and then apply it to the test data. This prevents test information from leaking into the training process and spoiling the results.

```python
scaler = StandardScaler()

X_trn_scaled = scaler.fit_transform(X_trn)

X_tst_scaled = scaler.transform(X_tst)
```

The logistic regression model achieved the following results:

| Outcome   | Accuracy | Recall |
| --------- | -------: | -----: |
| Failed    |     0.67 |   0.46 |
| Converted |     0.64 |   0.81 |

This tells us that the model correctly predicted a situation would lead to a fourth-down conversion **64% of the time**.

Of all the plays that actually resulted in a conversion, the model correctly identified **81%** of them.

The model also correctly predicted failed fourth downs **67% of the time**, but only caught **46% of the fourth downs that actually failed**.

---

## Decision Tree

Next, we create a decision tree to find which situations are best for going for it on fourth down.

We start by creating a decision tree that is not tuned yet.

The initial tree gave us a:

* **Training accuracy:** 1.00
* **Testing accuracy:** 0.542

This means the model is **overfitting**. The tree essentially memorized the training data rather than learning patterns that generalize to new data.

So, we need to find a better tree depth.

### Choosing the Tree Depth

We can't pick the depth by looking at the test set because that would be using our test data to tune the model.

Instead, we validate inside the training data.

For each of the last three training seasons (**2020, 2021, and 2022**), we train on the seasons before it and then check the accuracy on that season. We then average the three validation accuracies.

![Decision Tree Depth Validation](https://github.com/user-attachments/assets/f2a78508-d23d-4458-b447-759398044b67)

The graph shows that a **maximum tree depth of 3** gives us the best validation performance.

---

## What the Data Shows

These graphs show what we would expect to be true in the NFL.

Teams are more likely to convert when they are **close to the first-down marker** and **close to the opponent's goal line**.

Notably, teams also appear to be going for it more often as the seasons progress. There is about a **6% increase from 2017–2021**.
