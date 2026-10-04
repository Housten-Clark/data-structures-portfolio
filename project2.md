# In What Situation Does Going for It on Fourth Down Increase an NFL Team's Chances of Winning the Game?

> October 4, 2026

## Problem Definition

Fourth-down conversions are some of the most exciting and tense moments in football. Seeing your team convert in a high-pressure situation is one of the most exciting parts of the game.

However, is going for it on fourth down actually worth it? In which situations can we predict a conversion on fourth down?

In this project, I use two machine learning models to find the best situations to go for it on fourth down.

---

## Data Description

The main dataset I used for this project is the `nflreadpy` API, hosted by [nflverse](https://github.com/nflverse/nflreadpy).

I am looking at games from the 2015–2025 seasons. The entire dataset is far too large and contains information that is not needed for most of the project, so I only pull out the columns that we need.

These include:

* `season`
* `game_id`
* `desc` — shows what happened in the play
* `down`
* `play_type`
* `fourth_down_converted`
* `fourth_down_failed`
* `ydstogo` — yards to the first-down marker
* `yardline_100` — yards to the opponent's goal line
* `qtr`
* `game_seconds_remaining`
* `score_differential`
* `posteam_timeouts_remaining`
* `wp` — win probability
* `shotgun` — whether the play was run from shotgun

---

## Data Cleaning

First, we need to gather only the plays we care about: fourth downs that were not a kick or punt.

This is done with the following code:

```python
fourth = pbp.loc[
    (pbp["down"] == 4) &
    (pbp["play_type"].isin(["run", "pass"]))
]

print(f"All 4th-down run/pass plays: {len(fourth):,}")
```

Next, we filter for plays that have a clear result of either converted or failed:

```python
fourth = fourth.loc[
    (fourth["fourth_down_converted"] +
     fourth["fourth_down_failed"]) == 1
]

print(f"With a clear converted/failed result: {len(fourth):,}")
```

Finally, we create our target variable:

```python
fourth = fourth.copy()
fourth["converted"] = fourth["fourth_down_converted"].astype(int)
```

This gave us a total of **6,625 fourth-down plays** to work with.

Next, we choose variables that would be available before the snap to see if we can predict when a fourth down is worth going for.

Using variables such as yards gained, EPA, and win probability added would cause **data leakage** because those variables would not be available to our model before the snap.

We therefore use the following features:

```python
feature_cols = [
    "ydstogo",
    "yardline_100",
    "qtr",
    "game_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "wp",
    "shotgun",
]

df = fourth[
    ["season", "game_id", "desc", "converted"] + feature_cols
].copy()

df.head()
```

Next, we check if there are any missing values:

```python
df.isna().sum()
```

This returned that there were no empty columns, but to be safe, we still drop any rows containing missing values:

```python
df = df.dropna().reset_index(drop=True)
```

---

## Visualizations

Before we start creating any models, we first need to look at the data.

### Class Balance

First, let's look at our class balance.

![Class Balance](https://github.com/user-attachments/assets/dfd58df5-2ff5-482d-8736-4a7648224bc2)

The classes are relatively balanced, with a conversion rate of **52.2%**.

### Fourth-Down Conversions

Next, we're going to get a feel for fourth-down conversions in the NFL.

We look at conversions based on:

* How many yards the offense needs for a first down
* Field position
* When teams went for it most often by year

![Fourth Down Conversions by Yards to Go](https://github.com/user-attachments/assets/2b06adc4-c4b8-4c59-98e1-3a0ffe71e134)

![Fourth Down Conversions by Field Position](https://github.com/user-attachments/assets/1fa17a86-aa93-4feb-b817-91a28af9baf1)

![Fourth Down Attempts by Year](https://github.com/user-attachments/assets/3c826af7-00f5-45a5-a200-90f6352bdc95)

These graphs show what we would expect to be true in the NFL.

Teams convert more often when they are **close to the first-down marker**, and we see that fourth down conversion attempts have gone up over the years.  

### Choosing the Tree Depth

We can't pick the depth by looking at the test set because that would be using the test data to tune our model.

Instead, we validate within the training data. For each of the last three training seasons (**2020, 2021, and 2022**), we train on the seasons before it and check the accuracy on that season. We then average the three validation accuracies.

![Accuracy by Tree Depth](https://github.com/user-attachments/assets/f2a78508-d23d-4458-b447-759398044b67)

The graph shows that a **maximum tree depth of 3** is the best choice. At depth 3, the validation accuracy is at its highest level before beginning to decrease as the tree becomes deeper. Meanwhile, training accuracy continues to increase, which is a sign that deeper trees begin to overfit the training data.

Therefore, we will use a **maximum tree depth of 3** for our final decision tree.  

## Final Decision Tree Results

After choosing a maximum tree depth of 3, we ran the final decision tree on the test data.

The model produced the following results:

| Outcome | Precision | Recall | F1-Score | Support |
|:---|---:|---:|---:|---:|
| Failed | 0.61 | 0.62 | 0.61 | 750 |
| Converted | 0.67 | 0.66 | 0.67 | 898 |

The decision tree achieved an overall **accuracy of 64%**.

For fourth downs that **failed**, the model had a precision of **61%** and a recall of **62%**. This means that when the model predicted a fourth down would fail, it was correct about 61% of the time. It also correctly identified about 62% of the fourth downs that actually failed.

For fourth downs that were **converted**, the model had a precision of **67%** and a recall of **66%**. This means that when the model predicted a conversion, it was correct about 67% of the time. It also correctly identified about 66% of the fourth downs that actually converted.

Overall, the model had an **F1-score of 0.64** and an accuracy of **64%**, which is better than our baseline accuracy of **54.5%**. This shows that the decision tree was able to learn useful patterns about when teams are more likely to convert on fourth down.  

### What This Means

The decision tree improved on our baseline by about **9.5 percentage points**, showing that factors such as yards to go, field position, time remaining, score differential, and other pre-snap information provide useful information for predicting fourth-down conversions.  

## Visualizing the Decision Tree

![Tuned Decision Tree](https://github.com/user-attachments/assets/99eae58c-9b57-4302-9ebf-6bc1bc51b4c9)  

The decision tree helps us visualize the situations that the model believes are most important when predicting whether a fourth down will be converted.

The **first and most important split is yards to go**. The tree first separates plays where the offense has **2.5 yards or fewer to go** from plays where the offense has more than 2.5 yards to go. This shows that the distance needed for a first down is the most important factor in the model.

### Short Yardage Situations

When a team has **2.5 yards or fewer to go**, the model generally predicts a **conversion**.

The tree then looks at field position. If the offense is within about **20 yards of the opponent's goal line**, the model continues to predict a conversion. This makes sense because teams are more likely to convert short-yardage situations, and teams may also be more aggressive when they are close to scoring.

For other short-yardage situations, the tree considers **time remaining in the game**. This shows that game situation can also influence the model's prediction.

### Longer Yardage Situations

When a team has more than **2.5 yards to go**, the tree becomes more likely to predict a **failed conversion**.

For teams with between roughly **3 and 8 yards to go**, the model looks at the amount of time remaining. If there are only about **76 seconds or less remaining**, the model predicts a failure.

For teams with more than **8.5 yards to go**, the model continues to predict a failure. It then makes another split at **15.5 yards to go**, but both resulting groups are still predicted to fail.

### What the Tree Tells Us

Overall, the decision tree gives us a pretty intuitive result:

1. **Yards to go is the most important factor.**
2. **Short-yardage situations are more likely to be converted.**
3. **Field position becomes important in some short-yardage situations.**
4. **Time remaining can change the prediction in certain situations.**
5. **Long-yardage fourth downs are much more likely to be predicted as failures.**

The tree gives us a simple way to visualize how multiple game situations work together to predict a fourth-down conversion. 

### Best Path to Take

The best path through the tree starts with 2.5 yards or fewer to go, where the model predicts a conversion. From there, being within 20 yards of the opponent's goal line continues to favor a conversion. Overall, the tree suggests that short-yardage situations, especially near the goal line, are the best situations to go for it on fourth down.  

## Comparing the Models

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Baseline | 0.545 | 0.545 | 1.000 | 0.705 |
| Logistic Regression | 0.650 | 0.642 | 0.808 | 0.716 |
| Decision Tree (tuned) | 0.643 | 0.675 | 0.665 | 0.670 |

The Logistic Regression model performed the best overall. It had the highest accuracy at 65.0%, the highest recall at 80.8%, and the highest F1 score at 0.716. The Decision Tree had slightly lower accuracy at 64.3%, but it had the highest precision at 67.5%. Both models performed better than the baseline accuracy of 54.5%.

Overall, Logistic Regression is the better model for predicting fourth-down conversions, while the Decision Tree is more useful for understanding which situations lead to a conversion.  

### Confusion Matrices

The confusion matrices show how each model's predictions compare to the actual fourth-down outcomes.

![Confusion Matrices](https://github.com/user-attachments/assets/a130be78-84da-44be-a80d-a8a48a1c8ca6)  

![Confusion Matrices](https://github.com/user-attachments/assets/303b10e4-7045-4e42-8ad0-fe759279d227)


Logistic Regression correctly identified **726 of 898 converted plays (81%)**, but it only correctly identified **345 of 750 failed plays (46%)**. This shows that the model is much better at identifying conversions than failures.

The Decision Tree correctly identified **597 of 898 converted plays (66%)** and **462 of 750 failed plays (62%)**. This makes the Decision Tree more balanced between predicting conversions and failures.

Overall, Logistic Regression is better at finding conversions, while the Decision Tree does a better job of identifying failed fourth downs. This matches the model comparison, where Logistic Regression had the higher overall accuracy and recall, while the Decision Tree had higher precision.  


### What the Model Learned

![What the models learned](https://github.com/user-attachments/assets/841353d3-e4cb-4aee-972c-e93acff5860e)

The Logistic Regression coefficients show which variables had the biggest influence on the model's predictions. The largest effects came from `ydstogo` and `yardline_100`.

`ydstogo` had the largest negative coefficient, meaning that as the number of yards needed for a first down increases, the likelihood of a conversion decreases. This makes sense because longer fourth downs are harder to convert.

`yardline_100` had the largest positive coefficient. Since this variable measures the distance to the opponent's goal line, a smaller value means the offense is closer to scoring. The model therefore found field position to be an important factor when predicting fourth-down conversions.

Other variables, such as time remaining, win probability, and timeouts, had smaller effects on the prediction. Overall, the model learned that distance to the first down and field position were the most important factors in predicting whether a fourth down would be converted.  

## Most important factors in the model

![Most important factors](https://github.com/user-attachments/assets/f89b56f3-20f7-4125-9cfc-b14d5fab3627)  

`ydstogo` seems to the be most important feature in the models. Which makes sense because most first down conversions come from 4th and 1, or 4th and 2s. 




