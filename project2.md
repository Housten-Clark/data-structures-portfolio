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
