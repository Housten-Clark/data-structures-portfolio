# In what situation does going for it on fourth down increase an NFL team's chances of winning the game?
> October 4th, 2026

**Problem Definition**   

Fourth down conversions are some of the most exciting and tense moment in football. Seeing your team convert in a high pressure situation to win the game is a testament to show how hype football can be. However, is going for it on 4th actually worth it? In which situation does going for it on 4th down improve the offensive team's chances of winning the game? In this project I hope to answer this question. Using two machine learning models to find the best situation to go for it on fourth down.   

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






