# Does bunting in the MLB improve a teams chances of winning, verses actually swinging at the ball?
> September 12th, 2026

**Problem Definition**  
Bunting in the MLB has been a hot topic for the past 20 years, as we have seen a sharp decline in the number of bunts throughout each season. In baseball as a whole, you are racing one thing, the 27 out clock. bunting is a way in which you "spend" (or cost) your team an out in the hope of advancing a runner to hopefully score for your team. The problem to many MLB teams is that spending one of your 27 outs is almost never worth it. Coaches would much rather have players swing away and drive runners in without costing themselves an out. However, change in baseball usually comes relatively slow. For the most of its lifetime, coaches have been going off their "gut feeling" of what is right. But the one thing that changes the game continuously, is the statistics. In this project I hope to show why bunting has declined as a whole in the MLB at the highest stage, in the playoffs.

**Data Description**
The main data set used for this project is the [Todd rob MLB API GitHub wrapper](https://github.com/toddrob99/MLB-StatsAPI). I am currently not affiliated with any MLB, or Minor league baseball teams so I do not have direct access to the MLB's API. However this wrapper allows me to obtain data and perform the experiments necessary to answer the core question if bunting is worth it in the MLB. I have pulled every at bat from the past 20 years of playoff games in the MLB, put them into a csv and used as a data frame to answer our question. Why playoff games? Teams care about winning every playoff game, where as some regular season games are not as meaningful. As well as downloading every game in the past 20 years would have taken 15 hours straight. This data set contains a wide variety of entries, but for this project we are only concerned with 18 variables. Those being contained in the list below: 
- season for a general time frame
-  game_pk to categorize each game
-   innings
-   is_top_inning for various home and away team calculations
-   at_bat_index to show how many at bats are in each game
-   base_state_before for categorizing the 24 base out states
-   outs_before also for the 24 base out states
-   base_state_after
-   outs_after
-   is_bunt
-   bunt_outcome to see if the bunt was successful
-   raw_event to see what happend at the at bat
-   Description shows what happened in the play, used to confirm our data matches sources like baseball reference
-   runs_on_play
-   runs_after_play
-   runs_rest_of_inning
-   score_diff_before
-   home_team_wpa our primary predictor if a bunt helped the team

The most important variable to this project is the home teams WPA. This number is calculated by finding the difference between a team's chance of winning right after a specific play and their chance of winning right before that play. Using this we can see if the bunt helped a team or not. 

**Data cleaning and prep**  

For cleaning my data, all I needed to do was drop all null values. This was simply done with `df.dropna()`  
Next, to prepare my data, I created a state before variable to see all of the 24 base out states in one variable, rather than two.  
`df['state_before'] = df['base_state_before'].astype(str) + "-" + df['outs_before'].astype(str)`  

Then I checked the home teams wpa with `print(df[['home_team_wpa']].head(10))` then compared these values to the ones listed on baseball reference. After finding that they matched I knew my WPA was accurate.  

Finally, I created a batting_team_wpa variable, rather than using the home teams wpa. To do this I check if it is the top of the inning, if it is we flip the sign of the home teams wpa, if not leave it as is.  
`df['batting_team_wpa'] = df.apply(
    lambda row: row['home_team_wpa']
    if not row['is_top_inning']
    else -row['home_team_wpa'],
    axis=1
)`  

**Data visualization**  

In order to get initial context for bunts in the MLB, I created two main graphs. One to track bunts over time, then another to track bunts per inning.

<p align="center" style="text-align:center;">
    <img width="563" height="453" alt="image" src="https://github.com/user-attachments/assets/0b09d8dc-192d-4d57-ac6c-d5fa855e7187" />  
 

As we can see there is a decline in bunting over the past 20 years of playoff games. There is a very sharp decrease around 2019-2020, as well as seemingly bunting increasing again recently. At first I thought the decrease in 2020 was because the pandemic. However there were actually more games played than a regular postseason as the league expanded the bracket to 16 teams. However they also implemented a universal designated hitter for pitchers at the time. Meaning pitchers were not hitting, and therefore not bunting as much. (Pitchers tend to sac bunt way more than other players as they are worse hitters usually). 

</p> 


