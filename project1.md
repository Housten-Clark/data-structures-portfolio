# Does bunting in the MLB improve a teams chances of winning, verses actually swinging at the ball?
> September 12th, 2026

**Problem Definition**  
Bunting in the MLB has been a hot topic for the past 20 years, as we have seen a sharp decline in the number of bunts throughout each season. In baseball as a whole, you are racing one thing, the 27 out clock. bunting is a way in which you "spend" (or cost) your team an out in the hope of advancing a runner to hopefully score for your team. The problem to many MLB teams is that spending one of your 27 outs is almost never worth it. Coaches would much rather have players swing away and drive runners in without costing themselves an out. However, change in baseball usually comes relatively slow. For the most of its lifetime, coaches have been going off their "gut feeling" of what is right. But the one thing that changes the game continuously, is the statistics. In this project I hope to show why bunting has declined as a whole in the MLB at the highest stage, in the playoffs.

**Data Description**
The main data set used for this project is the [Todd rob MLB API GitHub wrapper](https://github.com/toddrob99/MLB-StatsAPI). I am currently not affiliated with any MLB, or Minor league baseball teams so I do not have direct access to the MLB's API. However this wrapper allows me to obtain data and perform the experiments necessary to answer the core question if bunting is worth it in the MLB. I have pulled every at bat from the past 20 years of playoff games in the MLB, put them into a csv and used as a data frame to answer our question. This data set contains a wide variety of entries, but for this project we are only concerned with 18 variables. Those being contained in the list below: 
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

