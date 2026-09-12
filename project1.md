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

<p align="center" style="text-align:center;">  
    <img width="571" height="432" alt="image" src="https://github.com/user-attachments/assets/aaa3c6ac-c827-4359-bba4-b8b1a23a3476" />
</p>  
The next graph is to show bunts by inning. This gives a feel of when bunting is most common for players. I initially expected it to be very populated at the 7-9th inning mark. However there is a very notable spike in the 3rd inning. This is due to the National League requiring pitchers to hit pre 2022. And as stated before, pitchers bunted way more than other players. They were at the bottom of the line up which would come around during the 3rd inning most of the time. This graph lets us know that, outside the outlier of the third inning, the back 3rd of the game was where bunts were common. Which makes sense, in close, late games, bunts may be valuable.   

I followed these graphs with scenario specific graphs to see how bunting effects WPA 

<p align="center" style="text-align:center;">  
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/481c3f65-aafd-4a33-8a37-199296e3c079" />  
</p>  

This plot contains 3 of the most common times to bunt, as well as a fourth scenario. The only increase to WPA is in the runner on second and zero outs base out state. This is most likely due to advancing the runner to 3rd on a successful bunt. The lowest WPA is with our runner on first, one out bar. This is most likely due to double plays being turned more frequently off these bunts, and ending the inning. 

Even though non-bunts decrease WPA, this graph demonstrates that bunting tends to plummet the batting teams WPA way more than not bunting.   

<p align="center" style="text-align:center;">  
    <img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/ea537a4a-3a5f-4798-b52c-87aead105ed3" />
</p>  
This graph shows that bunting actually can help score runs in the right scenarios. With runners on first and second, a sac bunt will advance them, and potentially leading to 2 further runs at the cost of an out. But this is the core issue, it costs an out. Even though run percentage is high, your opponent could still outscore you in less outs.  

<p align="center" style="text-align:center;">  
    <img width="776" height="590" alt="image" src="https://github.com/user-attachments/assets/8e879ef3-e262-42dd-ab7d-b0fc347e6e6e" />
    <img width="776" height="590" alt="image" src="https://github.com/user-attachments/assets/96d23e70-e318-42de-af60-84c13fc1edeb" />
</p>  

These final two graphs demonstrate the differences in WPA between bunting and not bunting. Ignoring the obvious outliers with very small samples, the most notable square is the runners on first and second with no outs. Bunting increase a teams WPA very very slightly. Where as hitting decreases the WPA very slightly. We continue to see the double play trend with the 1 out runner on first for bunting, as it decreases the WPA. 

**Story and interpretation**  
The visual analysis of our bunting data shows that bunting may have a slight negative impact on games. But I believe that there is not enough evidence to show that bunting will help MLB teams increase their WPA. It seems bunting is only useful, from the data I have, in one specific scenario. Runners on first and second with no outs. Mostly in later inning, close scored games. As the "27 out clock" is shortened in the 7th, 8th, and 9th innings. Meaning sacrificing an out this late into the game could be worth it as your opponent has less time to catch up. Overall, I wished I could pull a larger sample to test with. The ~780 bunts I had to work with were enough to make a claim, but I could have missed out on some trends from the regular season.  

**Limitations, ethics, and final closing thoughts**  
Being limited by time constrained me to not downloading every regular season for this project as well. Leaving my laptop running for 15 hours straight did not seem like the best idea, however I feel it is safe to assume that my results would have been similar, despite only having post season games. One of my favorite things about this project was learning about trends in the MLB. How bunting has been on the decline, how pitchers used to bunt extremely often, and how there are scenarios where bunting could be useful. Ethically, I do not know how much of a gray line using the GitHub wrapper was for this project. Technically I am supposed to be affiliated with an MLB or minor league team to have access to this data. But this wrapper has allowed me to use it and conduct projects of my own. Overall I am very proud of my very first data science focused project, and I am excited for the future.   

AI DISCLAMER: To be transparent, I used Claude to write the code for creating my dataframe. It pulled the data from the api and put them into a csv for me. However the graphs are my own.  

    References
[GitHub MLB API wrapper](https://github.com/toddrob99/MLB-StatsAPI)
[Baseball Reference](https://www.baseball-reference.com/)

    Code repository



