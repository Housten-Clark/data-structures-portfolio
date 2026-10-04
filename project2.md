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
