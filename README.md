David Sheehan	
David Sheehan
Data scientist interested in sports, politics and Simpsons references

Follow
Football (or soccer to my American readers) is full of clichés: “It’s a game of two halves”, “taking it one game at a time” and “Liverpool have failed to win the Premier League”. You’re less likely to hear “Treating the number of goals scored by each team as independent Poisson processes, statistical modelling suggests that the home team have a 60% chance of winning today”. But this is actually a bit of cliché too (it has been discussed here, here, here, here and particularly well here). As we’ll discover, a simple Poisson model is, well, overly simplistic. But it’s a good starting point and a nice intuitive way to learn about statistical modelling. So, if you came here looking to make money, I hear this guy makes £5000 per month without leaving the house.

Poisson Distribution
The model is founded on the number of goals scored/conceded by each team. Teams that have been higher scorers in the past have a greater likelihood of scoring goals in the future. We’ll import all match results from the recently concluded Premier League (2016/17) season. There’s various sources for this data out there (kaggle, football-data.co.uk, github, API). I built an R wrapper for that API, but I’ll go the csv route this time around.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam

epl_1617 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1617/E0.csv")
epl_1617 = epl_1617[['HomeTeam','AwayTeam','FTHG','FTAG']]
epl_1617 = epl_1617.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
epl_1617.head()
HomeTeam	AwayTeam	HomeGoals	AwayGoals
0	Burnley	Swansea	0	1
1	Crystal Palace	West Brom	0	1
2	Everton	Tottenham	1	1
3	Hull	Leicester	2	1
4	Man City	Sunderland	2	1
We imported a csv as a pandas dataframe, which contains various information for each of the 380 EPL games in the 2016-17 English Premier League season. We restricted the dataframe to the columns in which we’re interested (specifically, team names and numer of goals scored by each team). I’ll omit most of the code that produces the graphs in this post. But don’t worry, you can find that code on my github page. Our task is to model the final round of fixtures in the season, so we must remove the last 10 rows (each gameweek consists of 10 matches).

epl_1617 = epl_1617[:-10]
epl_1617.mean()
HomeGoals    1.591892
AwayGoals    1.183784
dtype: float64
You’ll notice that, on average, the home team scores more goals than the away team. This is the so called ‘home (field) advantage’ (discussed here) and isn’t specific to soccer. This is a convenient time to introduce the Poisson distribution. It’s a discrete probability distribution that describes the probability of the number of events within a specific time period (e.g 90 mins) with a known average rate of occurrence. A key assumption is that the number of events is independent of time. In our context, this means that goals don’t become more/less likely by the number of goals already scored in the match. Instead, the number of goals is expressed purely as function an average rate of goals. If that was unclear, maybe this mathematical formulation will make clearer:

P(x)=e−λλxx!,λ>0
λ
 represents the average rate (e.g. average number of goals, average number of letters you receive, etc.). So, we can treat the number of goals scored by the home and away team as two independent Poisson distributions. The plot below shows the proportion of goals scored compared to the number of goals estimated by the corresponding Poisson distributions.



We can use this statistical model to estimate the probability of specfic events.

P(≥2|Home)=P(2|Home)+P(3|Home)+...=0.258+0.137+...=0.47
The probability of a draw is simply the sum of the events where the two teams score the same amount of goals.

P(Draw)=P(0|Home)×P(0|Away)+P(1|Home)×P(1|Away)+...=0.203×0.306+0.324×0.362+...=0.248
Note that we consider the number of goals scored by each team to be independent events (i.e. P(A n B) = P(A) P(B)). The difference of two Poisson distribution is actually called a Skellam distribution. So we can calculate the probability of a draw by inputting the mean goal values into this distribution.

# probability of draw between home and away team
skellam.pmf(0.0,  epl_1617.mean()[0],  epl_1617.mean()[1])
0.24809376810717076
# probability of home team winning by one goal
skellam.pmf(1,  epl_1617.mean()[0],  epl_1617.mean()[1])
0.22558259663675409


So, hopefully you can see how we can adapt this approach to model specific matches. We just need to know the average number of goals scored by each team and feed this data into a Poisson model. Let’s have a look at the distribution of goals scored by Chelsea and Sunderland (teams who finished 1st and last, respectively).



Building A Model
You should now be convinced that the number of goals scored by each team can be approximated by a Poisson distribution. Due to a relatively sample size (each team plays at most 19 home/away games), the accuracy of this approximation can vary significantly (especially earlier in the season when teams have played fewer games). Similar to before, we could now calculate the probability of various events in this Chelsea Sunderland match. But rather than treat each match separately, we’ll build a more general Poisson regression model (what is that?).

# importing the tools required for the Poisson regression model
import statsmodels.api as sm
import statsmodels.formula.api as smf

goal_model_data = pd.concat([epl_1617[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
            columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
           epl_1617[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})])

poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()
poisson_model.summary()
Generalized Linear Model Regression Results
Dep. Variable:	goals	No. Observations:	740
Model:	GLM	Df Residuals:	700
Model Family:	Poisson	Df Model:	39
Link Function:	log	Scale:	1.0
Method:	IRLS	Log-Likelihood:	-1042.4
Date:	Sat, 10 Jun 2017	Deviance:	776.11
Time:	11:17:38	Pearson chi2:	659.
No. Iterations:	8		
coef	std err	z	P>|z|	[95.0% Conf. Int.]
Intercept	0.3725	0.198	1.880	0.060	-0.016 0.761
team[T.Bournemouth]	-0.2891	0.179	-1.612	0.107	-0.641 0.062
team[T.Burnley]	-0.6458	0.200	-3.230	0.001	-1.038 -0.254
team[T.Chelsea]	0.0789	0.162	0.488	0.626	-0.238 0.396
team[T.Crystal Palace]	-0.3865	0.183	-2.107	0.035	-0.746 -0.027
team[T.Everton]	-0.2008	0.173	-1.161	0.246	-0.540 0.138
team[T.Hull]	-0.7006	0.204	-3.441	0.001	-1.100 -0.302
team[T.Leicester]	-0.4204	0.187	-2.249	0.025	-0.787 -0.054
team[T.Liverpool]	0.0162	0.164	0.099	0.921	-0.306 0.338
team[T.Man City]	0.0117	0.164	0.072	0.943	-0.310 0.334
team[T.Man United]	-0.3572	0.181	-1.971	0.049	-0.713 -0.002
team[T.Middlesbrough]	-1.0087	0.225	-4.481	0.000	-1.450 -0.568
team[T.Southampton]	-0.5804	0.195	-2.976	0.003	-0.963 -0.198
team[T.Stoke]	-0.6082	0.197	-3.094	0.002	-0.994 -0.223
team[T.Sunderland]	-0.9619	0.222	-4.329	0.000	-1.397 -0.526
team[T.Swansea]	-0.5136	0.192	-2.673	0.008	-0.890 -0.137
team[T.Tottenham]	0.0532	0.162	0.328	0.743	-0.265 0.371
team[T.Watford]	-0.5969	0.197	-3.035	0.002	-0.982 -0.211
team[T.West Brom]	-0.5567	0.194	-2.876	0.004	-0.936 -0.177
team[T.West Ham]	-0.4802	0.189	-2.535	0.011	-0.851 -0.109
opponent[T.Bournemouth]	0.4109	0.196	2.092	0.036	0.026 0.796
opponent[T.Burnley]	0.1657	0.206	0.806	0.420	-0.237 0.569
opponent[T.Chelsea]	-0.3036	0.234	-1.298	0.194	-0.762 0.155
opponent[T.Crystal Palace]	0.3287	0.200	1.647	0.100	-0.062 0.720
opponent[T.Everton]	-0.0442	0.218	-0.202	0.840	-0.472 0.384
opponent[T.Hull]	0.4979	0.193	2.585	0.010	0.120 0.875
opponent[T.Leicester]	0.3369	0.199	1.694	0.090	-0.053 0.727
opponent[T.Liverpool]	-0.0374	0.217	-0.172	0.863	-0.463 0.389
opponent[T.Man City]	-0.0993	0.222	-0.448	0.654	-0.534 0.335
opponent[T.Man United]	-0.4220	0.241	-1.754	0.079	-0.894 0.050
opponent[T.Middlesbrough]	0.1196	0.208	0.574	0.566	-0.289 0.528
opponent[T.Southampton]	0.0458	0.211	0.217	0.828	-0.369 0.460
opponent[T.Stoke]	0.2266	0.203	1.115	0.265	-0.172 0.625
opponent[T.Sunderland]	0.3707	0.198	1.876	0.061	-0.017 0.758
opponent[T.Swansea]	0.4336	0.195	2.227	0.026	0.052 0.815
opponent[T.Tottenham]	-0.5431	0.252	-2.156	0.031	-1.037 -0.049
opponent[T.Watford]	0.3533	0.198	1.782	0.075	-0.035 0.742
opponent[T.West Brom]	0.0970	0.209	0.463	0.643	-0.313 0.507
opponent[T.West Ham]	0.3485	0.198	1.758	0.079	-0.040 0.737
home	0.2969	0.063	4.702	0.000	0.173 0.421
If you’re curious about the smf.glm(...) part, you can find more information here (edit: earlier versions of this post had erroneously employed a Generalised Estimating Equation (GEE)- what’s the difference?). I’m more interested in the values presented in the coef column in the model summary table, which are analogous to the slopes in linear regression. Similar to logistic regression, we take the exponent of the parameter values. A positive value implies more goals (ex>1∀x>0
), while values closer to zero represent more neutral effects (e0=1
). Towards the bottom of the table you might notice that home has a coef of 0.2969. This captures the fact that home teams generally score more goals than the away team (specifically, e0.2969
=1.35 times more likely). But not all teams are created equal. Chelsea has a coef of 0.0789, while the corresponding value for Sunderland is -0.9619 (sort of saying Chelsea (Sunderland) are better (much worse!) scorers than average). Finally, the opponent* values penalize/reward teams based on the quality of the opposition. This relfects the defensive strength of each team (Chelsea: -0.3036; Sunderland: 0.3707). In other words, you’re less likely to score against Chelsea. Hopefully, that all makes both statistical and intuitive sense.

Let’s start making some predictions for the upcoming matches. We simply pass our teams into poisson_model and it’ll return the expected average number of goals for that team (we need to run it twice- we calculate the expected average number of goals for each team separately). So let’s see how many goals we expect Chelsea and Sunderland to score.

poisson_model.predict(pd.DataFrame(data={'team': 'Chelsea', 'opponent': 'Sunderland',
                                       'home':1},index=[1]))
array([ 3.06166192])
poisson_model.predict(pd.DataFrame(data={'team': 'Sunderland', 'opponent': 'Chelsea',
                                       'home':0},index=[1]))
array([ 0.40937279])
Just like before, we have two Poisson distributions. From this, we can calculate the probability of various events. I’ll wrap this in a simulate_match function.

def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam,'home':1},
                                                      index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam,'home':0},
                                                      index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
simulate_match(poisson_model, 'Chelsea', 'Sunderland', max_goals=3)
