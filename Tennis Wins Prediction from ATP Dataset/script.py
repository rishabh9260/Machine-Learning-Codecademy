import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv("tennis_stats.csv")
# print(df.head())
service_features = ["Aces", "DoubleFaults", "FirstServe", "FirstServePointsWon", "SecondServePointsWon", "BreakPointsFaced", "BreakPointsSaved", "ServiceGamesPlayed", "ServiceGamesWon", "TotalServicePointsWon"]
return_features = ["FirstServeReturnPointsWon", "SecondServeReturnPointsWon","BreakPointsOpportunities", "BreakPointsConverted", "ReturnGamesPlayed", "ReturnGamesWon", "ReturnPointsWon", "TotalPointsWon"]
outcomes = ["Wins", "Losses", "Winnings", "Ranking"]

all_features = service_features + return_features

# perform exploratory analysis here:
# For printing all the graphs uncomment the below block

# for i in range(len(service_features)):
#   for j in range(len(outcomes)):
#     plt.figure()
#     plt.scatter(df[service_features[i]], df[outcomes[j]], alpha=0.4)
#     plt.xlabel(service_features[i])
#     plt.ylabel(outcomes[j])
#     plt.show()

# for i in range(len(return_features)):
#   for j in range(len(outcomes)):
#     plt.figure()
#     plt.scatter(df[return_features[i]], df[outcomes[j]], alpha=0.4)
#     plt.xlabel(return_features[i])
#     plt.ylabel(outcomes[j])
#     plt.show()

# Correlation between the data
def display_corr(service_features, outcomes):
  a = df[service_features+outcomes]
  b = df[return_features+outcomes]
  corrmat = a.corr() 
  corrmat2 = b.corr()
  mask1 = np.zeros_like(corrmat)
  for i in range(a.shape[1]):
    for j in range(a.shape[1]):
      mask1[i, j] = [False, True][j<10 or i>=10]
  mask2 = np.zeros_like(corrmat2)
  for i in range(b.shape[1]):
    for j in range(b.shape[1]):
      mask2[i, j] = [False, True][j<8 or i>=8]
  f, ax = plt.subplots(figsize =(9, 8)) 
  sns.heatmap(corrmat, ax = ax, cmap ="rocket_r", linewidths = 0.3, annot = True, fmt = ".2f", mask=mask1) 
  plt.show()
  f, ax = plt.subplots(figsize =(9, 8)) 
  sns.heatmap(corrmat2, ax = ax, cmap ="rocket_r", linewidths = 0.3, annot = True, fmt = ".2f", mask=mask2) 
  plt.show()

#display_corr(service_features, outcomes)

best_correlated_features = ["DoubleFaults", "BreakPointsFaced", "ServiceGamesPlayed", "BreakPointsOpportunities", "ReturnGamesPlayed"]
useful_outcomes = ["Wins", "Lossed", "Winnings"]

## Defining linearReg function that splits and models the passed data with score analysis

def linearReg(x, y, xlabel, ylabel):
  if len(x.shape) < 2:
    x = np.array(x).reshape(-1,1)

  reg = LinearRegression()
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
  reg.fit(X_train, y_train)
  y_predict = reg.predict(X_test)
  print(reg.score(X_test, y_test), xlabel + " vs. " + ylabel)
  plt.figure()
  plt.grid(True)
  plt.scatter(y_predict, y_test, alpha=0.4)
  plt.xlabel("Predicted $Y_i$")
  plt.ylabel("Actual $Y_i$")
  plt.title(xlabel + " vs. " + ylabel)
  plt.show()

## perform single feature linear regressions here:

# for j in range(len(useful_outcomes)):
#   for i in range(len(all_features)):
#     linearReg(df[all_features[i]], df[useful_outcomes[j]], all_features[i], useful_outcomes[j])

## perform two feature linear regressions here:

# for j in range(len(useful_outcomes)):
#   for i in range(len(all_features)-2):
#     linearReg(df[all_features[i:i+2]], df[useful_outcomes[j]], " & ".join(df[all_features[i:i+2]]), useful_outcomes[j])

## perform multiple feature linear regressions here:
custom_features = ["Aces", "DoubleFaults", "FirstServe", "FirstServePointsWon", "SecondServePointsWon", "BreakPointsFaced", "BreakPointsSaved", "ServiceGamesPlayed", "ServiceGamesWon", "TotalServicePointsWon", "FirstServeReturnPointsWon", "SecondServeReturnPointsWon","BreakPointsOpportunities", "BreakPointsConverted", "ReturnGamesPlayed", "ReturnGamesWon", "ReturnPointsWon", "TotalPointsWon"]

#display_corr(service_features, outcomes)

for i in range(len(useful_outcomes)):
  linearReg(df[custom_features], df[outcomes[i]], "All Features", outcomes[i])

#custom_feature currently set to all_features
#since all the features have a positive correlation and 
#contibute to the accuracy or score of the model for useful_outcomes
#ranking is negatively correlated to all the data