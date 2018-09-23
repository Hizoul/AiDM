from getData import getRatings, allNumericRatings, getRatingsForMovie, getRatingsForUser, getRatingSets
import numpy as np
import warnings
warnings.simplefilter('ignore', np.RankWarning)

calculatedMeans = {}
calculatedItemMeans = {}
calculatedUserMeans = {}

def resetMeans():
  global calculatedMeans
  global calculatedItemMeans
  global calculatedUserMeans
  calculatedMeans = {}
  calculatedItemMeans = {}
  calculatedUserMeans = {}

# Runtime: O(n)
# Memory: O(n)
def getMeanGlobalRatingSlow(ratings, movieId, userId):
  global calculatedMeans
  key = str(ratings[0]["userId"]) + str(ratings[0]["movieId"])
  if calculatedMeans.get(key) is None:
    total = 0.0
    amount = 0.0
    for rating in ratings:
      total += rating["rating"]
      amount += 1
    mean = total / amount
    calculatedMeans[key] = mean
    return mean
  else:
    return calculatedMeans[key]

def getMeanGlobalRatingFast():
  ratings = np.array(allNumericRatings())
  return np.mean(ratings)


# Worst Case Runtime: O(n^2)
# At least once n and in the worst case all ratings are for one movie hence n^2
# Memory: O(n^2)
def getMeanRatingForItem(ratings, movieId, userId):
  global calculatedItemMeans
  key = str(movieId) + str(ratings[0]["userId"]) + str(ratings[0]["movieId"])
  if calculatedItemMeans.get(key) is None:
    movieRatings = getRatingsForMovie(movieId, ratings)
    if len(movieRatings) == 0:
      return getMeanGlobalRatingSlow(ratings, movieId, userId)
    else:
      total = 0.0
      amount = 0.0
      for rating in movieRatings:
        total += rating["rating"]
        amount += 1
      itemMean = total / amount
      calculatedItemMeans[key] = itemMean
      return itemMean
  else:
    return calculatedItemMeans[key]

# Worst Case Runtime: O(n^2)
# At least once n and in the worst case all ratings are by one user hence n^2
# Memory: O(n^2)
def getMeanRatingForUser(ratings, movieId, userId):
  global calculatedUserMeans
  key = str(userId) + str(ratings[0]["userId"]) + str(ratings[0]["movieId"])
  if calculatedUserMeans.get(key) is None:
    userRatings = getRatingsForUser(userId, ratings)
    if len(userRatings) == 0:
      return getMeanGlobalRatingSlow(ratings, movieId, userId)
    else:
      total = 0.0
      amount = 0.0
      for rating in userRatings:
        total += rating["rating"]
        amount += 1
      userMean = total / amount
      calculatedUserMeans[key] = userMean
      return userMean
  else:
    return calculatedUserMeans[key]

alpha = 0.9999999999998403
beta = 1.000000000000228
gamma = 0.5000000000030178

# Worst Case Runtime: O(n^6)
# Combines user n^2 with movie n^2 and global n and has to iterate once over all entries hence n^6
# Memory: O(n^4)
def prepLinReg(ratings):
  print("about to prepare linreg params")
  global alpha
  global beta
  global gamma
  alphaEstimates = []
  betaEstimates = []
  gammaEstimates = []
  actual = []
  for rating in ratings:
    actual.append(rating["rating"])
    alphaEstimates.append(getMeanRatingForUser(ratings, rating["movieId"], rating["userId"]))
    betaEstimates.append(getMeanRatingForItem(ratings, rating["movieId"], rating["userId"]))
    gammaEstimates.append(getMeanGlobalRatingSlow(ratings, rating["movieId"], rating["userId"]))
  alpha = np.polyfit(alphaEstimates, actual, 1)[0]
  beta = np.polyfit(betaEstimates, actual, 1)[0]
  gamma = np.polyfit(gammaEstimates, actual, 1)[0]
  print("prepared linreg params", alpha, beta, gamma)


# Worst Case Runtime: O(n^5)
# Combines user n^2 with movie n^2 and global hence n^5
# Memory: O(n^2)
def getUserItemRecommendation(ratings, movieId, userId):
  global alpha
  global beta
  global gamma
  meanUserRating = getMeanRatingForUser(ratings, movieId, userId)
  meanItemRating = getMeanRatingForItem(ratings, movieId, userId)
  meanGlobal = getMeanGlobalRatingSlow(ratings, movieId, userId)
  return alpha * meanUserRating + beta * meanItemRating + gamma
