from getData import getRatings, allNumericRatings, getRatingsForMovie, getRatingsForUser, getRatingSets
import numpy as np

def getMeanGlobalRatingSlow(ratings=getRatings()):
  total = 0.0
  amount = 0.0
  for rating in ratings:
    total += rating["rating"]
    amount += 1
  return total / amount

def getMeanGlobalRatingFast():
  ratings = np.array(allNumericRatings())
  return np.mean(ratings)

def getMeanRatingForItem(movieId, ratings=getRatings()):
  movieRatings = getRatingsForMovie(movieId, ratings)
  total = 0.0
  amount = 0.0
  for rating in movieRatings:
    total += rating["rating"]
    amount += 1
  return total / amount

def getMeanRatingForUser(userId, ratings=getRatings()):
  movieRatings = getRatingsForUser(userId, ratings)
  total = 0.0
  amount = 0.0
  for rating in movieRatings:
    total += rating["rating"]
    amount += 1
  return total / amount

def getUserItemRecommendation(userId, movieId, ratings=getRatings()):
  alpha = 0.3
  beta = 0.3
  gamma = 0.3
  alphaEstimates = []
  betaEstimates = []
  gammaEstimates = []
  actual = []
  meanUserRating = getMeanRatingForUser(userId, ratings)
  meanItemRating = getMeanRatingForItem(movieId, ratings)
  meanGlobal = getMeanGlobalRatingSlow(ratings)
  for rating in ratings:
    actual.append(rating["rating"])
    alphaEstimates.append(meanUserRating)
    betaEstimates.append(meanItemRating)
    gammaEstimates.append(meanGlobal)
  alpha = np.polyfit(alphaEstimates, actual, 1)[0]
  beta = np.polyfit(betaEstimates, actual, 1)[0]
  gamma = np.polyfit(gammaEstimates, actual, 1)[0]
  return alpha * meanUserRating + beta * meanItemRating + gamma
