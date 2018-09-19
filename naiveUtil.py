from getData import getRatings, allNumericRatings, getRatingsForMovie, getRatingsForUser, getRatingSets
import numpy as np

calculatedMeans = {}
calculatedItemMeans = {}

def getMeanGlobalRatingSlow(ratings, movieId, userId):
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

def getMeanRatingForItem(ratings, movieId, userId):
  key = str(movieId) + str(ratings[0]["userId"]) + str(ratings[0]["movieId"])
  if calculatedItemMeans.get(key) is None:
    movieRatings = getRatingsForMovie(movieId, ratings)
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

def getMeanRatingForUser(ratings, movieId, userId):
  movieRatings = getRatingsForUser(userId, ratings)
  total = 0.0
  amount = 0.0
  for rating in movieRatings:
    total += rating["rating"]
    amount += 1
  return total / amount

def getUserItemRecommendation(ratings, movieId, userId):
  alpha = 0.3
  beta = 0.3
  gamma = 0.3
  alphaEstimates = []
  betaEstimates = []
  gammaEstimates = []
  actual = []
  meanUserRating = getMeanRatingForUser(ratings, movieId, userId)
  meanItemRating = getMeanRatingForItem(ratings, movieId, userId)
  meanGlobal = getMeanGlobalRatingSlow(ratings, movieId, userId)
  for rating in ratings:
    actual.append(rating["rating"])
    alphaEstimates.append(meanUserRating)
    betaEstimates.append(meanItemRating)
    gammaEstimates.append(meanGlobal)
  alpha = np.polyfit(alphaEstimates, actual, 1)[0]
  beta = np.polyfit(betaEstimates, actual, 1)[0]
  gamma = np.polyfit(gammaEstimates, actual, 1)[0]
  return alpha * meanUserRating + beta * meanItemRating + gamma
