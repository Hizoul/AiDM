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


# Worst Case Runtime: O(R)
# Memory: O(2)
def getMeanRatingForItem(ratings, movieId, userId):
  global calculatedItemMeans
  key = str(movieId) + str(ratings[0]["userId"]) + str(ratings[0]["movieId"])
  if calculatedItemMeans.get(key) is None:
    total = 0.0
    amount = 0.0
    for rating in ratings:
      if rating["movieId"] == movieId:
        total += rating["rating"]
        amount += 1
    if amount == 0:
      return getMeanGlobalRatingSlow(ratings, movieId, userId)
    else:
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
    total = 0.0
    amount = 0.0
    for rating in ratings:
      if rating["userId"] == userId:
        total += rating["rating"]
        amount += 1
    if total == 0:
      return getMeanGlobalRatingSlow(ratings, movieId, userId)
    else:
      userMean = total / amount
      calculatedUserMeans[key] = userMean
      return userMean
  else:
    return calculatedUserMeans[key]

alpha = 0.78212853
beta = 0.8757397
gamma = -2.35619748

def simple_linear_regression(X, y):
    '''
    Returns slope and intercept for a simple regression line
    
    inputs- Works best with numpy arrays, though other similar data structures will work fine.
        X - input data
        y - output data
        
    outputs - floats
    '''
    # initial sums
    n = float(len(X))
    sum_x = X.sum()
    sum_y = y.sum()
    sum_xy = (X*y).sum()
    sum_xx = (X**2).sum()
    
    # formula for w0
    slope = (sum_xy - (sum_x*sum_y)/n)/(sum_xx - (sum_x*sum_x)/n)
    
    # formula for w1
    intercept = sum_y/n - slope*(sum_x/n)
    
    return [intercept, slope]

# Worst Case Runtime: O(n^6)
# Combines user r*m with movie r*u and global r and has to iterate once over all entries
# optimizable to O(r)
# Memory: O(n^4)
def prepLinReg(ratings):
  print("about to prepare linreg params")
  global alpha
  global beta
  global gamma
  alphaEstimates = []
  betaEstimates = []
  actual = []
  for rating in ratings:
    actual.append(rating["rating"])
    alphaEstimates.append(getMeanRatingForUser(ratings, rating["movieId"], rating["userId"]))
    betaEstimates.append(getMeanRatingForItem(ratings, rating["movieId"], rating["userId"]))
  coefficients = np.linalg.lstsq(np.vstack([alphaEstimates, betaEstimates, np.ones(len(alphaEstimates))]).T, actual)
  print("prepared linreg params", coefficients)

# Worst Case Runtime: O(R)
# Memory: O(6)
def getUserItemRecommendation(ratings, movieId, userId):
  global alpha
  global beta
  global gamma
  meanUserRating = getMeanRatingForUser(ratings, movieId, userId)
  meanItemRating = getMeanRatingForItem(ratings, movieId, userId)
  meanGlobal = getMeanGlobalRatingSlow(ratings, movieId, userId)
  return alpha * meanUserRating + beta * meanItemRating + gamma
