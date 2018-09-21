from naiveUtil import resetMeans, getMeanGlobalRatingSlow, getMeanRatingForItem, getMeanRatingForUser, getUserItemRecommendation
from getData import getRatings, getRatingSets, rmse
from sklearn import metrics
import numpy as np
import time
import json
from naiveUtil import prepLinReg

ratings = getRatings()
sets = getRatingSets(ratings)


toTest = [
{
    "func": getMeanGlobalRatingSlow,
    "name": "Global Mean",
    "rmses": [],
    "maes": [],
    "ratings": [],
    "runtime": []
  },
  {
    "func": getMeanRatingForItem,
    "name": "Item Mean",
    "rmses": [],
    "maes": [],
    "ratings": [],
    "runtime": []
  },
  {
    "func": getMeanRatingForUser,
    "name": "User Mean",
    "rmses": [],
    "maes": [],
    "ratings": [],
    "runtime": []
  },
  {
    "func": getUserItemRecommendation,
    "name": "UserItem Recommendation",
    "rmses": [],
    "maes": [],
    "ratings": [],
    "runtime": []
  }
]

rmseForMeanRating = []
maeForMeanRating = []
print("going through subsets")
i = 0
for subSet in sets:
  resetMeans()
  i += 1
  print("in subset ", i)
  for algo in toTest:
    print("testing algo ", algo["name"])
    start = time.time()
    algo["ratings"] = []
    meanRating = []
    # for each test entry, estimate rating using training set
    for testData in subSet["test"]:
      algo["ratings"].append(algo["func"](subSet["train"], testData["movieId"], testData["userId"]))
    # get ratings
    # compare error
    meanRatingError = rmse(np.array(subSet["testNumOnly"]), np.array(algo["ratings"]))
    absoluteRatingError = metrics.mean_absolute_error(np.array(subSet["testNumOnly"]), np.array(algo["ratings"]))
    algo["rmses"].append(meanRatingError)
    algo["maes"].append(absoluteRatingError)
    algo["runtime"].append(time.time() - start)
  # save error

print("rmses are", toTest[0]["rmses"], np.mean(toTest[0]["rmses"]), np.mean(toTest[0]["runtime"]))
# do the same for MAE

for algo in toTest:
  algo["ratings"] = None
  algo["func"] = None
print(json.dumps(toTest))