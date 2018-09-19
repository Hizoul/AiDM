from naiveUtil import getMeanGlobalRatingSlow
from getData import getRatingSets, rmse
from sklearn import metrics
import numpy as np
import time
import json

sets = getRatingSets()


toTest = [
  {
    "func": getMeanGlobalRatingSlow,
    "name": "Global Average",
    "rmses": [],
    "maes": [],
    "ratings": [],
    "runtime": []
  }
]

rmseForMeanRating = []
maeForMeanRating = []
print("going through subsets")
for subSet in sets:
  for algo in toTest:
    start = time.time()
    algo["ratings"] = []
    meanRating = []
    # for each test entry, estimate rating using training set
    i = 0
    for testData in subSet["test"]:
      print("testset calc ", i)
      i += 1
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