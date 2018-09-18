from naiveUtil import getMeanGlobalRatingSlow
from getData import getRatingSets


sets = getRatingSets()


# take set
rmseForMeanRating = []
for subSet in sets:
  meanRating = []
  # for each test entry, estimate rating using training set
  for testData in subSet["test"]:
    meanRating.append(getMeanGlobalRatingSlow(subSet["train"]))
  
  # get ratings
  # compare error
  meanRatingError = rmse(subSet.test, meanRating)
  print("mean error is", meanRatingError)
  rmseForMeanRating.append(meanRatingError)
  # save error

print("the mean rating rmses are", rmseForMeanRating, np.mean(rmseForMeanRating))
# do the same for MAE