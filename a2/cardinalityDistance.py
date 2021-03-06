from loglog import noHashLogLog, relativeApproximationError, generateRandomInput
import numpy as np
import json

def generateNumbersToTryByCardinality(cardinality):
  numbersToTry = []
  test = []
  actualNumber = 2 ** cardinality
  for isAddition in [False, True]:
    for i in range(81):
      if i > 0:
        number = -1
        if isAddition:
          number = actualNumber + actualNumber * (i / 100.0)
        else:
          number = actualNumber - actualNumber * (i / 100.0)
        if number > 0:
          numbersToTry.append(int(round(number)))
        else:
          numbersToTry.append(0)
        test.append(i)
        if i == 80 and not isAddition:
          numbersToTry.reverse()
          numbersToTry.append(actualNumber)
          test.reverse()
          test.append(0)
  return numbersToTry

cardinalitiesToTry = []

experimentResults = {}
for cardinality in range(13):
  print("experimenting with cardinalitry", cardinality, len(generateNumbersToTryByCardinality(cardinality)))
  countResults = []
  for trueCount in generateNumbersToTryByCardinality(cardinality):
    if trueCount != 0:
      print("experimenting with count", trueCount)
      settingResults = []
      for settingIteration in range(10):
        estimation = noHashLogLog(generateRandomInput(trueCount), cardinality)
        settingResults.append(relativeApproximationError(estimation, trueCount))
      countResults.append(np.mean(settingResults))
    else:
      countResults.append(0)
    print("countres is", len(countResults))
    experimentResults[str(cardinality)] = countResults
print(json.dumps(experimentResults))
file = open("results_all.json", "w")
file.write(json.dumps(experimentResults))
file.close()