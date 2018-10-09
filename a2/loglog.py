import hashlib
import random
import sys
import json
import numpy as np
maxInt = sys.maxsize
minInt = -sys.maxsize -1

def myHash(astring, tablesize):
    sum = 0
    for pos in range(len(astring)):
        sum = sum + ord(astring[pos])

    return sum%tablesize

def countTrailingZeros(num):
  """Counts the number of trailing 0 bits in num."""
  if num == 0:
    return 32 # Assumes 32 bit integer inputs!
  p = 0
  while (num >> p) & 1 == 0:
    p += 1
  return p

def loglog(values, k):
  md5 = hashlib.sha1()
  m = 2 ** k
  M = [0] * m
  for value in values:
    md5.update(str(value).encode("utf-8"))
    h = int(md5.hexdigest(), 16)
    bucketId = h & (m - 1) # Mask out the k least significant bits as bucket ID
    bucketHash = h >> k
    M[bucketId] = max(M[bucketId], countTrailingZeros(bucketHash))
  sumOfM = 0
  for bucketVal in M:
    sumOfM += bucketVal
  return 2 ** (float(sum(M)) / m) * m * 0.79402

def noHashLogLog(values, k):
  m = 2 ** k
  M = [0] * m
  for value in values:
    bucketId = value & (m - 1) # Mask out the k least significant bits as bucket ID
    bucketHash = value >> k
    M[bucketId] = max(M[bucketId], countTrailingZeros(bucketHash))
  return 2 ** (float(sum(M)) / m) * m * 0.79402

def relativeApproximationError(estimate, true):
  return abs(true - estimate) / true

def generateRandomInput(amount):
  randomNums = []
  added = 0
  while added < amount:
    r = random.randint(minInt, maxInt)
    if r not in randomNums:
      added += 1
      randomNums.append(r)
  return randomNums

def doBasicExperiment():
  experimentResults = {}
  for trueCount in [250, 1000, 5000, 10000, 50000, 100000]:
    print("experimenting with count", trueCount)
    countResults = []
    for cardinality in [2, 4, 6, 8, 10, 12]:
      print("experimenting with cardinalitry", cardinality)
      settingResults = []
      for settingIteration in range(10):
        estimation = noHashLogLog(generateRandomInput(trueCount), cardinality)
        settingResults.append(relativeApproximationError(estimation, trueCount))
      countResults.append(np.mean(settingResults))
    experimentResults[str(trueCount)] = countResults
  print("results are ", experimentResults)
  file = open("results_selfchosen.json", "w")
  file.write(json.dumps(experimentResults))
  file.close()
doBasicExperiment()