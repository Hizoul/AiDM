import numpy as np 
import random 
import sys
maxInt = sys.maxsize
minInt = -sys.maxsize -1
import time
import json


def trailing_zeroes(num):
  """Counts the number of trailing 0 bits in num."""
  if num == 0:
    return 32 # Assumes 32 bit integer inputs!
  p = 0
  while (num >> p) & 1 == 0:
    p += 1
  return p

def fm(values):
	R = 0 #R = maximum number of trailing zeros
	for value in values:
		r = value
		R = max(R, trailing_zeroes(r)) #max tail length
	return 2**R

def flajoletMartin(values, iterations):
  hash_map = int(2*np.log2(len(values)))
  mean_bucket = [] # mean of the bucket 
  for j in range(iterations):
    hc = [] 
    for k in range(hash_map):
      hc.append(fm(values))
    mean_bucket.append(np.median(hc))
  return np.mean(mean_bucket)

def generateRandomInput(amount):
  randomNums = []
  added = 0
  while added < amount:
    r = random.randint(0, maxInt)
    if r not in randomNums:
      added += 1
      randomNums.append(r)
  return randomNums

def relativeApproximationError(estimate, true):
  return abs(true - estimate) / true

def flajoletMartinWithCorrection(values, nMap, maxLength):
  CORRECTION_CONST = 0.77351
  bitmap = np.zeros((nMap, maxLength), dtype=np.int)
  for value in values:
    a = value % nMap
    ix = bin(int(value / nMap))[2:][::-1]
    indexBeta = ix.find("1")
    if bitmap[a, indexBeta] == 0:
      bitmap[a, indexBeta] = 1
  sum = 0
  for row in range(nMap):
    sum += np.where(bitmap[row, :] == 0)[0][0]
  A = sum / nMap
  return nMap * (2**A) / CORRECTION_CONST

def multiples(x):
  arr = []
  for i in range(x):
    arr.append(2 ** i)
  return arr

# index0 = nMap index1 = maxLength
experimentSettings = [
  [64, 32],
  [128, 64],
  [256, 128],
  [512, 256],
  [1024, 512]
]

def doBasicExperiment():
  experimentResults = {}
  for setting in experimentSettings:
    countResults = []
    for trueCount in multiples(16):
      print("experimenting with count", trueCount)
      settingResults = []
      for settingIteration in range(10):
        estimation = flajoletMartinWithCorrection(generateRandomInput(trueCount), setting[0], setting[1])
        print("estimation is", estimation)
        settingResults.append(relativeApproximationError(estimation, trueCount))
      countResults.append(np.mean(settingResults))
    experimentResults[json.dumps(setting)] = countResults
  print(experimentResults)
  file = open("results_all_fm_with_correction.json", "w")
  file.write(json.dumps(experimentResults))
  file.close()

doBasicExperiment()