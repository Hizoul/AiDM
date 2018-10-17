import numpy as np
from scipy.spatial.distance import cosine
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix, dok_matrix
from operator import itemgetter
import time
import hashlib
start = time.time()
np.random.seed(18)
origData = np.load("user_movie.npy")
print("LAST USER IS", origData[len(origData) -1])
origData = origData
data = origData
# threshold t is approx (1 / b)^(1/r)
bands = 5
rows = 25
signatureSize = 150
amountOfMovies = 17770
amountOfUsers = origData[len(origData)  - 1][0] + 1 # 103702
print("user amm is ", origData[len(origData)  - 1][0])
amountOfUsers = origData[len(origData)  - 1][0] + 1 # 103702
k = bands * rows

buckets = []
for bucketNr in range(k):
  buckets.append([])
# specify the length of each minhash vector
N = k
max_val = (2**32)-1

# create N tuples that will serve as permutation functions
# these permutation values are used to hash all input sets

# initialize a sample minhash vector of length N
# each record will be represented by its own vec
vec = [float('inf') for i in range(N)]
primeToUse = len(origData) * 5
def minhash(s, prime=5000):
  '''
  Given a set `s`, pass each member of the set through all permutation
  functions, and set the `ith` position of `vec` to the `ith` permutation
  function's output if that output is smaller than `vec[i]`.
  '''
  # initialize a minhash of length N with positive infinity values
  vec = [float('inf') for i in range(N)]

  for val in s:

    # ensure s is composed of integers
    if not isinstance(val, int): val = hash(val)

    # loop over each "permutation function"
    for perm_idx, perm_vals in enumerate(perms):
      a, b = perm_vals

      # pass `val` through the `ith` permutation function
      output = (a * val + b) % prime

      # conditionally update the `ith` value of vec
      if vec[perm_idx] > output:
        vec[perm_idx] = output

  # the returned vector represents the minimum hash of the set s
  return vec

def getSeedForUserId(entry):
  return int(entry)+10000 *120000

permutations = []
for i in range(k):
  permutations.append(np.random.permutation(amountOfMovies))
def permutingMinHash(val):
  hashes = np.empty([k, signatureSize], dtype=np.bool)
  for v in range(k):
    hashes[v] = val[0, permutations[v]][0, :signatureSize].todense()
  return hashes

def jaccardSimilarity(s1, s2):
  return float(len(s1.intersection(s2))) / max(1, len(s1.union(s2)))

def sortBuckets(toSort):
  sortMe = []
  for i in range(len(toSort)):
    sortMe.append((i, len(toSort[i])))
  sortMe.sort(key=itemgetter(1))
  afterSort = []
  for i in range(len(sortMe)):
    afterSort.append(toSort[sortMe[i][0]])
  return afterSort

def addPairToFile(uId1, uId2):
  with open("result.txt", "a") as myfile:
      myfile.write(str(uId1)+","+str(uId2)+"\n")

print("DATA IS", len(origData))
hashedBands = np.empty([bands, amountOfUsers, rows, signatureSize], dtype=np.bool)

# size extracted by getting last user entry => users = 103702; assignment text says 17770 movies
print("creating sparse matrix")
movieMatrix = dok_matrix((amountOfUsers, amountOfMovies), dtype=np.bool)
print("processing file", time.time() - start)
# Prepare sparse matrix of user ratings

alreadyChecked = {}
pairsWithCheck = 0
usersToProcess = []
checkerThreshold = 750
atUser = 0
for entry in origData:
  movieMatrix[entry[0], entry[1]] = 1
  if entry[0] > atUser:
    usersToProcess.append(atUser)
    if len(usersToProcess) > checkerThreshold:
      cscMovies = movieMatrix.tocsc()
      for processMe in usersToProcess:
        minHashed = np.empty([k, signatureSize], dtype=np.bool)
        userEntry = cscMovies[processMe]
        for v in range(k):
          minHashed[v] = userEntry[0, permutations[v]][0, :signatureSize].todense()
        for i in range(bands):
          b = i + 1
          hashedBands[i][processMe] = minHashed[(len(minHashed) / bands) * i : (len(minHashed) / bands) * b]
        for band in range(bands):
          bucketId = int(hashlib.md5(hashedBands[band][processMe]).hexdigest(), 16) % k
          if processMe not in buckets[bucketId]:
            buckets[bucketId].append(processMe)
            if len(buckets[bucketId]) > 1:
              for uId1 in buckets[bucketId]:
                for uId2 in buckets[bucketId]:
                  if uId1 < uId2 and str(uId1+uId2) not in alreadyChecked:
                    alreadyChecked[str(uId1+uId2)] = True
                    similarity = jaccardSimilarity(set(np.where(cscMovies[uId1].todense() == 1)[1].A1), set(np.where(cscMovies[uId2].todense() == 1)[1].A1))
                    if similarity >= 0.5:
                      addPairToFile(uId1, uId2)
                      elapsed = time.time() - start / 60
                      print("found jaccard > 0.5 for ", uId1, uId2)
                      print("pairs per minute, pairs found, minutes calculated", pairsWithCheck / elapsed, pairsWithCheck, elapsed)
      usersToProcess = []
      elapsed = time.time() - start
      print("processed another batch of users. user per minutes, amount processed, elapsed time in seconds ", atUser / max(1, ((elapsed) / 60)), atUser, elapsed)
    atUser += 1
    

print("DONE PROCESSING after", time.time() - start)

