import numpy as np
from scipy.spatial.distance import cosine
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from operator import itemgetter
import time
start = time.time()
np.random.seed(18)
origData = np.load("user_movie.npy")
print("LAST USER IS", origData[len(origData) -1])
origData = origData
data = origData
# threshold t is approx (1 / b)^(1/r)
bands = 4
rows = 20
signatureSize = 75
amountOfMovies = 17770
amountOfUsers = 103702
k = bands * rows

def jaccardSimilarity(s1, s2):
  return float(len(s1.intersection(s2))) / max(1, len(s1.union(s2)))

print("DATA IS", len(origData))

userMovies = {}
# size extracted by getting last user entry => users = 103702; assignment text says 17770 movies
print("creating sparse matrix")
movieMatrix = lil_matrix((amountOfUsers, amountOfMovies))
userMovies = {}
for us in range(amountOfUsers):
  userMovies[str(us)] = set()
print("processing file", time.time() - start)
# Prepare sparse matrix of user ratings
for entry in origData:
  movieMatrix[entry[0], entry[1]] = 1
  key = str(entry[0])
  userMovies[key].add(entry[1])
movieMatrix = movieMatrix.tocsc()

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
  hashes = []
  for v in range(k):
    hashes.append(val[0, permutations[v]][0, :signatureSize].todense())
  return hashes


hashedBands = []
for i in range(bands):
  hashedBands.append([])
print("preparing hashes in bands and rows", time.time() - start)
for entry in movieMatrix:
  minHashed = permutingMinHash(entry)
  for i in range(bands):
    b = i + 1
    hashedBands[i].append(minHashed[(len(minHashed) / bands) * i : (len(minHashed) / bands) * b])
print("Hashed values now creating buckets", time.time() - start)
buckets = []
for bucketNr in range(k):
  buckets.append([])
for band in range(bands):
  bandContent = hashedBands[band]
  for uId1 in range(len(bandContent)):
    bucketId = hash(str(bandContent[uId1])) % k
    if uId1 not in buckets[bucketId]:
      buckets[bucketId].append(uId1)

print("GOT BUCKETS", time.time() - start)

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
      myfile.write(uId1+","+uId2)



buckets = sortBuckets(buckets)
alreadyChecked = {}
pairsWithoutCheck = []
pairsWithCheck = []
for bucketId in range(len(buckets)):
  if len(buckets[bucketId]) > 1:
    for uId1 in buckets[bucketId]:
      for uId2 in buckets[bucketId]:
        if uId1 != uId2 and str(uId1+uId2) not in alreadyChecked:
          alreadyChecked[str(uId1+uId2)] = True
          pairsWithoutCheck.append((uId1, uId2))
          similarity = jaccardSimilarity(userMovies[str(uId1)], userMovies[str(uId2)])
          if similarity >= 0.5:
            addPairToFile(uId1, uId2)
            pairsWithCheck.append((uId1, uId2, similarity))
print("PARIS done", pairsWithCheck, time.time() - start)


# print("got hashed bands now checking for similarities")
# similarities = []
# calced = {}
# i = 0
# for entry in hashed:
#   b = 0
#   for secondEntry in hashed:
#     if i < b:
#       similarities.append((jaccardSimilarity(set(entry[0]), set(secondEntry[0])), i, b))
#     b += 1
#   i += 1

# amount = 0
# biggerOne = []
# for sim in similarities:
#   if sim[0] > 0.5 and sim[0] < 1:
#     amount += 1
#     biggerOne.append(sim)
# for entry in biggerOne:
#   print(entry[1], entry[2])
# print("similarities are", biggerOne, amount, len(hashed), len(similarities))