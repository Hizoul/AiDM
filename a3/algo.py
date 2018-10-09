import numpy as np
from scipy.spatial.distance import cosine
np.random.seed(18)
data = np.load("user_movie.npy")[:10000]

bands = 15
rows = 10
k = bands * rows

def jaccardSimilarity(s1, s2):
  return float(len(s1.intersection(s2))) / len(s1.union(s2))

print("DATA IS", len(data))

userMovies = {}

for entry in data:
  key = str(entry[0])
  if key not in userMovies:
    userMovies[key] = set()
  userMovies[key].add(entry[1])



# specify the length of each minhash vector
N = k
max_val = (2**32)-1

# create N tuples that will serve as permutation functions
# these permutation values are used to hash all input sets
for entry in userMovies:
  perms = [ (np.random.randint(0,max_val), np.random.randint(0,max_val)) for i in range(N)]

# initialize a sample minhash vector of length N
# each record will be represented by its own vec
vec = [float('inf') for i in range(N)]

def minhash(s, prime=4294967311):
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

hashedBands = []
for i in range(bands):
  hashedBands.append([])
for entry in userMovies:
  minHashed = minhash(userMovies[entry], getSeedForUserId(entry))
  for i in range(bands):
    b = i + 1
    hashedBands[i].append(minHashed[(len(minHashed) / bands) * i : (len(minHashed) / bands) * b])

similarities = []
found = 0
for entry in userMovies:
  hashedEntry = minhash(userMovies[entry], getSeedForUserId(entry))
  for i in range(bands):
    if hashedEntry[(len(hashedEntry) / bands) * i : (len(hashedEntry) / bands) * (i+1)] in hashedBands[i]:
      print("FOUND")
      found += 1
      break
print("FOUND IS", found, len(userMovies))
# similarities = []
# for entry in userMovies:
#   for secondEntry in userMovies:
#     similarities.append(jaccardSimilarity(userMovies[entry], userMovies[secondEntry]))

# amount = 0
# biggerOne = []
# for sim in similarities:
#   if sim > 0.5:
#     amount += 1
#     biggerOne.append(sim)
# print("similarities are", amount, similarities, biggerOne)