import hashlib
import random

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
  md5 = hashlib.md5()
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
  return m * (2 ** ((1 / m) * sumOfM))

print([100000/loglog([random.random() for i in range(100000)], 10) for j in range(10)])
