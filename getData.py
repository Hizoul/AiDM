import numpy as np

#to make sure you are able to repeat results, set the random seed to something:
np.random.seed(17)

def getMovies():
  movies = []
  with open("data/movies.dat") as f:
    for line in f:
      content = line.split("::")
      movie = {"id": int(content[0]), "name": content[1], "category": content[2]}
      movies.append(movie)
  return movies

def getRatings():
  ratings = []
  with open("data/ratings.dat") as f:
    for line in f:
      content = line.split("::")
      rating = {"userId": int(content[0]), "movieId": int(content[1]), "rating": int(content[2]), "timestamp": content[3]}
      ratings.append(rating)
  return ratings

def getRatingsForMovie(movieId, ratings):
  movieRatings = []
  for rating in ratings:
    if (rating["movieId"] == movieId):
      movieRatings.append(rating)
  return movieRatings

def getRatingsForUser(userId, ratings=getRatings()):
  userRatings = []
  for rating in ratings:
    if (rating["userId"] == userId):
      userRatings.append(rating)
  return userRatings

def allNumericRatings():
  ratings = []
  with open("data/ratings.dat") as f:
    for line in f:
      content = line.split("::")
      ratings.append(int(content[2]))
  return ratings

def getUsers():
  users = []
  with open("data/users.dat") as f:
    for line in f:
      content = line.split("::")
      # UserID::Gender::Age::Occupation::Zip-code
      user = {"userId": int(content[0]), "gender": content[1], "age": int(content[2]), "occupation": content[3], "zip": content[4]}
      users.append(user)
  return users

def getRatingSets(ratings):
  nfolds = 5
  np.random.shuffle(ratings)
  sampleSize = len(ratings) / nfolds
  print("each sample set is gonna be", sampleSize)
  # for each fold:
  sets = [
    {"train": [], "test": [], "testNumOnly": [], "min": 0, "max": sampleSize},
    {"train": [], "test": [], "testNumOnly": [], "min": sampleSize, "max": sampleSize * 2},
    {"train": [], "test": [], "testNumOnly": [], "min": sampleSize * 2, "max": sampleSize * 3},
    {"train": [], "test": [], "testNumOnly": [], "min": sampleSize * 3, "max": sampleSize * 4},
    {"train": [], "test": [], "testNumOnly": [], "min": sampleSize * 4, "max": sampleSize * 5}
  ]
  for ratingIndex in range(len(ratings)):
    rating = ratings[ratingIndex]
    for set in sets:
      if (ratingIndex >= set["min"] and ratingIndex < set["max"]):
        set["train"].append(rating)
      else:
        set["test"].append(rating)
        set["testNumOnly"].append(rating["rating"])
  return sets

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())