#from getData import getRatings
import numpy as np 
import time 

def timer():
    return time.time()

#parameters
num_factors = 10
num_iter = 5
regularization = 0.05
lr = 0.005
folds=5

#to make sure you are able to repeat results, set the random seed to something:
np.random.seed(17)

#read rating data 
ratings = np.genfromtxt("D:/Leiden/Semester 1_Sept/Assignment1/AiDM/ml-1m/ratings.dat", usecols=(0,1,2), delimiter='::',dtype='int')

#number of users and movies in data. 
num_users= np.max(ratings[:,0])
num_movies= np.max(ratings[:,1])


def split_matrix(ratings, num_users, num_movies):
    #Convert data into (IxJ) matrix
    X= np.zeros((num_users, num_movies))*np.nan
    for r in np.arange(len(ratings)):
        X[ratings[r,0]-1,ratings[r,1]-1] = ratings[r,2]

    #print(X.shape)
    return X


def mf_gd(ratings, num_users, num_movies):
    X_data= split_matrix(ratings, num_users, num_movies)

    X_pred = np.zeros(num_users, num_movies) #predicted rating matrix
    err = np.zeros(num_users, num_movies) #error values

    # Randomly initialize weights in U and M 
    U = np.random.rand(num_users, num_factors)
    M = np.random.rand(num_factors, num_movies)
    U_prime = U
    M_prime = M

    
    for nr in np.arange(num_iter):
        for i in np.arange(len(ratings)):
            userID = ratings[i,0]-1
            movieID = ratings[i,1]-1
            actual = ratings[i,2]
            prediction = np.sum(U[userID,:]*M[:,movieID]) 
            error = actual - prediction  #compute e(i,j)

            #update U and M using following equations:
            #Uprime(i,k) = u(i,k) + lr(2e*m(k,j)-lamda.u(i,k))
            #Mprime(k,j) = m(k,j) + lr(2e*u(i,k)-lamda.m(k,j))
            for k in np.arange(num_factors):
                U_prime[userID,k] = U[userID,k]+ lr * (2*error*M[k,movieID] - regularization * U[userID,k])
                M_prime[k,movieID] = M[k,movieID] + lr * (2*error*U[userID,k] - regularization * M[k,movieID])

        U = U_prime
        M = M_prime

        #Intermediate RMSE
        X_pred = np.dot(U,M)
        err = X_data-X_pred
        e = err[np.where(np.isnan(err)==False)]
        ir = np.sqrt(np.mean(e**2))

        print ("Error for iteration #", nr+1, ":", ir)

    #Return the result 
    X_pred = np.dot(U,M)
    return X_pred


def mf():
    #Read dataset 
    #ratings = getRatings()
    #ratings = np.genfromtxt("D:/Leiden/Semester 1_Sept/Assignment1/AiDM/ml-1m/ratings.dat", usecols=(0,1,2), delimiter='::',dtype='int')

   

    #print(num_users, num_movies)
    #print(len(ratings))
    

    start = timer()
    #5-fold cross validation
    for f in np.arange(folds):
        print ("Fold #", f)

        #shuffle data  for train and test
        np.random.shuffle(ratings)
        train_set = np.array([ratings[x] for x in np.arange(len(ratings)) if (x%folds) !=f])
        test_set = np.array([ratings[x] for x in np.arange(len(ratings)) if (x%folds) == f])

        
        #Matrix fact
        elapsed=0
        X_pred = mf_gd(train_set, num_users, num_movies)

        elapsed += timer() - start

        X_train = split_matrix(train_set, num_users, num_movies)
        X_test = split_matrix(test_set, num_users, num_movies)

        err_train = X_train- X_pred
        err_test = X_test - X_pred

        #RMSE
        e_mf = err_train[np.where(np.isnan(err_train)==False)]
        error_train_mf = np.sqrt(np.mean(e_mf**2))

        e2_mf = err_test[np.where(np.isnan(err_test)==False)]
        error_test_mf = np.sqrt(np.mean(e2_mf**2))
        

        print ('Matrix Factorization Error -> training set: ', error_train_mf)
        print ('Matrix Factorization Error -> test set: ', error_test_mf)
        print("Time: " + str(elapsed % 60) + "seconds")
mf()


#https://medium.com/coinmonks/recommendation-engine-python-401c080c583e; followed this blogpost 