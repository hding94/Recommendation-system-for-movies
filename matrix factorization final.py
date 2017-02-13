# -*- coding: utf-8 -*-
"""

@author: Petra Kubernatova & Haoran Ding
Cross-validation based on code written by Wojtek Kowalczyk

"""

import numpy as np

# Loading the data file
ratings=[]
ratings = np.genfromtxt("ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int')

rmse = []
mae = []

# Matrix Factoralization function
def matrix_factorization(R, P, Q, K, steps=75, learn_rate=0.005, regular=0.05):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + learn_rate * (2 * eij * Q[k][j] - regular * P[i][k])
                        Q[k][j] = Q[k][j] + learn_rate * (2 * eij * P[i][k] - regular * Q[k][j])
        
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (regular/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T



# Variable for the number of folds
nfolds=5


# Seeding the random function
np.random.seed(17)

seqs=[x%nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)

# Looping through folds
for fold in range(nfolds):
    # Splitting into train and test sets
    train_sel=np.array([x!=fold for x in seqs])
    test_sel=np.array([x==fold for x in seqs])
    train=ratings[train_sel]
    test=ratings[test_sel]
    
    # Allocating space for the input matrix of the Matrix Factorization
    mf = np.zeros((6041,3953), dtype=np.int)

# Filling the input matrix with ratings of each movie by each user
    for i in range(0,len(train)):
        userid = train[i,0]
        movieid = train[i,1]
        mf[userid][movieid] = train[i,2]

# Setting up inputs for the Matrix Factorization function              
    N = len(mf)
    M = len(mf[0])
    K = 10
    
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    
    nP, nQ = matrix_factorization(mf, P, Q, K)
    # The result matrix = the matrix filled in with predictions for each user rating of each movie
    nR = np.dot(nP, nQ.T)
    print("Fold no. :" + str(fold+1))


    # Calculating the RMSE and MAE
    rmse_temp = []
    mae_temp = []
    for j in range(len(test)):
        e2 = (test[j,2] - nR[test[j,0],test[j,1]])**2
        rmse_temp.append(e2)
        mae_temp.append(np.sqrt(e2))
    rmse.append(np.sqrt(np.mean(rmse_temp)))
    mae.append(np.mean(mae_temp))
    print ("RMSE of " + str(rmse[fold]))
    print ("MAE of " + str(mae[fold]))
print ("Overall mean of RMSE is " + str(np.mean(rmse))) 
print ("Overall mean of MAE is " + str(np.mean(mae)))      

   
   
   