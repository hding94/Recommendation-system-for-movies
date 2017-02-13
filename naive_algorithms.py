
"""

@author: Petra Kubernatova & Haoran Ding
Cross-fold validation implementation based on code written by Wojtek Kowalczyk

"""

import numpy as np



# Loading the data
ratings=[]
ratings = np.genfromtxt("ratings.dat", usecols=(0, 1, 2), delimiter='::', dtype='int')

rmse = []
mae = []
nfolds=5



# Seeding of the random function
np.random.seed(17)

seqs=[x%nfolds for x in range(len(ratings))]
np.random.shuffle(seqs)

# For each fold
for fold in range(nfolds):
    # Split into training and test sets
    train_sel=np.array([x!=fold for x in seqs])
    test_sel=np.array([x==fold for x in seqs])
    train=ratings[train_sel]
    test=ratings[test_sel]

 
    # Mean of all ratings of the train set
    train_mean_all = np.mean(train[:,2])
    
    # Calculate the mean of each user in the training set and replace the nonavailable ones with fallback values
    users_mean_temp = np.zeros(6041)
    for userid in range(1,6041):
         user_mean = np.mean(train[train[:,0]==userid][:,2])
         users_mean_temp[userid] = user_mean
    for i in range(1,6041):
        if np.isnan(users_mean_temp[i]):
            users_mean_temp[i] = train_mean_all
            
    train_users_mean = np.array(users_mean_temp)

    
    
    # Mean of each movie in the training set and replace the nonavailable ones with fallback values
    movies_mean_temp = np.zeros(3953)
    for movieid in range(1,3953):
         movie_mean = np.mean(train[train[:,1]==movieid][:,2])
         movies_mean_temp[movieid] = movie_mean
    for i in range(1,3953):
        if np.isnan(movies_mean_temp[i]):
            movies_mean_temp[i] = train_mean_all
    
    train_movies_mean = np.array(movies_mean_temp)
     
    ratings_new_temp = []
    
    
    # Replacing elements in the original array with their respective means
    for i in range(0,len(train)):
        user_id = train[i,0]
        movie_id = train[i,1]
        ratings_new_temp.append([train_users_mean[user_id],train_movies_mean[movie_id],1])
    
    ratings_new = np.array(ratings_new_temp)
    
    ## Linear regression
    
    a,b,r = np.linalg.lstsq(ratings_new,train[:,2])[0]
    print("the a b r No. " + str(fold))
    print (a,b,r)
    
    # Mean of rating in test set
    test_mean_all = np.mean(test[:,2])    
    
    # Calculate the mean of each user of the testing set and replace the nonavailable ones with fallback values
    test_users_mean_temp = np.zeros(6041)
    for userid in range(1,6041):
         user_mean = np.mean(test[test[:,0]==userid][:,2])
         test_users_mean_temp[userid] = user_mean
    for i in range(1,6041):
        if np.isnan(test_users_mean_temp[i]):
            test_users_mean_temp[i] = test_mean_all
            
    test_users_mean = np.array(test_users_mean_temp)

    
    
    # Mean of each movie in test set and replace the nonavailable ones with fallback values
    test_movies_mean_temp = np.zeros(3953)
    for movieid in range(1,3953):
         movie_mean = np.mean(train[train[:,1]==movieid][:,2])
         test_movies_mean_temp[movieid] = movie_mean
    for i in range(1,3953):
        if np.isnan(test_movies_mean_temp[i]):
            test_movies_mean_temp[i] = test_mean_all
    
    test_movies_mean = np.array(movies_mean_temp)
    
    rmse_temp = []
    mae_temp = []
    
    # Calculating the prediction and the error rate
    for j in range(0,len(test)):
        pre = a*test_users_mean[test[j,0]] + b*test_movies_mean[test[j,1]] + r
        # Rounding the rating prediction in the range of 1 to 5

        if pre > 5: 
            pre = 5
        if pre < 1:
            pre = 1   
        e2 = (test[j,2] - pre)**2
        rmse_temp.append(e2)
        mae_temp.append(np.sqrt(e2))
    
    rmse.append(np.sqrt(np.mean(rmse_temp)))
    mae.append(np.mean(mae_temp))
    print ("RMSE of the " + str(fold) + " fold is " + str(rmse[fold]))
    print ("MAE of " + str(fold) + " fold is "+ str(mae[fold]))
print ("Over all mean of RMSE is " + str(np.mean(rmse))) 
print ("Over all mean of MAE is " + str(np.mean(mae)))  





