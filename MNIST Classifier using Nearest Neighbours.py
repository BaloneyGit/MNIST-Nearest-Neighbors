#!/usr/bin/env python
# coding: utf-8

# # Implementation from scratch

# Algorithm:-
# 
# A) Distance
# 1. Find respective distance between all training images and query image. All training images from training set needed for this. 
# Selected query image from testing set. 
# 2. Distance found using distance metric = euclidean. Other distance metrics: manhattan, minkowski etc. Manhattan generally used for large dimensions. Image data can be represented in 2 dimensions, hence euclidean would be better fit.
# 3. Custom euclidean function created. The distance measured are the distances between all training nodes. In this, a scattered cluster of image data (as a vector) represents one node. Distances between all points of all nodes and all points of query image found. Target: Closest node distances to query image.
# 
# B) Nearest Neighbors
# 1. Set of nodes having closest distances obtained. Selecting a 'k' value, nearest k nodes are found out. An odd value for k is preferable to avoid a clash in predicting class of query image.
# 2. Custom nearest neighbors function created. Converted matrix image data to numpy array and sorted the same. Finding 'k' no. of nodes closest to query image.
# 
# C) Prediction
# 1. We know the classes (labels) of all training image nodes (y_train). Using the same, we find the labels of the nearest neighbors found using the previous function. The class occuring with highest frequency can be assigned hence to the query image.
# 
# Finding accuracy:-
# 
# 1. We have the testing set data (X_test), with corresponding ground truth labels (y_test). 
# 
# Accuracy = (no. of correct classfications)/(total no. of classifications)
# 
# 2. Correct classifications can be found: compare predicted class found by the 'queryImagePredictedClass()' function with ground truth (y_test). Count no. of correct predictions (will be <10,000).
# 3. Total classifications: Is the total no of testing data points present (i.e., 10,000).

# In[1]:


from tensorflow.keras.datasets import mnist
import numpy as np
from math import sqrt
from statistics import mode


# In[2]:


(X_train, y_train),(X_test, y_test) = mnist.load_data()


# In[3]:


X_train_reshape = X_train.reshape(60000, 28*28)
print(X_train_reshape.shape)

X_test_reshape = X_test.reshape(10000, 28*28)
print(X_test_reshape.shape)


# In[6]:


X_train_reshape[0]


# In[4]:


def euclidean_dist(set1, set2):
    dist = sqrt(np.sum(np.square(np.subtract(set1, set2))))
    return dist


# In[5]:


def distances_and_vectors(smallestDists, vecsDists):
    distsVecs = {i[2]:i[1] for i in vecsDists}
    return distsVecs[smallestDists]


# In[6]:


def nearest_neighbors(k, vecsDists):
    onlyDists = []
    for i in vecsDists:
        onlyDists.append(i[2])
                         
    onlyDistsSorted = np.sort(np.array(onlyDists))
    indices = np.argsort(np.array(onlyDists))
                         
    k_smallestDists, k_indices = [], []
    for i in range(0, k):
        k_smallestDists.append(onlyDistsSorted[i])
        k_indices.append(indices[i])
    
    NearestNeighs = [(distances_and_vectors(i, vecsDists), j) for i,j in zip(k_smallestDists, k_indices)]#(nearest neighbor vec, indx)
                         
    return NearestNeighs


# In[7]:


def vecIndices_and_labels(nearestNeigh, vecsLabels):
    vecIndLabels = {i[0]:i[2] for i in vecsLabels}
    return vecIndLabels[nearestNeigh]


# In[8]:


def queryImagePredictedClass(labels):
    return mode(labels)


# ## Implemention for 1 test image

# In[9]:


#distances between training images and a random query image from testing set
testImageIndx = 15
testImage = X_test_reshape[testImageIndx]

#when nearest neighbors dont need to be displayed
vectorDist_NoNN = {i:euclidean_dist(j, testImage) for i, j in zip(range(len(X_train_reshape)), X_train_reshape)}

#nearest neigbors being displayed
vectorDist = [(i, j, euclidean_dist(j, testImage)) for i, j in zip(range(len(X_train_reshape)), X_train_reshape)]#(Index,Vec,Dist)
#vectorLabel = [enumerate(i) for i in y_train] #enumerated(Labels) needed for 
                                              #indexing (Labels)
vecsLabels = [(i, j, k) for i,j,k in zip(range(len(X_train_reshape)), X_train_reshape, y_train)]#(Index,Vec,Label)

print(f'Size of vectorDist = {len(vectorDist)}')
print(f'Size of vecsLabels = {len(vecsLabels)}')


# In[13]:


#nearest neighbors to query image
nearestNeighbors = nearest_neighbors(5, vectorDist) # selecting k=5, odd no of nodes
print(f'Nearest Neighbors: {nearestNeighbors}')


# In[14]:


#predicted labels of nearest neighbors on training data
predictedLabels =[(vecIndices_and_labels(i[1],vecsLabels), i[1]) for i in nearestNeighbors]
print('Nearest Neighbors:')
for i in predictedLabels:
    print(f'Nearest Neighbor {i[1]} has label {i[0]}')


# In[15]:


#predicted class of query image
_class = queryImagePredictedClass([i[0] for i in predictedLabels])
print(f'Predicted Class of Query Image: {_class}')


# In[16]:


#ground truth class of query image
vectorLabel_test = [(ind, v, l) for ind, v, l in zip(range(len(X_test_reshape)), X_test_reshape, y_test)]
vecIndxLabel_test = {i[0]:i[2] for i in vectorLabel_test}
label_test = vecIndxLabel_test[testImageIndx]

print(f'Ground Truth label of Test Image: {label_test}')


# ## Implementation for entire training set

# In[18]:


#ACCURACY on training set: nodes = 3 : 75.661
cntr = 0
done  = 1
trainImageIndx = list(enumerate(X_train_reshape))

for i in trainImageIndx:
    trainImage = i[1]

    #when nearest neighbors dont need to be displayed
    #vectorDist_NoNN = {i:euclidean_dist(j, testImage) for i, j in zip(range(len(X_train_reshape)), X_train_reshape)}

    #nearest neigbors being displayed
    vectorDist = [(i, j, euclidean_dist(j, trainImage)) for i, j in zip(range(len(X_train_reshape)), X_train_reshape)]#(Index,Vec,Dist)
    #vectorLabel = [enumerate(i) for i in y_train] #enumerated(Labels) needed for 
                                                  #indexing (Labels)
    vecsLabels = [(i, j, k) for i, j, k in zip(range(len(X_train_reshape)), X_train_reshape, y_train)]#(Index,Vec,Label)

    #nearest neighbors to query image
    nearestNeighbors = nearest_neighbors(3, vectorDist) # selecting k=3, odd no of nodes

    #predicted labels of nearest neighbors on training data
    predictedLabels =[(vecIndices_and_labels(i[1], vecsLabels), i[1]) for i in nearestNeighbors]

    #predicted class of query image
    _class = queryImagePredictedClass([i[0] for i in predictedLabels])

    #ground truth class of query image
    vectorLabel_train = [(ind, v, l) for ind, v, l in zip(range(len(X_train_reshape)), X_train_reshape, y_train)]
    vecIndxLabel_train = {i[0]:i[2] for i in vectorLabel_train}
    label_test = vecIndxLabel_train[i[0]]
    
    #count no of correct predictions
    if _class == label_test:
        cntr+=1
        
    #measuring how much training done as of yet
    print(f'Done : ({done}/60,000)')
    done += 1

acc = cntr/len(y_train)*100
print(f'Accuracy: {acc}%')


# Training accuracy:
# (1) k = 3 : 75.662

# ## Implementation for entire testing set

# In[10]:


#ACCURACY on testing set: nodes = 15 : 28.79%, nodes = 3 : 27.43%
cntr = 0
done = 0
testImageIndx = list(enumerate(X_test_reshape))

for i in testImageIndx:
    testImage = i[1]

    #when nearest neighbors dont need to be displayed
    #vectorDist_NoNN = {i:euclidean_dist(j, testImage) for i, j in zip(range(len(X_train_reshape)), X_train_reshape)}

    #nearest neigbors being displayed
    vectorDist = [(i, j, euclidean_dist(j, testImage)) for i, j in zip(range(len(X_train_reshape)), X_train_reshape)]#(Index,Vec,Dist)
    #vectorLabel = [enumerate(i) for i in y_train] #enumerated(Labels) needed for 
                                                  #indexing (Labels)
    vecsLabels = [(i, j, k) for i, j, k in zip(range(len(X_train_reshape)), X_train_reshape, y_train)]#(Index,Vec,Label)

    #nearest neighbors to query image
    nearestNeighbors = nearest_neighbors(3, vectorDist) # selecting k=3, odd no of nodes

    #predicted labels of nearest neighbors on training data
    predictedLabels =[(vecIndices_and_labels(i[1], vecsLabels), i[1]) for i in nearestNeighbors]

    #predicted class of query image
    _class = queryImagePredictedClass([i[0] for i in predictedLabels])

    #ground truth class of query image
    vectorLabel_test = [(ind, v, l) for ind, v, l in zip(range(len(X_test_reshape)), X_test_reshape, y_test)]
    vecIndxLabel_test = {i[0]:i[2] for i in vectorLabel_test}
    label_test = vecIndxLabel_test[i[0]]
    
    #count no of correct predictions
    if _class == label_test:
        cntr+=1
        
    #measuring how much training done as of yet
    print(f'Done : ({done}/10,000)')
    done += 1

acc = cntr/len(y_test)*100
print(f'Accuracy: {acc}%')


# Testing accuracy:
# (1) k = 3 : 27.43%
# (2) k = 15 : 28.79%

# ===========================+++++============================

# ## Rough Space

# In[109]:


len(list(enumerate(X_train_reshape)))


# In[10]:


lst = [1,2,3,4,5,6,7,8,9,10]
print(dict(enumerate(lst)))
for i in enumerate(lst):
    print(i[1])


# In[17]:


n = np.array([1,2,3,4,5])
print(np.square(n))
print(sqrt(np.sum(n)))


# In[48]:


for i in enumerate([(0,1), (1,2), (2, 3), (3,4)]):
    print(i, i[1][1])
    


# In[33]:


tejas = [(0,'tejas'),(1,'maneesh'),(2,'joshi')]
name = []
for i in tejas:
    name.append(i[1])
print(name)
_sorted = np.argsort(name)
print(_sorted)
print(name(_sorted))


# In[4]:


a = [(1,2,3),(4,5,6),(7,8,9),(10,11,12)]
print(a[0], a[0][2])


# In[8]:


lsty = [1,2,3,4,5]
for i in lsty:
    print(i)


# In[9]:


for i in enumerate([1,0,0,1,1,0,1,0]):
    print(i)


# In[24]:


my_dict = {enumerate(i[0]):i[1] for i in [(np.array([0,0,0]),'t'), (np.array([1,1,1]),'e'), 
                                         (np.array([2,2,2]),'j'), (np.array([3,3,3]),'a'), 
                                         (np.array([4,4,4]),'s')]}
print(my_dict[np.array([2,2,2])])


# In[ ]:


my_dict = {(, i[0]):i[1] for i in [(np.array([0,0,0]),'t'), (np.array([1,1,1]),'e'), 
                                         (np.array([2,2,2]),'j'), (np.array([3,3,3]),'a'), 
                                         (np.array([4,4,4]),'s')]}
print(my_dict[np.array([2,2,2])])


# In[26]:


for i, j in enumerate(zip([0,1,2,3,4,5], [6,7,8,9,10])):
    print(i, j)


# In[1]:


for i, j in zip([0,1,2,3,4,5], [6,7,8,9,10]):
    print(i, j)


# In[12]:


v = [(type(i), type(euclidean_dist(i, testImage))) for i in X_train_reshape[3]]
print(v)


# In[13]:


x = [1.1, 2.2, 3.3, 4.4, 5.5]
y = [enumerate(i,0) for i in x]
print(y)


# In[18]:


print(type({1:'a', 2:'b', 3:'c', 4:'d', 5:'e'}))
for i in len({1:'a', 2:'b', 3:'c', 4:'d', 5:'e'}):
    print(i)


# In[32]:


x = ['t','e','j','a','s']
y = ['j','o','s','h','i']
z = [1,2,3,4,5]
for i, j, k in zip(x,y,z):
    print(i, j, k)


# In[33]:


np.argsort([10,9,8,7,6,5])


# In[47]:


a,b = [],[]
print(a,b)


# In[27]:


x = [1,2,3,4,5,6,7,8,9,10]
print([(i, j) for i, j in zip(range(len(x)), x)])


# In[34]:


x = [10,4,3,6,8,9,7,2,1,5]
print(np.sort(np.array(x)))
_sorted = np.argsort(np.array(x))
print(_sorted)
for i in _sorted:
    print(x[i])


# In[36]:


print(mode([1,1,1,1,2,2,3,4]))


# In[52]:


X_train_reshape[2]


# # Library-based Implementation

# ## Loading dataset

# Pre-processing:
# Original shape of training and testing data is (60000,28,28) and (10000,28,28) respectively.
# Needed to reshape both to (60000,28*28) and (10000,28*28) respectively for feeding into sklearn.
# 
# Model fitting and prediction:
# Parameter specification which gives the max accuracy as of yet.
# 
# Visualizing results:
# Displays training images, while the title shows KNN's prediction for the training images.
# 
# Rough space:
# Process I undertook to get to the updated max accuracy. 
# - Firstly, tried the eucleadean metric, followed by the manhattan metric for a range of nodes from 1 to 30. Tried the eucleadean metric with a similar range of nodes as well, found 3 nodes consideration gives the best result [More nodes -> more generalized model].
# - Tried cross-validation with folds ranging from 5 to 10 (default folds = 5), to see if shuffling training and testing data improves accuracy.
# - Trying grid search with changing following parameters:
# > nodes: 2, 3, 4 and 5
# > weights: uniform and distance(closer points have more weight in prediction)
# > algorithm: ball tree and k-dimensional tree (brute force also an option, but does not apply to our dataset, hence discarded)
# > metric: eucleadean and manhattan
# > leaf size (for ball tree and k-dimensional tree): 15, 20 and 30 (default is 30)
# 
# Sources used:
# 1. Data Science Stack Exchange
# 2. Stackoverflow
# 3. The 'Machine Learning Mastery' website and blogs - by Jason Brownlee
# 4. tensorflow and keras documentation
# 5. scikit-learn documentation
# 6. GeeksforGeeks website and blogs
# 7. Personal notes from 'Data Science using Python' and 'ML using Python' courses

# ## Loading dataset

# In[2]:


from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.neighbors as NearestNeighbors 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[4]:


(X_train, y_train),(X_test, y_test) = mnist.load_data()


# In[5]:


print(X_train.shape, y_train.shape, X_test.shape, y_train.shape)


# In[4]:


# X_train_resize = np.resize(X_train, (60000, 28*28))
# print(X_train_resize.shape)


# In[9]:


X_train[0]


# In[8]:


y_train[0]


# ## Pre-processing 

# In[6]:


X_train_reshape = X_train.reshape(60000, 28*28)
print(X_train_reshape.shape)


# In[7]:


X_test_reshape = X_test.reshape(10000, 28*28)
print(X_test_reshape.shape)


# In[8]:


plt.imshow(X_train[0])
plt.title(y_train[0])


# In[10]:


plt.imshow(X_test[272])
plt.title(y_test[272])


# In[29]:


X_train_reshape[0]


# In[25]:


sns.scatterplot(X_train_reshape[2])


# ## Model fitting and prediction

# In[8]:


k = 3 #gives best accuracy (check code in 'Rough space' block below for how I've found that out)
w_nearestNeighs = NearestNeighbors.KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto', 
                                                      p=2, metric='minkowski')
w_fitted_modelKNN = w_nearestNeighs.fit(X_train_reshape, y_train)
print(w_fitted_modelKNN)

w_prediction_KNN = w_nearestNeighs.predict(X_train_reshape)
print(w_prediction_KNN)

w_score_KNN = w_nearestNeighs.score(X_test_reshape, y_test)
print("Score: "+str(w_score_KNN))


# ## Visualizing results

# In[10]:


fig = plt.figure(figsize=(15, 7))
rows = 3
columns = 10
for i in range(0, 30):
    fig.add_subplot(rows, columns, (i+1))
    plt.imshow(X_train[i])
    plt.axis('off')
    plt.title(w_prediction_KNN[i])


# ## Rough Space

# Current accuracy updated: 0.9717 with parameters (n_neighbors=3, weights='distance', algorithm='auto', p=2, metric='minkowski')
# 
# Progress to get to current accuracy:

# In[11]:


k_scores = [] #values of k with euclidean metric
for i in range(1, 31):
    k_nearestNeighs = NearestNeighbors.KNeighborsClassifier(n_neighbors=i, weights='uniform', algorithm='auto', 
                                                      p=2, metric='minkowski')
    k_fitted_modelKNN = k_nearestNeighs.fit(X_train_reshape, y_train)
    print(k_fitted_modelKNN)
    
    k_score_KNN = k_nearestNeighs.score(X_test_reshape, y_test)
    k_scores.append(k_score_KNN)

print(k_scores)


# In[16]:


k_scores = [] #various values of k with manhattan metric
for i in range(1, 31):
    k_nearestNeighs = NearestNeighbors.KNeighborsClassifier(n_neighbors=i, weights='uniform', algorithm='auto', 
                                                      p=1, metric='manhattan')
    k_fitted_modelKNN = k_nearestNeighs.fit(X_train_reshape, y_train)
    print(k_fitted_modelKNN)
    
    k_score_KNN = k_nearestNeighs.score(X_test_reshape, y_test)
    k_scores.append(k_score_KNN)

print(k_scores)


# In[42]:


k = 3 
nearestNeighs = NearestNeighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', 
                                                      p=2, metric='minkowski')
fitted_modelKNN = nearestNeighs.fit(X_train_reshape, y_train)
print(fitted_modelKNN)

prediction_KNN = nearestNeighs.predict(X_train_reshape)
print(prediction_KNN)

score_KNN = nearestNeighs.score(X_test_reshape, y_test)
print("Score: "+str(score_KNN))


# In[37]:


for i in range(5, 11):
    crossValScore = cross_val_score(fitted_modelKNN, X_test_reshape, y_test, cv=i)
    print(f"for cv={i}, mean cross validated score = "+str(np.mean(crossValScore)))


# In[ ]:


parameters = [{'n_neighbors':[2,3,4,5], 'weights':['uniform','distance'], 
               'algorithm':['ball_tree', 'kd_tree'], 'metric':['euclidean', 'manhattan'], 
               'leaf_size':[15,20,30]}]

Grid = GridSearchCV(NearestNeighbors.KNeighborsClassifier(), parameters, cv=10, refit=True, verbose=3)
Grid.fit(X_train_reshape, y_train)

print("Best estimator: "+str(Grid.best_estimator_))
print("Best parameters: "+str(Grid.best_params_))
print("Score = "+(Grid.best_score_))


# In[54]:


NearestNeighbors.KNeighborsClassifier().get_params().keys()


# In[ ]:




