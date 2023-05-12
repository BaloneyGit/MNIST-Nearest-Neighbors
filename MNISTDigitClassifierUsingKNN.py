from tensorflow.keras.datasets import mnist
import numpy as np
from math import sqrt
from statistics import mode

(X_train, y_train),(X_test, y_test) = mnist.load_data()

X_train_reshape = X_train.reshape(60000, 28*28)
X_test_reshape = X_test.reshape(10000, 28*28)

class DigitClassifierKNN:

    def __init__(self) -> None:
        pass

    ## FUNCS
    # CORE functions

    def euclidean_dist(set1, set2):
        dist = sqrt(np.sum(np.square(np.subtract(set1, set2))))
        return dist


    def minkowski_dist_numpy(set1, set2, p):
        dist = np.power(np.sum(np.power(np.absolute(np.subtract(set1, set2)), p)), (1/p)) # numpy method
        return dist 


    def minkowski_dist_math(set1, set2, p):
        dist = 0      # math method
        for i in range(0, len(set1)):
                dist += np.sum((abs(set1[i] - set2))**p)
        dist = dist**(1/p)    
        return dist 


    def distances_and_vectors(smallestDists, vecsDists):
        distsVecs = {i[2]:i[1] for i in vecsDists}
        return distsVecs[smallestDists]


    def nearest_neighbors_numpy(k, vecLabelDist): #USE ONLY WHILE MINKOWSKI NUMPY METHOD
        onlyDists = []
        for i in vecLabelDist:
            onlyDists.append(i[2])

        onlyDistsSorted = np.sort(np.array(onlyDists))
        indices = np.argsort(np.array(onlyDists))
                            
        k_smallestDists, k_indices = [], []
        for i in range(0, k):
            k_smallestDists.append(onlyDistsSorted[i])
            k_indices.append(indices[i])
        
        NearestNeighs = [(DigitClassifierKNN.distances_and_vectors(i, vecLabelDist), j) for i, j in zip(k_smallestDists, k_indices)] #(nearest neighbor vec, indx)
                            
        return NearestNeighs


    def nearest_neighbors_math(k, vecLabelDist):  #USE ONLY WHILE MINKOWSKI MATH METHOD
        onlyDists, temp, indices = [], [], []
        for i in vecLabelDist:
            onlyDists.append(i[2])
        onlyDists.sort() # sorted distance elements
        
        for i in range(0, len(onlyDists)):
            temp.append((onlyDists[i], i))
        temp.sort()
        for i in temp:
            indices.append(i[1]) # sorted indices of distance elements
                            
        k_smallestDists, k_indices = [], []
        for i in range(0, k):
            k_smallestDists.append(onlyDists[i])
            k_indices.append(indices[i])
        
        NearestNeighs = [(DigitClassifierKNN.distances_and_vectors(i, vecLabelDist), j) for i,j in zip(k_smallestDists, k_indices)] #(nearest neighbor vec, indx)
                            
        return NearestNeighs


    def vecIndices_and_labels(nearestNeigh, vecsLabels):
        vecIndLabels = {i[0]:i[2] for i in vecsLabels}
        return vecIndLabels[nearestNeigh]


    # predicts accuracy of this KNN implementation against MNIST database's testing set

    def predict_modelAcc_KNN(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, k, minkowski_PValue, method):  # textImgIndx: index of testing set image (to be used as query image); 
        cntr = 0                                                                              # minkowski_PValue: 1 for manhattan, 2 for euclidean; method: numpy/math methods
        done  = 1 ## use if verbose desired
        testImages = list(enumerate(X_TEST))

        vectorLabel_test = [(ind, v, l) for ind, v, l in zip(range(len(X_TEST)), X_TEST, Y_TEST)] # to find query image label 
        vecIndxLabel_test = {i[0]:i[2] for i in vectorLabel_test}

        if(minkowski_PValue == 1):
                
            if(method == 'numpy'):
                for i in testImages:
                    testImage = i[1]

                    vectorDist = [(i, j, DigitClassifierKNN.minkowski_dist_numpy(j, testImage, 1)) for i, j in zip(range(len(X_TRAIN)), X_TRAIN)] #(Index,Vec,Dist)

                    vecsLabels = [(i, j, k) for i, j, k in zip(range(len(X_TRAIN)), X_TRAIN, Y_TRAIN)] #(Index,Vec,Label)

                    #nearest neighbors to query image
                    nearestNeighbors = DigitClassifierKNN.nearest_neighbors_numpy(k, vectorDist)

                    #predicted labels of nearest neighbors on training data
                    predictedLabels =[(DigitClassifierKNN.vecIndices_and_labels(i[1], vecsLabels), i[1]) for i in nearestNeighbors]

                    #predicted class of query image
                    predList = [i[0] for i in predictedLabels]
                    lambda_class = lambda x: mode(x)
                    _class = lambda_class(predList)

                    #ground truth class of query image
                    label_test = vecIndxLabel_test[i[0]]
                    
                    #count no of correct predictions
                    if _class == label_test:
                        cntr+=1
                        
                    #measuring how much training done as of yet ## use if verbose desired
                    print(f'Done : ({done}/60,000)')
                    done += 1
            

            if(method == 'math'):
                    for i in testImages:
                        testImage = i[1]

                        vectorDist = [(i, j, DigitClassifierKNN.minkowski_dist_math(j, testImage, 1)) for i, j in zip(range(len(X_TRAIN)), X_TRAIN)] #(Index,Vec,Dist)

                        vecsLabels = [(i, j, k) for i, j, k in zip(range(len(X_TRAIN)), X_TRAIN, Y_TRAIN)] #(Index,Vec,Label)

                        #nearest neighbors to query image
                        nearestNeighbors = DigitClassifierKNN.nearest_neighbors_math(k, vectorDist)

                        #predicted labels of nearest neighbors on training data
                        predictedLabels =[(DigitClassifierKNN.vecIndices_and_labels(i[1], vecsLabels), i[1]) for i in nearestNeighbors]

                        #predicted class of query image
                        predList = [i[0] for i in predictedLabels]
                        lambda_class = lambda x: mode(x)
                        _class = lambda_class(predList)

                        #ground truth class of query image
                        label_test = vecIndxLabel_test[i[0]]
                        
                        #count no of correct predictions
                        if _class == label_test:
                            cntr+=1
                            
                        #measuring how much training done as of yet ## use if verbose desired
                        print(f'Done : ({done}/60,000)')
                        done += 1

            acc = cntr/len(y_train)*100
            return acc
        
        elif (minkowski_PValue == 2):

            if(method == 'numpy'):
                for i in testImages:
                    testImage = i[1]

                    vectorDist = [(i, j, DigitClassifierKNN.minkowski_dist_numpy(j, testImage, 2)) for i, j in zip(range(len(X_TRAIN)), X_TRAIN)] #(Index,Vec,Dist)

                    vecsLabels = [(i, j, k) for i, j, k in zip(range(len(X_TRAIN)), X_TRAIN, Y_TRAIN)] #(Index,Vec,Label)

                    #nearest neighbors to query image
                    nearestNeighbors = DigitClassifierKNN.nearest_neighbors_numpy(k, vectorDist)

                    #predicted labels of nearest neighbors on training data
                    predictedLabels =[(DigitClassifierKNN.vecIndices_and_labels(i[1], vecsLabels), i[1]) for i in nearestNeighbors]

                    #predicted class of query image
                    predList = [i[0] for i in predictedLabels]
                    lambda_class = lambda x: mode(x)
                    _class = lambda_class(predList)

                    #ground truth class of query image
                    label_test = vecIndxLabel_test[i[0]]
                    
                    #count no of correct predictions
                    if _class == label_test:
                        cntr+=1
                        
                    #measuring how much training done as of yet ## use if verbose desired
                    print(f'Done : ({done}/60,000)')
                    done += 1


            if(method == 'math'):
                    for i in testImages:
                        testImage = i[1]

                        vectorDist = [(i, j, DigitClassifierKNN.minkowski_dist_math(j, testImage, 2)) for i, j in zip(range(len(X_TRAIN)), X_TRAIN)] #(Index,Vec,Dist)

                        vecsLabels = [(i, j, k) for i, j, k in zip(range(len(X_TRAIN)), X_TRAIN, Y_TRAIN)] #(Index,Vec,Label)

                        #nearest neighbors to query image
                        nearestNeighbors = DigitClassifierKNN.nearest_neighbors_math(k, vectorDist)

                        #predicted labels of nearest neighbors on training data
                        predictedLabels =[(DigitClassifierKNN.vecIndices_and_labels(i[1], vecsLabels), i[1]) for i in nearestNeighbors]

                        #predicted class of query image
                        predList = [i[0] for i in predictedLabels]
                        lambda_class = lambda x: mode(x)
                        _class = lambda_class(predList)

                        #ground truth class of query image
                        label_test = vecIndxLabel_test[i[0]]
                        
                        #count no of correct predictions
                        if _class == label_test:
                            cntr+=1
                            
                        #measuring how much training done as of yet ## use if verbose desired
                        print(f'Done : ({done}/60,000)')
                        done += 1

            acc = cntr/len(y_train)*100  
            return acc                 


    # cross-validation funcs

    def cross_val_split(splitLowerBound, splitUpperBound, setArray):
        valSplit = []
        i = splitLowerBound 
        while i <= (splitUpperBound-1):
            valSplit.append(setArray[i])
            i+=1
        valSplit = np.array(valSplit)
        trainSplit = np.delete(setArray, [i for i in range(splitLowerBound, splitUpperBound)], axis=0)
        
        return valSplit, trainSplit


    def cross_val_predict_KNN(splitsNo, k, minkowski_PValue, method):
        upperBound = len(X_train_reshape)/splitsNo
        i, j = 0, 0
        a = upperBound
        boundsUpper, boundsLower = [], []
        while i < len(splitsNo):
            boundsLower.append(j)
            boundsUpper.append(a)
            j += upperBound
            a += upperBound
            i += 1

        avg_acc, done = 0, 0
        for lower, upper in zip(boundsLower, boundsUpper):
            xValSplit, xTrainSplit = DigitClassifierKNN.cross_val_split(lower, upper, X_train_reshape)
            yValSplit, yTrainSplit = DigitClassifierKNN.cross_val_split(lower, upper, y_train)
            print(f'for {lower} to {upper} : xValSplit shape = {xValSplit.shape}, xTrainSplit shape = {xTrainSplit.shape}, yValSplit shape = {yValSplit.shape}, yTrainSplit shape = {yTrainSplit.shape}')

            prediction = DigitClassifierKNN.predict_modelAcc_KNN(xTrainSplit, yTrainSplit, xValSplit, yValSplit, k, minkowski_PValue, method)
            avg_acc = (avg_acc + prediction)/splitsNo
            print(f'{done} splits done')        # for verbose
            print(f'Accuracy for split {done}: {prediction}%')
            print(f'Average Accuracy: {avg_acc}')
            
            done+=1 # for verbose




    def ground_truth_class(indx):
        return y_test[indx]

    def predict_class(X_TRAIN, Y_TRAIN, queryImg):

        vectorDist = [(i, j, DigitClassifierKNN.minkowski_dist_numpy(j, queryImg, 2)) for i, j in zip(range(len(X_TRAIN)), X_TRAIN)]

        vecsLabels = [(i, j, k) for i, j, k in zip(range(len(X_TRAIN)), X_TRAIN, Y_TRAIN)] #(Index,Vec,Label)

        #nearest neighbors to query image
        nearestNeighbors = DigitClassifierKNN.nearest_neighbors_numpy(3, vectorDist)

        #predicted labels of nearest neighbors on training data
        predictedLabels =[(DigitClassifierKNN.vecIndices_and_labels(i[1], vecsLabels), i[1]) for i in nearestNeighbors]

        #predicted class of query image
        predList = [i[0] for i in predictedLabels]
        lambda_class = lambda x: mode(x)
        _class = lambda_class(predList)

        return _class