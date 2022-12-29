import numpy as np
from math import sqrt
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = np.sqrt(np.sum((row1-row2)**2))
    return distance

    
class KNN:
    def __init__(self , k=3):
        self.k = k

    def fit(self , X , y):
        self.X_train = X
        self.y_train = y
    
    def predict(self , X):
        predictions = [self._predict(x) for x in X]
        return predictions
    
    def _predict(self , x):
        #compute the distaces
     
        distances = [euclidean_distance(x , x_train) for x_train in self.X_train]

        # get the closest k
        #tell where are the indices from the original array are after sorted
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        #get class using majority
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    def get_accuracy(self, predicted , test):
        predictedsum = 0
        for i in range(len(predicted)):
            if(predicted[i] == test[i]):
                predictedsum += 1
        return predictedsum / len(test)