'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels
        random.seed(0)  # Make results consistent

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        distances = [(i, dist) for i, dist in enumerate(self.l2_distance(test_point))]
        distances.sort(key=lambda x: x[1])
        distances = distances[:k]
        class_votes = {}
        class_min_dist = {}
        for i, dist in distances:
            if self.train_labels[i] not in class_votes:
                class_votes[self.train_labels[i]] = 0
                class_min_dist[self.train_labels[i]] = float("inf")
            class_votes[self.train_labels[i]] += 1
            if dist < class_min_dist[self.train_labels[i]]:
                class_min_dist[self.train_labels[i]] = dist
        max_vote_count = max(class_votes.values())
        conflicting_classes = [cls for cls, votes in class_votes.items() if votes == max_vote_count]
        ret = [min(conflicting_classes, key=lambda cls: class_min_dist[cls])]
        return ret

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    k_average_accuracy = {}
    for k in k_range:
        k_average_accuracy[k] = 0
        kf = KFold(n_splits=10)
        for train_indx, test_indx in kf.split(train_data):
            k_average_accuracy[k] += classification_accuracy(
                KNearestNeighbor(np.array([train_data[i] for i in train_indx]),
                                 np.array([train_labels[i] for i in train_indx])),
                k, [train_data[i] for i in test_indx], [train_labels[i] for i in test_indx])
        k_average_accuracy[k] /= 10
        print("[", k, "]", "Cross validation accuracy: ", k_average_accuracy[k])  # Debugging
    best_k = max(k_average_accuracy.keys(), key=lambda key: k_average_accuracy[key])
    return best_k, k_average_accuracy[best_k]

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    predicted_labels = [knn.query_knn(eval_datum, k) for eval_datum in eval_data]
    return sum(1 for i, label in enumerate(predicted_labels) if eval_labels[i] == label)/len(eval_data)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    best_k, cross_validation_avg_accuracy = cross_validation(train_data, train_labels)
    print("Best K: ", best_k, "\nCross validation accuracy: ", cross_validation_avg_accuracy)

    knn = KNearestNeighbor(train_data, train_labels)
    print("Train accuracy: ", classification_accuracy(knn, best_k, train_data, train_labels))
    print("Test accuracy: ", classification_accuracy(knn, best_k, test_data, test_labels))

    knn = KNearestNeighbor(train_data, train_labels)
    print("K == 1, Train accuracy: ", classification_accuracy(knn, 1, train_data, train_labels))

if __name__ == '__main__':
    main()
