import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCH_SIZE = 50
K = 500


class BatchSampler(object):
    """
    A (very) simple wrapper to randomly sample batches without replacement.
    You shouldn't need to touch this.
    """
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def _random_batch_indices(self, m=None):
        """
        Get random batch indices without replacement from the dataset.
        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        """
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        """
        Get a random batch without replacement from the dataset.
        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        """
        indices = self._random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch    


def load_data_and_init_params():
    """Load the Boston houses dataset and randomly initialise linear regression weights."""
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity (cos theta) between two vectors."""
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)


def lin_reg_gradient(X, y, w):
    """Compute gradient of linear regression model parameterized by w"""
    # Computing L(w)=(1/m)*(2*X^TXw + 2*X^Ty)
    return (1/X.shape[0])*(2*np.linalg.multi_dot([np.transpose(X), X, w]) + 2*np.dot(np.transpose(X), y))


def main():
    np.random.seed(0)  # Make runs repeatable

    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCH_SIZE)

    batch_gradients = list()
    for k in range(K):
        X_b, y_b = batch_sampler.get_batch()
        batch_gradients.append(lin_reg_gradient(X_b, y_b, w))

    mean_batch_gradient = np.array(batch_gradients).sum(axis=0)*(1/K)
    true_gradient = lin_reg_gradient(X, y, w)

    squared_distance_metric = np.linalg.norm(true_gradient-mean_batch_gradient)**2
    cosine_similarity_metric = cosine_similarity(true_gradient, mean_batch_gradient)

    print("Squared Distance: " + str(squared_distance_metric))
    print("Cosine Similarity: " + str(cosine_similarity_metric))

    variances = list()
    for m in range(1, 400):
        batch_sampler = BatchSampler(X, y, m)
        batch_gradients = list()
        for k in range(K):
            X_b, y_b = batch_sampler.get_batch()
            batch_gradients.append(lin_reg_gradient(X_b, y_b, w))
        variances.append(np.var([gradient[0] for gradient in batch_gradients]))

    plt.plot([np.log(m) for m in range(1, 400)], [np.log(var) for var in variances])
    plt.ylabel("Sample Variance of mini-batch gradient estimate (logscale)")
    plt.xlabel("m (logscale)")


if __name__ == '__main__':
    main()
