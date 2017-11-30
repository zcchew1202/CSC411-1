import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum

        lr - learning rate
        beta - momentum hyperparameter
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.vel = 0.0

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        self.vel = self.beta * np.array(self.vel) + self.lr * np.array(grad)
        return params - self.vel


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        self.w[0] = 1  # Setting bias parameter to 1
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        loss = []
        for i, x in enumerate(X):
            loss.append(max([1 - y[i] * np.linalg.multi_dot([np.transpose(self.w), x]), 0]))
        return np.array(loss)

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        hinge_loss_grads = np.array([np.zeros(X[0].shape) if loss == 0 else -np.dot(y[i], X[i])
                                     for i, loss in enumerate(self.hinge_loss(X, y))])
        zero_bias_w = self.w
        zero_bias_w[0] = 0
        return (self.c/len(X)) * np.sum(np.array([zero_bias_w + loss_grad for loss_grad in hinge_loss_grads]), axis=0)

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        return np.array([1 if np.dot(np.transpose(self.w), x) >= 0 else -1 for x in X])

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets


def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]

    for _ in range(steps):
        # Optimize and update the history
        pass
    return w_history


def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.

    SVM weights can be updated using the attribute 'w'. i.e. 'svm.w = updated_weights'
    '''
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)
    svm = SVM(penalty, train_data.shape[1])
    for _ in range(iters):
        x_batch, y_batch = batch_sampler.get_batch()
        svm.w = optimizer.update_params(svm.w, svm.grad(x_batch, y_batch))
    return svm


if __name__ == '__main__':
    # Testing SGD with
    wt_b1 = [10.0]
    wt_b2 = [10.0]
    gdoptimizer_b1 = GDOptimizer(1.0, 0.0)
    gdoptimezer_b2 = GDOptimizer(1.0, 0.9)
    for i in range(200):
        wt_b1.append(gdoptimizer_b1.update_params(wt_b1[-1], 0.02 * wt_b1[-1]))
        wt_b2.append(gdoptimezer_b2.update_params(wt_b2[-1], 0.02 * wt_b2[-1]))

    plt.plot(wt_b1)
    plt.plot(wt_b2)
    plt.show()

    # Q2.3, applying SVM on 4-vs-9 digits on MNIST
    train_data, train_targets, test_data, test_targets = load_data()
    train_data = np.array([np.append([1], train) for train in train_data])
    test_data = np.array([np.append([1], test) for test in test_data])


    def do_everything(beta):
        svm = optimize_svm(train_data, train_targets, 1.0, GDOptimizer(0.05, beta), 100, 500)
        train_predictions = svm.classify(train_data)
        test_predictions = svm.classify(test_data)

        train_accuracy = (100.0/len(train_data)) * \
                         sum([1 if train_predictions[i] == train_targets[i] else 0 for i in range(len(train_data))])
        test_accuracy = (100.0/len(test_data)) * \
                        sum([1 if test_predictions[i] == test_targets[i] else 0 for i in range(len(test_data))])

        print("Beta == " + str(beta))
        print("Training set classification accuracy: " + str(train_accuracy))
        print("Test set classification accuracy: " + str(test_accuracy))

        plt.imshow(np.reshape(svm.w[1:], (-1, 28)), cmap='gray')
        plt.show()

    do_everything(0.0)
    do_everything(0.1)
