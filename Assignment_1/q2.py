from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import logsumexp
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506, 1)), x), axis=1)  # add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']


def l2(A,B):  # helper function
    """
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    """
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B**2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist


def run_on_fold(x_test, y_test, x_train, y_train, taus):  # helper function
    """
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    """
    losses = np.zeros(taus.shape)
    for j, tau in enumerate(taus):
        predictions = LRLS(x_test, x_train, y_train, tau)
        losses[j] = ((predictions-y_test)**2).mean()
    return losses
 

def LRLS(test_data, x_train, y_train, tau, lam=1e-5):  # to implement
    """
    Input: test_data is a Mxd test matrix, of M test vectors
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat, an Mx1 vector of predictions
    """
    dist = l2(test_data, x_train)
    y_hat = list()
    for i in range(test_data.shape[0]):
        # Computing A
        ln_ai_den = logsumexp([dist[i, j]/(2*tau*tau) for j in range(x_train.shape[0])])
        ln_ai_numerators = [dist[i, j]/(2*tau*tau) for j in range(x_train.shape[0])]
        ln_ais = [ln_ai_num - ln_ai_den for ln_ai_num in ln_ai_numerators]
        A = np.diag([np.exp(ln_ai) for ln_ai in ln_ais])

        # Solving (X^TAX+lambdaI)w = X^TAy, in ax=b form, a=X^TAX+lambdaI and b=X^TAy
        w = np.linalg.solve(
            np.add(np.linalg.multi_dot([np.transpose(x_train), A, x_train]), lam * np.identity(d)),  # a=X^TAX+lambdaI
            np.linalg.multi_dot([np.transpose(x_train), A, y_train]))  # b=X^TAy

        y_hat.append(np.dot(w, test_data[i]))

    return np.array(y_hat)


def run_k_fold(x, y, taus, k):
    """
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    """
    idx = np.random.permutation(range(N))
    fold_size = int(N/k)
    k_folds_indices = [idx[i:i + fold_size] for i in range(0, N, fold_size)]
    k_folds_indices = k_folds_indices[0:k]  # There may be a last array with < k elements that we throw out
    losses = list()

    in_fold = lambda data_set: np.array([data_set[i] for i in kf_indx])
    not_in_fold = lambda data_set: np.array([data_set[i] for i in idx if i not in kf_indx])

    for kf_indx in k_folds_indices:
        losses.append(run_on_fold(x_test=in_fold(x), y_test=in_fold(y),
                                  x_train=not_in_fold(x), y_train=not_in_fold(y), taus=taus))

    return np.array([np.average([losses[j][i] for j in range(k)]) for i in range(len(losses[0]))])


if __name__ == "__main__":
    np.random.seed(0)  # Make runs repeatable

    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value.
    # Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    losses = run_k_fold(x, y, taus, k=5)
    plt.plot(losses)
    print("min loss = {}".format(losses.min()))
