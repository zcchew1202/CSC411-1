'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    for i in range(0, 10):
        means[i] = np.mean([train_data[j] for j in range(len(train_data)) if train_labels[j] == i], axis=0)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)
    for k in range(10):
        for i in range(64):
            for j in range(64):
                k_indx = [indx for indx in range(len(train_data)) if train_labels[indx] == k]
                N = len(k_indx)
                X_i = [train_data[indx][i] for indx in k_indx]
                X_j = [train_data[indx][j] for indx in k_indx]

                cov_i_j = (1/N) * sum([(X_i[n] - means[k][i]) * (X_j[n] - means[k][j]) for n in range(N)])
                covariances[k][i][j] = cov_i_j

                # Calculating covariance in other ways to verify correctness
                # cov_i_j_2 = (1/(N*N)) * sum([sum([(1/2) * (X_i[n1] - X_i[n2]) * (X_j[n1] - X_j[n2])
                #                                   for n2 in range(N)]) for n1 in range(N)])
                # cov_i_j_3 = np.cov(X_i, X_j)[0][1]
                # print(cov_i_j, ", ", cov_i_j_2, ", ", cov_i_j_3)

    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    cov_diagonals = []
    for k in range(10):
        cov_diagonals.append(np.reshape(np.diag(covariances[k]), (-1, 8)))

    all_concat = np.concatenate(np.log(cov_diagonals), axis=1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    ret = []
    for digit in digits:
        class_log_likelihoods = []
        for k in range(10):
            exp_part = np.exp((-1 / 2) * np.linalg.multi_dot([
                np.transpose(digit - means[k]), np.linalg.inv(covariances[k] + 0.01 * np.identity(64)),
                digit - means[k]]))
            class_log_likelihoods.append(np.log(
                ((2*np.pi)**(-64/2)) * (np.linalg.det(covariances[k] + 0.01*np.identity(64)) ** (-1/2)) * exp_part))
        ret.append(class_log_likelihoods)
    return ret

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    ret = []
    generative_likelihoods = generative_likelihood(digits, means, covariances)
    for gen in generative_likelihoods:
        den = np.log(sum(np.exp(gen_k) for gen_k in gen)*(1/10))
        digit_class_conditional_likelihoods = []
        for gen_k in gen:
            digit_class_conditional_likelihoods.append(gen_k + np.log(1/10) - den)
        ret.append(digit_class_conditional_likelihoods)
    return ret

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihoods = conditional_likelihood(digits, means, covariances)
    total_likelihoods = 0
    for n_indx, cond_like in enumerate(cond_likelihoods):
        total_likelihoods += cond_like[int(labels[n_indx])]
    return total_likelihoods / len(digits)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    ret = []
    for indx, cond_l in enumerate(cond_likelihood):
        ret.append(max([(k, prob) for k, prob in enumerate(cond_l)], key=lambda x: x[1])[0])
    return ret

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    plot_cov_diagonal(covariances)

    # Evaluation
    avg_cond_like_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    avg_cond_like_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    classes_train = classify_data(train_data, means, covariances)
    classes_test = classify_data(test_data, means, covariances)
    class_accuracy_train = sum(1 for indx, k in enumerate(classes_train) if train_labels[indx] == k)/len(train_data)
    class_accuracy_test = sum(1 for indx, k in enumerate(classes_test) if test_labels[indx] == k)/len(test_data)

    print("Average conditional likelihood ->", "\nTrain: ", avg_cond_like_train, "\nTest: ", avg_cond_like_test)
    print("Classification accuracy->", "\nTrain: ", class_accuracy_train, "\nTest: ", class_accuracy_test)

if __name__ == '__main__':
    main()
