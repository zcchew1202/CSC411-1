'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = np.zeros((10, 64))
    alpha = 2
    beta = 2
    for k in range(10):
        for i in range(64):
            N_c = sum([x_n[i] for n, x_n in enumerate(train_data) if train_labels[n] == k])
            N = sum([1 for n, x_n in enumerate(train_data) if train_labels[n] == k])
            eta[k][i] = (N_c + alpha - 1)/(N + alpha + beta - 2)
    return eta

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    class_images = np.concatenate([np.reshape(class_image, (-1, 8)) for class_image in class_images], axis=1)
    plt.imshow(class_images, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for k in range(10):
        for i in range(64):
            generated_data[k][i] = np.random.binomial(1, eta[k][i])
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    gen_log_likes = []
    for digit in bin_digits:
        k_likes = []
        for k in range(10):
            like = 1
            for i in range(64):
                like *= (eta[k][i]**digit[i]) * ((1-eta[k][i])**(1-digit[i]))
            k_likes.append(np.log(like))
        gen_log_likes.append(k_likes)
    return gen_log_likes

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gen_log_likes = generative_likelihood(bin_digits, eta)
    cond_log_likes = []
    for gen_log in gen_log_likes:
        den = np.log(sum([np.exp(gen_k) for gen_k in gen_log]) * (1/10))
        pixel_log_likes = []
        for gen_k_log in gen_log:
            pixel_log_likes.append(gen_k_log + np.log(1/10) - den)
        cond_log_likes.append(pixel_log_likes)
    return cond_log_likes

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihoods = conditional_likelihood(bin_digits, eta)
    total_likelihoods = 0
    for n_indx, cond_like in enumerate(cond_likelihoods):
        total_likelihoods += cond_like[int(labels[n_indx])]
    return total_likelihoods / len(bin_digits)

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihoods = conditional_likelihood(bin_digits, eta)
    return [cond_like.index(max(cond_like)) for cond_like in cond_likelihoods]

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)

    # Evaluation
    plot_images(eta)
    generate_new_data(eta)

    avg_cond_like_train = avg_conditional_likelihood(train_data, train_labels, eta)
    avg_cond_like_test = avg_conditional_likelihood(test_data, test_labels, eta)
    classes_train = classify_data(train_data, eta)
    classes_test = classify_data(test_data, eta)
    class_accuracy_train = sum(1 for indx, k in enumerate(classes_train) if train_labels[indx] == k) / len(train_data)
    class_accuracy_test = sum(1 for indx, k in enumerate(classes_test) if test_labels[indx] == k) / len(test_data)

    print("Average conditional likelihood ->", "\nTrain: ", avg_cond_like_train, "\nTest: ", avg_cond_like_test)
    print("Classification accuracy->", "\nTrain: ", class_accuracy_train, "\nTest: ", class_accuracy_test)

if __name__ == '__main__':
    main()
