from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
import matplotlib.pyplot as plt
import numpy as np
import colorsys


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize_single_graph(X, y, features):
    fig = plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    marker_style = '.'
    marker_colors = [colorsys.hsv_to_rgb(i*1.0/feature_count, 1, 1) for i in range(feature_count)]

    for i in range(feature_count):
        Xi = [x[i] for x in X]
        plt.plot(Xi, y, label=features[i], marker=marker_style, color=marker_colors[i], linestyle='None')

    fig.canvas.set_window_title("CSC411_A1_Q1: Fig 1")
    plt.title("The Boston Housing Dataset Visualization")
    plt.ylabel("Target: Median Value of Home (in thousands)")
    plt.xlabel("Features")

    plt.legend()
    plt.tight_layout()
    plt.show()


def get_split_indices(set_size, training_fraction):
    """Returns two arrays of indices following len(train U test) == set_size, and train/set_size == training_faction"""
    assert 0 <= training_fraction <= 1, "training_fraction must be in the range [0,1]"

    train = np.random.choice(range(set_size), int(training_fraction*set_size), replace=False)
    test = [i for i in range(set_size) if i not in train]

    assert len(train) + len(test) == set_size, "len(train U test) != len(X)"
    return train, test


def fit_regression(X, y):
    """Does a linear regression fit of the data X and target y, returning w"""
    # Adding bias to the model by adding an element 1 to all x vectors
    X_biased = [np.append([1], x) for x in X]

    # Solving X^TXw = X^Ty, in Ax=B form, A=X^TX and B=X^Ty
    return np.linalg.solve(np.dot(np.transpose(X_biased), X_biased), np.dot(np.transpose(X_biased), y))


def predict(X, w):
    """Returns a prediction vector y_hat from X test data and w params"""
    # Notice that bias is added before computing, since w will have the extra element w0
    return [np.dot(w, np.append([1], x)) for x in X]


def main():
    X, y, features = load_data()
    visualize_single_graph(X, y, features)

    train_indices, test_indices = get_split_indices(X.shape[0], 0.8)

    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]

    w = fit_regression(X_train, y_train)
    y = predict(X_test, w)

    # sklearn.LinearRegression comparison
    skl_regr = linear_model.LinearRegression()
    skl_regr.fit(X_train, y_train)
    skl_y = skl_regr.predict(X_test)

    print("Mean Squared Error, Mean Absolute Error, Mean Squared Log Error")
    for y_hat, name in [(y, "RSV Linear Regression"), (skl_y, "sklearn.linear_model.LinearRegression")]:
        print(name + str(": ") + ', '.join([
            str(mean_squared_error(y_test, y_hat)),
            str(mean_absolute_error(y_test, y_hat)),
            str(mean_squared_log_error(y_test, y_hat))
        ]))


if __name__ == "__main__":
    main()
