from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import colorsys


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
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


def fit_regression(X,Y):
    # TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    raise NotImplementedError()


def main():
    # Load the data
    X, y, features = load_data()

    # Visualize the features
    visualize(X, y, features)

    # TODO: Split data into train and test

    # Fit regression model
    w = fit_regression(X, y)

    # Compute fitted values, MSE, etc.


if __name__ == "__main__":
    main()
