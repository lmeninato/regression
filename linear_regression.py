import numpy as np


def linear_regression(x, y, learning_rate=0.01, n_iterations=100):
    n, m = x.shape[0], x.shape[1]
    theta = np.ones(m)/2
    for i in range(n_iterations):
        for j in range(theta.size):
            delta = learning_rate*gradient(theta, x, y, j)
            theta[j] = theta[j] + delta
    return theta


def gradient(theta, x, y, j):
    res = 0
    n = x.shape[0]
    m = x.shape[1]
    for i in range(n):
        res += (1/m)*(y[i] - theta.dot(x[i]))*x[i, j]
    return res


def linear_regression_matrix(x, y):
    return np.linalg.inv(x.T @ x) @ x.T @ y


if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    all_data = fetch_california_housing()

    data = all_data['data']
    target = all_data['target']
    features = all_data['feature_names']

    subset_data = data[:, [0, 2]]
    subset_data = np.insert(subset_data, 0, 1, axis=1)

    theta_hat = linear_regression(subset_data, target)
    print(theta_hat)
    new_theta_hat = linear_regression_matrix(subset_data, target)
    print(new_theta_hat)

