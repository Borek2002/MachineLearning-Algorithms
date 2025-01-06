import copy, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import plot_date
from scipy.stats import norm

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def cost_function(X, y, w, b, lambda_):
    m = X.shape[0]
    cost = 0.0
    # for i in range(m):
    #     z = np.dot(X[i], w) + b
    #     g = sigmoid(z)
    #     cost += -y[i] * np.log(g) - (1-y[i]) * np.log(1 - g)

    z = np.dot(X, w) + b
    g = sigmoid(z)
    cost = np.sum(-y * np.log(g) - (1-y) * np.log(1 - g))
    reg = (np.sum(w ** 2) * lambda_) / (2 * m)  # compute regralization
    cost = cost/m + reg
    return cost

def compute_gradient(X, y, w, b, lambda_):
    m,n = X.shape

    z = np.dot(X, w) + b
    err = sigmoid(z) - y
    dj_dw = (1 / m) * np.dot(X.T, err)
    dj_db = (1 / m) * np.sum(err)
    reg = (lambda_ / m) * w
    return dj_dw + reg, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, iterations, lambda_):

    w = copy.deepcopy(w_in)
    b = b_in

    J_history = []
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history.append(cost_function(X, y, w, b, lambda_))
    return w, b, J_history

w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
print(cost_function(X_train, y_train, w_tmp, b_tmp, 0.7))
print(compute_gradient(X_train, y_train, w_tmp, b_tmp, 0.7))

w, b, J_history = gradient_descent(X_train, y_train, w_tmp, b_tmp,0.1, 10000, 0.7)
print(w, b)
# print(J_history)

# Wizualizacja kosztu
plt.plot(range(len(J_history)), J_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost over iterations')
plt.show()

# Wizualizacja granicy decyzyjnej
def plot_decision_boundary(X, y, w, b):
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, style=y, markers=['o', 's'])
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = -(w[0] * x1 + b) / w[1]
    plt.plot(x1, x2, color="red")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(X_train, y_train, w, b)