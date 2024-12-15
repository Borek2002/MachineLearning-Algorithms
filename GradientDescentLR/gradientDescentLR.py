import math, copy
import numpy as np
import matplotlib.pyplot as plt


def calculate_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def calculate_gradient(x, y, w, b):
    m = x.shape[0]
    cost = 0
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        fx = (w * x[i] + b) - y[i]
        dj_dw = fx * x[i]
        dj_db = fx
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db

def gradient_descent(x, y, init_w, init_b, alpha, iterations):
    w = init_w
    b = init_b
    costs = []
    for i in range(iterations):
        dw, db = calculate_gradient(x, y, w, b)
        tmp_w = w - alpha * dw
        tmp_b = b - alpha * db
        w = tmp_w
        b = tmp_b
        costs.append(calculate_cost(x, y, w, b))
    return w, b, costs

x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([2.0, 4.3, 6.3, 8.5, 10.0])
w_init = 0
b_init = 0
iterations = 1000000
alpha = 0.000001

w_final, b_final, costs = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)

fig, ax1= plt.subplots(1, 1, constrained_layout=True, figsize=(12,12))
ax1.plot(costs)
ax1.set_title("Cost vs. iteration")
ax1.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
plt.show()

print(w_final, b_final)
