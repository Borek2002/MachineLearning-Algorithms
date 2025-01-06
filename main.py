import copy, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

np.set_printoptions(precision=2)

X_train = np.array([[1240, 3, 1, 64], [1950, 3, 2, 17], [1720, 3, 2, 42], [1960, 3, 2, 15], [1310, 2, 1, 14], [864, 2, 1, 66],
          [1840, 3, 1, 17], [1030, 3, 1, 43], [3190, 4, 2, 87], [788, 2, 1, 80], [1200, 2, 2, 17], [1560, 2, 1, 18],
          [1430, 3, 1, 20], [1220, 2, 1, 15], [1090, 2, 1, 64], [848, 1, 1, 17], [1680, 3, 2, 23], [1770, 3, 2, 18],
          [1040, 3, 1, 44], [1650, 2, 1, 21], [1090, 2, 1, 35], [1320, 3, 1, 14], [1590, 0, 1, 20], [972, 2, 1, 73],
          [1100, 3, 1, 37], [1000, 2, 1, 51], [904, 3, 1, 55], [1690, 3, 1, 13], [1070, 2, 1, 102], [1420, 3, 2, 19],
          [1160, 3, 1, 52], [1940, 3, 2, 12], [1220, 2, 2, 74], [2480, 4, 2, 16], [1200, 2, 1, 18], [1840, 3, 2, 20],
          [1850, 3, 2, 57], [1660, 3, 2, 19], [1100, 2, 2, 97], [1780, 3, 2, 28], [2030, 4, 2, 45], [1780, 4, 2, 107],
          [1070, 2, 1, 102], [1550, 3, 1, 16], [1950, 3, 2, 16], [1220, 2, 2, 12], [1620, 3, 1, 16], [816, 2, 1, 58],
          [1350, 3, 1, 21], [1570, 3, 1, 14], [1490, 3, 1, 57], [1510, 2, 1, 16], [1100, 3, 1, 27], [1760, 3, 2, 24],
          [1210, 2, 1, 14], [1470, 3, 2, 24], [1770, 3, 2, 84], [1650, 3, 1, 19], [1030, 3, 1, 60], [1120, 2, 2, 16],
          [1150, 3, 1, 62], [816, 2, 1, 39], [1040, 3, 1, 25], [1390, 3, 1, 64], [1600, 3, 2, 29], [1220, 3, 1, 63],
          [1070, 2, 1, 102], [2600, 4, 2, 22], [1430, 3, 1, 59], [2090, 3, 2, 26], [1790, 4, 2, 49], [1480, 3, 2, 16],
          [1040, 3, 1, 25], [1430, 3, 1, 22], [1160, 3, 1, 53], [1550, 3, 2, 12], [1980, 3, 2, 22], [1060, 3, 1, 53],
          [1180, 2, 1, 99], [1360, 2, 1, 17], [960, 3, 1, 51], [1460, 3, 2, 16], [1450, 3, 2, 25], [1210, 2, 1, 15],
          [1550, 3, 2, 16], [882, 3, 1, 49], [2030, 4, 2, 45], [1040, 3, 1, 62], [1620, 3, 1, 16], [803, 2, 1, 80],
          [1430, 3, 2, 21], [1660, 3, 1, 61], [1540, 3, 1, 16], [948, 3, 1, 53], [1220, 2, 2, 12], [1430, 2, 1, 43],
          [1660, 3, 2, 19], [1210, 3, 1, 20], [1050, 2, 1, 65]])  # size, bedrooms, floors, age
y_train = np.array([300., 509.8, 394., 540., 415., 230., 560., 294., 718.2, 200.,
          302., 468., 374.2, 388., 282., 311.8, 401., 449.8, 301., 502.,
          340., 400.28, 572., 264., 304., 298., 219.8, 490.7, 216.96, 368.2,
          280., 526.87, 237., 562.43, 369.8, 460., 374., 390., 158., 426.,
          390., 277.77, 216.96, 425.8, 504., 329., 464., 220., 358., 478.,
          334., 426.98, 290., 463., 390.8, 354., 350., 460., 237., 288.3,
          282., 249., 304., 332., 351.8, 310., 216.96, 666.34, 330., 480.,
          330.3, 348., 304., 384., 316., 430.4, 450., 284., 275., 414.,
          258., 378., 350., 412., 373., 225., 390., 267.4, 464., 174.,
          340., 430., 440., 216., 329., 388., 390., 356., 257.8])
X_features = ['size(sqft)','bedrooms','floors','age']
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])


def predict(x, w, b):
    return np.dot(x, w) + b


def compute_cost(x, y, w, b, lambda_):
    sum = 0
    for i in range(x.shape[0]):
        sum += (predict(x[i], w, b) - y[i]) ** 2
    reg = (np.sum(w**2) * lambda_)/(2 * x.shape[0]) #compute regralization
    return sum / (2 * x.shape[0]) + reg

def compute_gradient(X, y, w, b, lambda_):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err

    # err = predict(X, w, b) - y
    # dj_db = np.sum(err)
    # dj_dw = np.dot(X.T, err)
    reg = (lambda_ / m) * w
    return dj_db/X.shape[0], dj_dw/X.shape[0] + reg

def gradient_descent(x, y, w, b, alpha, num_iter, lambda_):

    w = copy.deepcopy(w)
    costsJ = np.zeros(num_iter)

    for i in range(num_iter):
        dj_db, dj_dw = compute_gradient(x, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        costsJ[i] = compute_cost(x, y, w, b, lambda_)

    return w, b, costsJ

def zscore_normalization(x):
    mu = np.mean(x, axis=0) #średnia po kolumnach
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma , mu, sigma


def compare_feature_distributions(X, feature_names=None):
    """
    Porównuje rozkłady cech przed i po skalowaniu za pomocą histogramów i wykresów rozkładu normalnego.

    Args:
        X (ndarray): Dane wejściowe (m, n), gdzie m to liczba przykładów, a n to liczba cech.
        feature_names (list): Opcjonalne nazwy cech.
    """
    # Skalowanie danych
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std

    # Tworzenie siatki wykresów
    num_features = X.shape[1]
    fig, axes = plt.subplots(2, num_features, figsize=(4 * num_features, 8))

    if feature_names is None:
        feature_names = [f"Feature {i + 1}" for i in range(num_features)]

    # Iteracja przez cechy
    for i in range(num_features):
        # Dane przed skalowaniem
        sns.histplot(X[:, i], kde=False, bins=20, ax=axes[0, i], stat="count", color="blue")
        xmin, xmax = axes[0, i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean[i], std[i])
        axes[0, i].plot(x, p * len(X) * (xmax - xmin) / 20, 'r-', lw=2)  # Dopasowanie do histogramu
        axes[0, i].set_title(f"Original - {feature_names[i]}")
        axes[0, i].set_xlabel(feature_names[i])
        axes[0, i].set_ylabel("Count")

        # Dane po skalowaniu
        sns.histplot(X_scaled[:, i], kde=False, bins=20, ax=axes[1, i], stat="count", color="blue")
        xmin, xmax = axes[1, i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, 0, 1)  # Średnia 0 i odchylenie standardowe 1 po skalowaniu
        axes[1, i].plot(x, p * len(X_scaled) * (xmax - xmin) / 20, 'r-', lw=2)
        axes[1, i].set_title(f"Scaled - {feature_names[i]}")
        axes[1, i].set_xlabel(feature_names[i])
        axes[1, i].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

initial_w = np.zeros_like(w_init)
initial_b = 0.
x_norm, X_mu, X_sigma = zscore_normalization(X_train)
print(x_norm)
num_features = X_train.shape[1]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
compare_feature_distributions(X_train)

plt.tight_layout()
plt.show()
w_final, b_final, costs = gradient_descent(x_norm, y_train, initial_w, initial_b, 1.0e-1,1000, 0.7)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(costs)
ax2.plot(100 + np.arange(len(costs[100:])), costs[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
plt.show()