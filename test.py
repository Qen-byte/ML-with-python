import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([[1400], [1600], [1700]]).reshape(-1, 1)
y = np.array([245000, 312000, 279000])
model = LinearRegression().fit(X, y)

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.show()

print("Price for 2000 sq ft:", model.predict([[2000]]))
