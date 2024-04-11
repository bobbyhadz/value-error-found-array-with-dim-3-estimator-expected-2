# ValueError: Found array with dim 3. Estimator expected 2

import numpy as np
from sklearn.linear_model import LinearRegression


x = np.array(
    [
        [[19.42, 43.4]],
        [[19.22, 43.9]],
        [[19.68, 44.1]],
        [[19.67, 44.2]],
        [[19.67, 44.2]]
    ]
)


# ✅ make the array 2-dimensional
x = x.reshape(-1, 2)

print(x.shape)  # 👉️ (5, 2)


y = np.array([[41.4], [42.9], [44], [45.1], [41.2]])

print(y.shape)  # 👉️ (5, 1)

model = LinearRegression()

reg = model.fit(x, y)

print(reg.score(x, y))  # 👉️ 0.08035714285714268