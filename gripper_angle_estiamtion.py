from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

data = np.array([[62, 30], [61, 45], [58, 60], [56, 75], [51, 90],
                    [46, 105], [39, 120], [31, 135], [22, 150], [14, 165], [2,180]])

x = data[:, 0].reshape(-1,1)
y = data[:, 1]

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(X_poly, y)

coef = model.coef_
intercept = model.intercept_

print(coef, intercept)


y_pred = model.predict(X_poly)
mae = mean_absolute_error(y, y_pred)
print('MAE:', mae)

x_range = np.linspace(x.min(), x.max(), 300).reshape(-1,1)
x_range_poly = poly.transform(x_range)
y_range_pred = model.predict(x_range_poly)

plt.figure(figsize=(8,6))
plt.scatter(x, y, color='blue', label='Detection Data')
plt.plot(x_range, y_range_pred, color='red', linewidth=2, label='Prediction line (degree=3)')
plt.xlabel('Opening Width of Gripper(mm)')
plt.ylabel('Rotation Angle of Servo Motor (degree)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
