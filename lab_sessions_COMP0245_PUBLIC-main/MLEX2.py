import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import matplotlib.pyplot as plt

# 加载波士顿房价数据集
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
X = data
y = target

# 添加常数项
X = np.hstack([np.ones((X.shape[0], 1)), X])

# 拟合模型
model = LinearRegression().fit(X, y)
coefficients = model.coef_
intercept = model.intercept_

# 计算 R^2
y_pred = model.predict(X)
RSS = np.sum((y - y_pred) ** 2)  # Residual Sum of Squares
TSS = np.sum((y - np.mean(y)) ** 2)  # Total Sum of Squares
R_squared = 1 - (RSS / TSS)
n = X.shape[0]
k = X.shape[1] - 1  # 不包括常数项

# 计算 F 统计量
F_statistic = (R_squared / k) / ((1 - R_squared) / (n - k - 1))
print("F-statistic:", F_statistic)

adjusted_r_squared = 1 - ((1 - R_squared) * (n - 1)) / (n - k - 1)
print("Adjusted R-squared:", adjusted_r_squared)

# 计算标准误差
MSE = RSS / (n - k - 1)  # Mean Squared Error
standard_errors = np.sqrt(MSE * np.diag(np.linalg.inv(np.dot(X.T, X))))

# 置信区间
confidence_intervals = np.array([
    coefficients - 1.96 * standard_errors,
    coefficients + 1.96 * standard_errors
]).T

print("Confidence intervals for coefficients:\n", confidence_intervals)

# 新观察数据（例如取第一个样本）
new_observation = np.array([[1] + X[0, 1:].tolist()])  # 添加常数项
predicted_value = model.predict(new_observation)

# 计算预测标准误差
predicted_std_err = np.sqrt(MSE * (1 + np.dot(new_observation, np.linalg.inv(np.dot(X.T, X))).dot(new_observation.T)))

# 计算预测区间
prediction_interval = (predicted_value - 1.96 * predicted_std_err, predicted_value + 1.96 * predicted_std_err)
print("Predicted value:", predicted_value[0])
print("Prediction intervals:", prediction_interval)

# # 绘制散点图
# plt.figure(figsize=(10, 6))
# RM = X[:, 5] 
# plt.scatter(RM, y, color='blue', alpha=0.5, label='实际数据')

# # 绘制回归曲线
# RM_range = np.linspace(RM.min(), RM.max(), 100).reshape(-1, 1)
# predicted_prices = model.predict(RM_range)
# plt.plot(RM_range, predicted_prices, color='red', linewidth=2, label='预测曲线')

# # 添加标题和标签
# plt.title('Boston Housing Data: RM vs. Price')
# plt.xlabel('Number of Rooms (RM)')
# plt.ylabel('House Price (MEDV)')
# plt.legend()
# plt.grid(True)
# plt.show()
