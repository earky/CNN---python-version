import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 设置随机种子以保证结果的可复现性
np.random.seed(0)

# 生成输入数据
x = np.linspace(-10, 10, 10)

#print("Input: ", y)

# 定义真实函数
def f(x):
    return 2 * x + 3

# 生成带噪声的目标数据
#y = f(x) + np.random.normal(0, 2, size=len(x))  # 添加噪声
y = f(x) + np.random.normal(0, 2, size=len(x))  # 添加噪声

# 拟合线性模型
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)

# 得到模型预测
y_pred = model.predict(x.reshape(-1, 1))

# 计算偏差
bias = np.mean(y_pred - f(x))


print("True value: ", y)
print("Predicted value: ", y_pred)
print("Bias: ", bias)

# 可视化
plt.scatter(x, y, color='blue', label='True value')
plt.plot(x, y_pred, color='red', label='Predicted value')
plt.legend()
plt.show()
