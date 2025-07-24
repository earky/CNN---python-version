import numpy as np

d_t_d_w = np.array([1352, 1])  # 转换为NumPy数组
d_L_d_t = np.array([1, 10])  # 转换为NumPy数组

d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
print(d_t_d_w[np.newaxis].T)
print(d_L_d_w)