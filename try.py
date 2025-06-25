import numpy as np
from tslearn.barycenters import softdtw_barycenter
import matplotlib.pyplot as plt

# 步骤 2: 生成示例时间序列数据
# 创建三个不同长度的时间序列
time_series_1 = np.array([[1], [2], [3], [4], [5]])
time_series_2 = np.array([[2], [3], [4], [5], [6], [7]])
time_series_3 = np.array([[0], [1], [2], [3]])

# 将这些时间序列组合成一个列表
X = [time_series_1, time_series_2, time_series_3]
# 转换为适合 softdtw_barycenter 函数输入的三维数组
X = np.array(X, dtype=object)

# 步骤 3: 计算 SoftDTW 重心
# 设置 gamma 参数，控制 SoftDTW 的平滑程度
gamma = 0.1
# 调用 softdtw_barycenter 函数计算重心
barycenter = softdtw_barycenter(X, gamma=gamma)

# 步骤 4: 可视化结果
# 设置图片清晰度
plt.rcParams['figure.dpi'] = 100

# 绘制原始时间序列
for ts in X:
    plt.plot(ts, label='Original Time Series', linestyle='--', alpha=0.5)

# 绘制计算得到的重心
plt.plot(barycenter, label='SoftDTW Barycenter', color='red', linewidth=2)

# 添加图例和标题
plt.legend()
plt.title('SoftDTW Barycenter of Time Series')
plt.xlabel('Time Index')
plt.ylabel('Value')

# 显示图形
plt.show()