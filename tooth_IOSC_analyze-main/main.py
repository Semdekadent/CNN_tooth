import numpy as np
from pyswarm import pso

# 示例数据
data = {
    '样本': [1, 2, 3, 4, 5, 6, 7, 8],
    '牙位': [[11],[][2, 4, 6], [8, 10], [12]],
    'RMS值': [0.0259, 0.0224, 0.0227]
}

# 扫描范围的上下限
lower_bound = 2
upper_bound = 12


# 目标函数
def objective(x):
    scan_range = int(x[0])
    rms_values = []
    for i in range(scan_range):
        tooth_indices = data['牙位'][i]
        rms_values.extend([data['RMS值'][j - 2] for j in tooth_indices])
    return np.mean(rms_values)


# 使用PSO优化寻找最优解
dimensions = 1  # 扫描范围的维度
x0 = [2]  # 初始值
lb = [lower_bound]  # 下限
ub = [upper_bound]  # 上限
x_opt, f_opt = pso(objective, lb, ub, x0=x0)

print('最优扫描范围:', int(x_opt[0]))
print('最小扫描误差(RMS值):', f_opt)
