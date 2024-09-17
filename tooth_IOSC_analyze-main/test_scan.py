import torch.cuda
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import regress_SCAN_pytorch as rg
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def map2idx(x):
    quadrant = int(x[0])  # 第一位是象限
    tooth = int(x[1])  # 第二位是牙齿列号
    # 将象限和列号映射到0-13
    column = 0
    if quadrant == 1 or quadrant == 4:
        column = 7 - tooth
    else:
        column = 6 + tooth
    return column


def process_teeth_positions_2(x):
    # print(x)
    positions_list = x.split('&')
    pos = positions_list[0]
    damaged_pos = np.zeros(14)
    column = map2idx(pos)
    damaged_pos[column] = 1
    if len(positions_list) > 1:
        pos = positions_list[1]
        column2 = map2idx(pos)
        r1 = min(column, column2)
        r2 = max(column, column2)
        for idx in range(r1, r2 + 1):
            damaged_pos[idx] = 1
    max_range = 14 - np.sum(damaged_pos)
    return pd.Series([damaged_pos, max_range], index=['牙位', 'max_range'])


def process_teeth_positions(x):
    positions_list = x.split('&')
    pos_vec = np.zeros(28)  # 创建一个长度为28的向量
    damaged_pos = np.zeros(14)
    for pos in positions_list:
        quadrant = int(pos[0])  # 第一位是象限
        tooth = int(pos[1])  # 第二位是牙齿列号
        # 将象限和列号映射到0-13
        column = 0
        if quadrant == 1 or quadrant == 4:
            column = 7 - tooth
        else:
            column = 6 + tooth
        damaged_pos[column] = 1
        index = (quadrant - 1) * 7 + (tooth - 1)  # 映射到0-27的范围内
        pos_vec[index] = 1
    max_range = 14 - np.sum(damaged_pos)
    return pd.Series([pos_vec, max_range], index=['牙位', 'max_range'])


# 加载模型

model = rg.Net()  # 创建一个新的模型实例
model.load_state_dict(torch.load('model.pth'))
model.to(device)
# 准备输入数据
tooth_position = '12&22'
best_range = [-1, 10]
scan_ranges = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
predicted_RMS = []
toothData = pd.DataFrame({
    '牙位': tooth_position,
    '扫描范围': scan_ranges
})
tooth_processed = toothData['牙位'].apply(process_teeth_positions_2)
toothData['max_range'] = tooth_processed['max_range'][0]

for i in range(14):
    toothData[f'tooth_{i + 1}'] = tooth_processed['牙位'][0][i]
toothData = toothData.drop('牙位', axis=1)
print(toothData['max_range'][0])
input_data_scaled = rg.scaler_X.transform(toothData)
input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32).to(device)
# 进行预测
pre_x = []
model.eval()
with torch.no_grad():
    prediction = model(input_tensor)
    prediction = prediction.cpu().numpy().tolist()
    # print(type(prediction[0][0]))
    for i, pred in enumerate(prediction):
        pre = rg.scaler_Y.inverse_transform(np.asarray(prediction[i][0]).reshape(-1, 1))
        if scan_ranges[i] <= toothData['max_range'][0]:
            predicted_RMS.append(prediction[i][0])
            pre_x.append(scan_ranges[i])
            print(pre[0][0], end=" ")
    print()
    # idx=np.argmin(prediction)
    # print(f'最优扫描范围是{best_range[0] + 1},扫描误差：{best_range[1]}')

pred_RMS = np.asarray(predicted_RMS)
pred_RMS = rg.scaler_Y.inverse_transform(pred_RMS.reshape(-1, 1)).flatten()
best_range[1] = np.min(pred_RMS)  # 找到列表中的最小值
best_range[0] = pre_x[np.argmin(pred_RMS)]  # 找到最小值的索引
print(f'最优扫描范围是{best_range[0]},扫描误差：{best_range[1]}')

data = pd.read_csv("data.csv")
filtered_data = data[data['牙位'] == tooth_position]
# print(filtered_data)
ori_x = []
ori_rms = []
sum = 0
for idx, row in filtered_data.iterrows():
    # if (idx + 1) % 8 == 0:
    #     sum += row['RMS']
    #     ori_rms.append(sum / 8.0)
    #     ori_x.append(row['扫描范围'])
    #     sum = 0
    # else:
    #     sum += row['RMS']
    ori_rms.append(row['RMS'])
    ori_x.append(row['扫描范围'])
print(ori_x)
print(ori_rms)
X = np.asarray(pre_x)
y = pred_RMS
# 多项式拟合
degree = 4  # 多项式的阶数
coeff_upper = np.polyfit(X, y, degree)
smooth_curve = np.polyval(coeff_upper, X)

plt.figure(figsize=(10, 6))
plt.scatter(ori_x, ori_rms, label='ori', color='red')
# plt.scatter(pre_x, pred_RMS, label='pre', marker='o', color='blue')
plt.scatter(best_range[0], np.polyval(coeff_upper,best_range[0]), label='best', marker='o', color='black')
# plt.annotate("best_range: " + str(best_range[0] + 1), xy=(best_range[0] + 1, best_range[1]), xytext=(5, 10),
#              textcoords='offset points')

plt.plot(X, smooth_curve, label='fitted', linestyle='--', color='blue')

# 获取y轴的数据范围
y_min, y_max = plt.ylim()

# 将y轴的显示范围扩大成两倍
plt.ylim(0, y_max * 1.1)
plt.xlabel('scan range')
plt.ylabel('rms')
plt.title('visual ' + tooth_position)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
