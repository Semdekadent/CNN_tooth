import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib

# 假设你的数据已经读取到 DataFrame 中
data = pd.read_csv('data_ra_reviced.csv')
data = data.drop(['实验样品', 'length(μm)', 'Angle(°)'], axis=1)
# print(data)
# 将抛光程度的三种情况转换为三个二值特征
numeric_features = ['酸蚀时间(s)']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_features = ['抛光程度(mesh)']
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder())])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features),
], remainder='passthrough')
# columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [2])], remainder='passthrough')
data = preprocessor.fit_transform(data)
# print(data)
data = np.array(data, dtype=float)
# print(data)
# 通过 '酸蚀时间' 和 '抛光程度' 来预测 'Ra'
X = data[:, :-1]
y = data[:, -1]

# print(X)
# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 预测5%和95%分位数
quantiles = [0.05, 0.95]

# 创建两个模型，每个模型预测一个分位值【上界/下界】
models = []
for i in quantiles:
    model = GradientBoostingRegressor(loss='quantile', alpha=i)
    model.fit(X_train, y_train)
    models.append(model)

# 用训练好的模型预测测试集，得到预测的范围
lower_bound = models[0].predict(X_test)
upper_bound = models[1].predict(X_test)


# print(f'upper_bound:{upper_bound}')
# print(f'lower_bound:{lower_bound}')


def quantile_absolute_error(y_true, y_pred, quantile):
    error = y_true - y_pred
    loss = np.where(y_true >= y_pred, quantile * error, (quantile - 1) * error)
    return np.mean(loss)


# 计算每个模型的QAE
qae_lower = quantile_absolute_error(y_test, lower_bound, quantiles[0])
qae_upper = quantile_absolute_error(y_test, upper_bound, quantiles[1])
coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
print(f'coverage: {coverage}')

# 保存模型
joblib.dump(models,'model_1.pkl')

''' 预测新点
new_data_point = [[1, 0, 0, 30]]
l_b = models[0].predict(new_data_point)
u_b = models[1].predict(new_data_point)
print(l_b)
print(u_b)
'''
# 测试结果可视化
predicted_Ra = (lower_bound + upper_bound) / 2  # 简单取中值
actual_Ra = y_test
used = dict()
processed_x = []
processed_y = {}
predicted_y = []
lower_bound_processed = []
upper_bound_processed = []
for idx, i in enumerate(X_test):
    if str(i) not in used:
        used[str(i)] = 1
        processed_x.append(i)
        processed_y[str(i)] = actual_Ra[idx]
        predicted_y.append(predicted_Ra[idx])
        lower_bound_processed.append(lower_bound[idx])
        upper_bound_processed.append(upper_bound[idx])
    else:
        used[str(i)] += 1
        processed_y[str(i)] += actual_Ra[idx]

Ra = []
for i in processed_x:
    Ra.append(processed_y[str(i)] / used[str(i)])
# print(Ra)
ppx = []
for i in processed_x:
    ppx.append(i.tolist())
ppx = np.array(ppx)
print(f'ppx: {ppx}')
print(f'predicted_y: {predicted_y}')
x = ppx[:, -1]
print(f'x:{x}')
mesh = ppx[:, 0:-1]
mp = {
    '001': 500,
    '010': 240,
    '100': 120,
}
mesh_trans = []
for m in mesh:
    st = ""
    for d in m:
        d = str(int(d))
        st += d
    mesh_trans.append(mp[st])

# print(x)
print(f'mesh_trans: {mesh_trans}')
print(f'upper:{upper_bound_processed}')
print(f'lower:{lower_bound_processed}')
print(f'Ra:{Ra}')
fig = plt.figure(figsize=(12, 8))
ax_3d = fig.add_subplot(111, projection='3d')

ax_3d.scatter(x, mesh_trans, predicted_y, c='red')
ax_3d.scatter(x, mesh_trans, Ra, c='blue')

# 添加标签和标题
ax_3d.set_xlabel('time (s)')
ax_3d.set_ylabel('(mesh)')
ax_3d.set_zlabel('Ra (nm)')
# 扩展x、y、z轴范围
ax_3d.auto_scale_xyz(x, mesh_trans, predicted_y)
ax_3d.auto_scale_xyz(x, mesh_trans, Ra)
ax_3d.margins(x=0.2, y=0.2, z=0.2)

# 设置视角使x、y、z轴向后延伸展示
ax_3d.view_init(elev=30, azim=-60)
# plt.show()

fig = plt.figure(figsize=(12, 8))

ax_xz = fig.add_subplot(221)

# 绘制预测区间
plt.fill_between(range(len(Ra)), lower_bound_processed, upper_bound_processed, alpha=0.2, color='gray', label='predicted')

# 绘制真实值
plt.plot(range(len(Ra)), Ra, 'o-', markersize=8, label='True', color='blue')

plt.xlabel('No')
plt.ylabel('Ra')
plt.legend()
plt.title('')
# plt.grid(True)

# 创建xz平面子图
ax_xz = fig.add_subplot(222)
# 沿着y=120处的xz平面
y_120_indices = [i for i, y_val in enumerate(mesh_trans) if y_val == 120]
x_120 = [x[i] for i in y_120_indices]
z_120_1 = [predicted_y[i] for i in y_120_indices]
z_120_2 = [Ra[i] for i in y_120_indices]
ax_xz.scatter(x_120, z_120_1, c='red')
ax_xz.scatter(x_120, z_120_2, c='blue')
ax_xz.set_xlabel('time (s)')
ax_xz.set_ylabel('Ra (nm)')
ax_xz.set_title('mesh = 120')

# 沿着y=240处的xz平面
ax_xz = fig.add_subplot(223)
y_240_indices = [i for i, y_val in enumerate(mesh_trans) if y_val == 240]
x_240 = [x[i] for i in y_240_indices]
z_240_1 = [predicted_y[i] for i in y_240_indices]
z_240_2 = [Ra[i] for i in y_240_indices]
ax_xz.scatter(x_240, z_240_1, c='red')
ax_xz.scatter(x_240, z_240_2, c='blue')
ax_xz.set_xlabel('time (s)')
ax_xz.set_ylabel('Ra (nm)')
ax_xz.set_title('mesh = 240')

# 沿着y=500处的xz平面
ax_xz = fig.add_subplot(224)
y_500_indices = [i for i, y_val in enumerate(mesh_trans) if y_val == 500]
x_500 = [x[i] for i in y_500_indices]
z_500_1 = [predicted_y[i] for i in y_500_indices]
z_500_2 = [Ra[i] for i in y_500_indices]
ax_xz.scatter(x_500, z_500_1, c='red')
ax_xz.scatter(x_500, z_500_2, c='blue')
ax_xz.set_xlabel('time (s)')
ax_xz.set_ylabel('Ra (nm)')
ax_xz.set_title('mesh = 500')
# 调整子图的间距和位置
fig.tight_layout()
plt.show()
