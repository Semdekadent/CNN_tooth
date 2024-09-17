import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据样本
df = pd.read_csv('AFM.csv')  # 根据实际文件名修改
# 筛选出抛光度为120的数据
df = df[df['抛光程度(mesh)'] == 120]
print(df)
# 假设你的数据存储在pandas DataFrame中，名为df
# 首先，我们需要将数据分割为特征和目标
features = df[['酸蚀时间(s)']]
targets = df[['Ra(nm)']]

# 之后，我们将数据分为训练集和测试集
features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2,
                                                                              random_state=42)


# 创建一个对特征进行对数变换的函数
def log_transform(x):
    return np.log1p(x)
    # return x


# 因为“抛光程度(mesh)”是类别变量，我们需要对它进行独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', FunctionTransformer(log_transform), ['酸蚀时间(s)']),
    ],
)

# 为每个目标变量指定一个不同的模型
estimators = [
    Ridge(),  # 用于预测 'Ra(nm)'
    RandomForestRegressor(),  # 用于预测 'length(μm)'
    LinearRegression()  # 用于预测 'Angle(°)'
]

# 创建一个管道来进行预处理和模型拟合
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# 使用训练数据拟合模型
model.fit(features_train, targets_train)

# 使用测试集进行预测
predictions = model.predict(features_test)

# 计算并打印MSE
mse = mean_squared_error(targets_test, predictions, multioutput='raw_values')
print('Mean squared error: ', mse)

# 计算并打印R² score
r2 = r2_score(targets_test, predictions, multioutput='raw_values')
print('R2 score: ', r2)

print(predictions)

predicted_ra = predictions

# 酸蚀时间和抛光程度的真实值
actual_acid_etching_time = features_test['酸蚀时间(s)'].values
# actual_polishing_degree = features_test['抛光程度(mesh)'].values

# Ra的真实值
actual_ra = targets_test['Ra(nm)'].values
print(actual_ra)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)  # 调整画布大小和子图数量
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax1.scatter(actual_acid_etching_time, actual_ra)
ax1.set_xlabel('time(s)')
ax1.set_ylabel('RA(nm)')
ax1.set_title('origins')
ax2.scatter(actual_acid_etching_time, predicted_ra)
ax2.set_xlabel('time(s)')
ax2.set_ylabel('RA(nm)')
ax2.set_title('predictions')

plt.show()
