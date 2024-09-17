import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 从CSV文件中读取数据
data = pd.read_csv('data.csv')

# 对离散特征A进行独热编码
data = pd.get_dummies(data, columns=['牙位'])
data = data.drop(columns='样本')

# 分离特征和目标变量
X = data.drop('RMS', axis=1)
Y = data['RMS']

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 创建并训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# 保存模型
joblib.dump(model, 'model.pkl')

# 在测试集上进行预测
Y_pred = model.predict(X_test)
r2 = r2_score(Y_test, Y_pred)

# 计算并打印均方误差
mse = mean_squared_error(Y_test, Y_pred)
print(f'Mean Squared Error: {mse}')
print(f'r2_score: {r2}')

# 对新数据进行预测
# 加载模型
# loaded_model = joblib.load('model.pkl')

# 准备输入数据
# 这里的input_data是一个包含特征值的DataFrame，与训练时使用的特征一致
# input_data = pd.DataFrame({
#     '特征1': [value1],
#     '特征2': [value2],
#     # 添加其他特征...
#     '牙位_牙位1': [0],
#     '牙位_牙位2': [1],
#     # 添加其他独热编码特征...
# })
#
# # 进行预测
# prediction = loaded_model.predict(input_data)
#
# print(f'预测结果: {prediction}')
