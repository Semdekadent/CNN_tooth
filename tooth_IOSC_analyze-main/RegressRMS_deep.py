import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import History
from sklearn.metrics import mean_squared_error, r2_score
# 从CSV文件中读取数据
data = pd.read_csv('data.csv')

# 对离散特征进行独热编码（如果有需要）
# 对离散特征A进行独热编码
data = pd.get_dummies(data, columns=['牙位'])
data = data.drop(columns='样本')

# 分离特征和目标变量
X = data.drop('RMS', axis=1)
Y = data['RMS']

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建神经网络模型
model = Sequential()
model.add(Dense(units=32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=1))

# 编译模型
model.compile(optimizer=Adam(), loss='mean_squared_error')

# 创建History回调函数
history = History()

# 训练模型
model.fit(X_train_scaled, Y_train, epochs=100, batch_size=32, verbose=0, callbacks=[history])

# 在测试集上进行预测
Y_pred = model.predict(X_test_scaled)
r2 = r2_score(Y_test, Y_pred)

# 计算并打印均方误差
mse = mean_squared_error(Y_test, Y_pred)
print(f'Mean Squared Error: {mse}')
print(f'r2_score: {r2}')

# 打印损失变化
loss_values = history.history['loss']
print('Loss变化:')
for epoch, loss in enumerate(loss_values, 1):
    print(f'Epoch {epoch}: {loss}')
