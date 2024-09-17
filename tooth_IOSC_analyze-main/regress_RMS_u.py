import torch.cuda
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


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
        for idx in range(column + 1, column2 + 1):
            damaged_pos[idx] = 1
    max_range = 14 - np.sum(damaged_pos)
    return pd.Series([damaged_pos, max_range], index=['牙位', 'max_range'])


def process_teeth_positions(x):
    # print(x)
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


class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet1D, self).__init__()
        self.encoder_l1 = nn.Sequential(
            nn.Conv1d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.encoder_l2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        self.encoder_l3 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        self.encoder_l4 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )
        self.decoder_l1 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        self.decoder_l2 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=2, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.out_layer = nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool_layer = nn.MaxPool1d(kernel_size=2)
        self.output = nn.Linear(16,1)


    def forward(self, x):
        # 编码器
        conv1 = self.encoder_l1(x)
        pool1 = self.maxpool_layer(conv1)
        conv2 = self.encoder_l2(pool1)
        pool2 = self.maxpool_layer(conv2)
        # 中间层
        middle = self.middle(pool2)
        # 解码器
        convt1 = self.decoder_l1(middle)
        concat1 = torch.cat([convt1, conv2], dim=0)
        conv3 = self.encoder_l3(concat1)

        convt2 = self.decoder_l2(conv3)
        concat2 = torch.cat([convt2, conv1], dim=0)
        conv4 = self.encoder_l4(concat2)

        out = self.out_layer(conv4)
        out = torch.relu(self.output(out))
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)

data = pd.read_csv('data.csv')
# data['牙位'] = data['牙位'].apply(process_teeth_positions)
data[['牙位', 'max_range']] = data['牙位'].apply(process_teeth_positions_2)
tooth_df = pd.DataFrame(data['牙位'].to_list(), columns=[f'tooth_{i + 1}' for i in range(14)])
# 将这些列添加到原始数据框中
data = pd.concat([data, tooth_df], axis=1)

# 现在可以删除原始的'牙位'列
data = data.drop(columns=['牙位'])
# 分离特征和变量
# 数据集划分
X = data.drop(['RMS', '样本'], axis=1)
Y = data['RMS']

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将数据转换为tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# print(X_train.shape[1])

if __name__ == '__main__':
    in_channels = 1
    out_channels = 1
    model = UNet1D(in_channels, out_channels).to(device)
    epoch_num = 300
    # epoch_num = 1
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # 定义正则化权重
    weight_decay = 1e-5  # 正则化权重系数
    # 训练模型
    model.train()
    best_train_loss = 222
    # writer=SummaryWriter()
    # 存储损失值的列表
    losses = []
    for epoch in range(epoch_num):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # print(inputs)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            # 添加 L2 正则化
            l2_regularization = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_regularization += torch.norm(param, 2)

            loss += weight_decay * l2_regularization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # writer.add_scalar('Loss', running_loss/len(train_loader), epoch)
        losses.append(running_loss)

        best_train_loss = min(running_loss / len(train_loader), best_train_loss)
        print(f'Epoch {epoch + 1}: Loss {running_loss / len(train_loader)}, best_train_loss:{best_train_loss}')

    # 在测试集上进行预测
    model.eval()
    Y_pred=[]
    with torch.no_grad():
        # Y_pred = model(X_test_tensor)
        # Y_pred = Y_pred.cpu().numpy()
        for x_test in X_test_scaled:
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
            y_pred=model(x_test_tensor.view(1,-1))
            y_pred=y_pred.cpu().numpy()
            Y_pred.append(y_pred)

    # 计算并打印均方误差
    Y_pred=pd.Series(Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'r2_score: {r2}')

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
