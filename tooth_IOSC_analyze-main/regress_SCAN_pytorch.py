import torch.cuda
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn.functional as F

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


# 定义神经网络模型
class N2et(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding=nn.Embedding(16,32)
        # self.fc1 = nn.Linear(X_train.shape[1], 32)
        self.Conv1=nn.Sequential(
            nn.Conv1d(16,32,3,stride=1,padding=1,bias=True),
            nn.BatchNorm1d(32),
            nn.Conv1d(32,32,3,stride=1,padding=1, bias=True),
            nn.BatchNorm1d(32),
            # nn.ReLU(),
        )
        self.Conv2=nn.Sequential(
            nn.Conv1d(32,64,3,padding=1,stride=1,bias=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64,64,3,padding=1,stride=1, bias=True),
            nn.BatchNorm1d(64),
            # nn.ReLU(),
        )
        self.Conv3=nn.Sequential(
            nn.Conv1d(64,32,3,stride=1,padding=1,bias=True),
            nn.BatchNorm1d(32),
            nn.Conv1d(32,32,3,stride=1,padding=1, bias=True),
            nn.BatchNorm1d(32),
        )
        self.deconv1=nn.Sequential(
            nn.ConvTranspose1d(64,32,kernel_size=2,stride=2),
            nn.BatchNorm1d(32),
        )
        # 32->32
        self.layer1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )
        # 32->64
        self.layer2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        # 64->64
        self.layer3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # 64->32
        self.layer4 = nn.Sequential(
            nn.Linear(64, 32),
            # nn.LayerNorm(16),
            nn.ReLU()
        )
        self.Conv= nn.Sequential(
            nn.Conv1d(32,1,1)
        )

        self.output = nn.Sequential(
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        conv1 = self.Conv1(x) #(bs,16,32) -> (bs, 32, 32)
        pool1 = F.max_pool1d(conv1,2) # (bs,32,32) -> (bs,32,16)
        
        conv2=self.Conv2(pool1) #(bs,32,16) -> (bs,64,16)
        
        convt1 = self.deconv1(conv2) # (bs,64,16) -> (bs,32,32)
        concat1=torch.cat([convt1,conv1],dim=1) # (bs,32,32) -> (bs,64,32)
        conv3=self.Conv3(concat1) # (bs,64,32) -> (bs,32,32)
        outout=self.output(self.Conv(conv3))
        return outout

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding=nn.Embedding(16,32)
        # self.fc1 = nn.Linear(X_train.shape[1], 32)
        self.Conv1=nn.Sequential(
            nn.Conv2d(1,4,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(4),
            # nn.ReLU(),            
            nn.Conv2d(4,4,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.Conv2=nn.Sequential(
            nn.Conv2d(4,8,3,padding=1,stride=1,bias=True),
            nn.BatchNorm2d(8),
            # nn.ReLU(),
            nn.Conv2d(8,8,3,padding=1,stride=1,bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        # 全连接层
        self.fc1 = nn.Linear(8*16*32, 128)
        self.fc2 = nn.Linear(128, 1)  # 输出层，输出一个值
        self.fc3 = nn.Linear(1*4*4,1)
        self.dropout = nn.Dropout(0.3)  # dropout层
        self.skip_connection=nn.Conv2d(1,8,kernel_size=1,stride=1)
        self.concat=nn.Sequential(
            nn.Conv2d(16,8,kernel_size=1,stride=1),
        )
        
    def forward(self, x):
        # x = self.embedding(x) # (bs,16) -> (bs,16,16)
        # x = x.unsqueeze(1) # (bs,16,8) -> (bs,1,16,8)
        # conv1 = self.Conv1(x) #(bs,1,16,8) -> (bs,4, 16, 8)
        # pool1 = F.max_pool2d(conv1,2) # (bs,4,8,4)
        
        # conv2=self.Conv2(pool1) #(bs,4,8,4) -> (bs,8,8,4)
        # pool2 = F.max_pool2d(conv2,2) # (bs,8,8,4) -> (bs,8,4,2)
        
        # x=pool2.view(pool2.size(0),-1) # (bs,8,4,2) -> (bs,8*4*2)
        # fc1=F.relu(self.fc1(x))
        # fc1=self.dropout(fc1)
        # output=self.fc2(fc1)
        # """ x=self.Conv3(pool2)
        # x=x.view(x.size(0),-1)
        # output=self.fc3(x) """
        # return output
        
        x = self.embedding(x) # (bs,16) -> (bs,16,32)
        x = x.unsqueeze(1) # (bs,16,32) -> (bs,1,16,32)
        conv1 = self.Conv1(x) #(bs,1,16,32) -> (bs,4, 16, 32)

        conv2=self.Conv2(conv1) #(bs,4,16,32) -> (bs,8,16,32)
        # conv2=conv2+self.skip_connection(x)
        o=torch.concat([conv2,self.skip_connection(x)],dim=1)
        conv2=self.concat(o)
        x=conv2.view(conv2.size(0),-1) # (bs,8,16,32) -> (bs,8*16*32)
        fc1=F.relu(self.fc1(x))
        fc1=self.dropout(fc1)
        output=self.fc2(fc1)
        """ x=self.Conv3(pool2)
        x=x.view(x.size(0),-1)
        output=self.fc3(x) """
        return output



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

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

# 划分训练集和测试集，再从训练集划分出验证集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)
# 特征缩放
""" scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
"""
scaler_Y = StandardScaler()
Y_train = scaler_Y.fit_transform(np.asarray(Y_train).reshape(-1, 1)).flatten()
Y_test = scaler_Y.transform(np.asarray(Y_test).reshape(-1, 1)).flatten()
Y_val = scaler_Y.transform(np.asarray(Y_val).reshape(-1, 1)).flatten()

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# 将数据转换为tensor
# X_train_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
# Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).to(device)
# X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
# Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)

X_train=X_train.values
# y_train=Y_train.values.reshape(-1,1)
X_test=X_test.values
# Y_test=Y_test.values.reshape(-1,1)
X_val=X_val.values
train_dataset = CustomDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataset=CustomDataset(X_test,Y_test)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)
val_dataset=CustomDataset(X_val,Y_val)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)

# 创建数据加载器
# train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# print(X_train.shape[1])

if __name__ == '__main__':

    model = Net().to(device)
    epoch_num = 100
    # 定义tensorboard
    writer = SummaryWriter('../tf-logs')
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00005)

    # 定义正则化权重
    weight_decay = 1e-5  # 正则化权重系数
    # 训练模型
    model.train()
    best_train_loss = 222
    # writer=SummaryWriter()
    # 存储损失值的列表
    losses = []
    best_loss=1e6
    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            # print(inputs)
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            # 添加 L2 正则化
            l2_regularization = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_regularization += torch.norm(param, 1).to(device)

            loss += weight_decay * l2_regularization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch + 1)
        losses.append(running_loss)
        train_loss=running_loss / len(train_loader)
        best_train_loss = min(train_loss, best_train_loss)
        # print(f'Epoch {epoch + 1}: Loss {running_loss / len(train_loader)}, best_train_loss:{best_train_loss}')
        #  测试集进行测试
        model.eval()
        total_loss=0
        for i_batch, sample_batched in enumerate(test_loader):
            inputs, targets = sample_batched
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_loss+=loss.item()
        mean_loss=total_loss/len(test_loader)
        writer.add_scalar('Loss/test', mean_loss, epoch + 1)
        print(f'Epoch {epoch + 1}: Train Loss:{train_loss}, Best Train Loss:{best_train_loss} Test Loss {mean_loss}')
        if mean_loss<best_loss:
            print(f">>>>>>>>>>Test Loss Decreased from {best_loss} to {mean_loss}, model saved in best_model.pth>>>>>>>>>>>>")
            best_loss=mean_loss
            torch.save(model.state_dict(), 'best_model.pth')

    writer.close()


# 在测试集上进行预测
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
total_loss=0
for i_batch, sample_batched in enumerate(val_loader):
    inputs, targets = sample_batched
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs.squeeze(), targets)
    total_loss+=loss.item()
mean_loss=total_loss/len(val_loader)
print(f'Val Loss {mean_loss}')

