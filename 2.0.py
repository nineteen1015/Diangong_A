import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载与预处理（仅对目标变量归一化）
def load_data(file_path, target_col=0):
    data = pd.read_csv(file_path, index_col=0)
    # 单独提取目标列进行归一化
    target = data.iloc[:, target_col].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_target = scaler.fit_transform(target)
    # 合并回原始数据结构（保持其他特征不变）
    data.iloc[:, target_col] = scaled_target.flatten()
    return data.values, scaler

# 定义数据集
class SolarDataset(Dataset):
    def __init__(self, data, seq_length=96):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length, :]
        y = self.data[idx + self.seq_length, 0]  # 假设目标变量在第一列
        return x, y

# 修正后的Transformer-LSTM模型
class TransformerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.transformer(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze()

# 参数设置
BATCH_SIZE = 32
SEQ_LENGTH = 96
EPOCHS = 50

# 数据准备（示例路径需替换）
train_data, train_scaler = load_data('train_data.csv')
test_data, _ = load_data('test_data.csv')

# 创建DataLoader
train_dataset = SolarDataset(train_data, SEQ_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型
model = TransformerLSTM(
    input_dim=train_data.shape[1],
    hidden_dim=64,
    num_heads=4,
    num_layers=2
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 训练循环
for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.4f}')

# 测试集评估
test_dataset = SolarDataset(test_data, SEQ_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

y_true, y_pred = [], []
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        pred = model(x)
        y_pred.extend(pred.numpy())
        y_true.extend(y.numpy())

# 反归一化（仅处理目标变量）
y_true = train_scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()
y_pred = train_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()

# 计算指标
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
mae = np.mean(np.abs(y_true - y_pred))
print(f'RMSE: {rmse:.2f} MW')
print(f'MAE: {mae:.2f} MW')

# 可视化结果
# 1. 预测结果对比图
plt.figure(figsize=(12, 6))
plt.plot(y_true[:200], label='真实值', marker='o', markersize=3)
plt.plot(y_pred[:200], label='预测值', linestyle='--', alpha=0.8)
plt.xlabel('时间步')
plt.ylabel('发电功率 (MW)')
plt.title('光伏发电功率预测结果对比（前200个样本）')
plt.legend()
plt.grid(True)
plt.savefig('prediction_comparison.png', dpi=300)
plt.show()

# 2. 残差分布图
residuals = y_true - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('预测残差 (MW)')
plt.ylabel('频数')
plt.title('预测残差分布')
plt.grid(True)
plt.savefig('residual_distribution.png', dpi=300)
plt.show()

# 3. 真实值-预测值散点图
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='w')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
plt.xlabel('真实值 (MW)')
plt.ylabel('预测值 (MW)')
plt.title('真实值 vs 预测值')
plt.grid(True)
plt.savefig('true_vs_pred.png', dpi=300)
plt.show()

# 4. 误差分析图（误差绝对值随时间变化）
plt.figure(figsize=(12, 6))
abs_errors = np.abs(residuals)
plt.plot(abs_errors, alpha=0.7)
plt.xlabel('时间步')
plt.ylabel('绝对误差 (MW)')
plt.title('预测绝对误差变化趋势')
plt.grid(True)
plt.savefig('error_trend.png', dpi=300)
plt.show()
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
#
#
# # 数据加载与预处理
# def load_data(file_path):
#     data = pd.read_csv(file_path, index_col=0)
#     # 数据标准化
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(data)
#     return scaled_data, scaler
#
#
# # 定义数据集（增加归一化处理）
# class SolarDataset(Dataset):
#     def __init__(self, data, seq_length=96):
#         self.data = torch.FloatTensor(data)
#         self.seq_length = seq_length
#
#     def __len__(self):
#         return len(self.data) - self.seq_length - 1
#
#     def __getitem__(self, idx):
#         x = self.data[idx:idx + self.seq_length, :]
#         y = self.data[idx + self.seq_length, 0]
#         return x, y
#
#
# # 修正后的Transformer-LSTM模型
# class TransformerLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
#         super().__init__()
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=input_dim,
#                 nhead=num_heads,
#                 batch_first=True  # 使用batch_first模式
#             ),
#             num_layers=num_layers
#         )
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         # Transformer处理
#         x = self.transformer(x)  # (batch, seq, features)
#
#         # LSTM处理
#         _, (h_n, _) = self.lstm(x)
#
#         # 取最后一层的最后一个隐藏状态
#         out = self.fc(h_n[-1])
#         return out.squeeze()
#
#
# # 参数设置
# BATCH_SIZE = 32
# SEQ_LENGTH = 96
# EPOCHS = 50
#
# # 数据准备（示例路径，需替换实际路径）
# train_data, train_scaler = load_data('train_data.csv')
# test_data, _ = load_data('test_data.csv')
#
# # 创建DataLoader
# train_dataset = SolarDataset(train_data, SEQ_LENGTH)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
# # 初始化模型
# model = TransformerLSTM(
#     input_dim=train_data.shape[1],
#     hidden_dim=64,
#     num_heads=4,
#     num_layers=2
# )
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.MSELoss()
#
# # 训练循环
# for epoch in range(EPOCHS):
#     total_loss = 0
#     for x, y in train_loader:
#         optimizer.zero_grad()
#         pred = model(x)
#         loss = criterion(pred, y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f'Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.4f}')
#
# # 测试集评估
# test_dataset = SolarDataset(test_data, SEQ_LENGTH)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
#
# y_true, y_pred = [], []
# model.eval()
# with torch.no_grad():
#     for x, y in test_loader:
#         pred = model(x)
#         y_pred.extend(pred.numpy())
#         y_true.extend(y.numpy())
#
# # 反归一化（假设目标变量在第一个位置）
# y_true = train_scaler.inverse_transform(np.array(y_true).reshape(-1, 1))[:, 0]
# y_pred = train_scaler.inverse_transform(np.array(y_pred).reshape(-1, 1))[:, 0]
#
# # 计算指标
# rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
# mae = np.mean(np.abs(y_true - y_pred))
#
# print(f'RMSE: {rmse:.2f} MW')
# print(f'MAE: {mae:.2f} MW')