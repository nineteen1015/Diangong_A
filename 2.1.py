import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# 数据加载与预处理
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=[0], index_col=0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return data.index, scaled_data, scaler, data.columns


# 数据集类
class SolarDataset(Dataset):
    def __init__(self, timestamps, data, seq_length=96):
        self.timestamps = timestamps.astype(np.int64) // 10 ** 9
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length, :]
        y = self.data[idx + self.seq_length, 0]
        ts = self.timestamps[idx + self.seq_length]
        return x, y, torch.tensor(ts, dtype=torch.float32)


# 模型类
class TransformerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=256,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=num_layers
        )
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.transformer(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze()


# 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs):
    loss_history = []
    device = next(model.parameters()).device

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch {epoch + 1:02d} | Loss: {avg_loss:.4f}')
    return loss_history


# 评估函数
def evaluate_model(model, test_loader, scaler, feature_names):
    model.eval()
    y_true, y_pred, timestamps = [], [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for x, y, ts in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(pred)
            timestamps.extend(ts.numpy())

    # 反标准化
    dummy = np.zeros((len(y_true), len(feature_names)))
    dummy[:, 0] = y_true
    y_true = scaler.inverse_transform(dummy)[:, 0]
    dummy[:, 0] = y_pred
    y_pred = scaler.inverse_transform(dummy)[:, 0]

    # 生成中文图表
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 误差分布图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(y_true - y_pred, bins=50, edgecolor='k')
    plt.title('预测误差分布')
    plt.xlabel('误差 (MW)')

    # 实际vs预测散点图
    plt.subplot(1, 2, 2)
    daytime_mask = (pd.Series(pd.to_datetime(timestamps)).dt.hour.between(6, 18))
    plt.scatter(y_true[daytime_mask], y_pred[daytime_mask], alpha=0.3)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.title('白天时段实际功率 vs 预测功率')
    plt.xlabel('实际功率 (MW)')
    plt.ylabel('预测功率 (MW)')

    plt.tight_layout()
    plt.savefig('result_analysis_zh.png')

    return np.array(y_true), np.array(y_pred), timestamps


if __name__ == "__main__":
    # 参数配置
    BATCH_SIZE = 128
    SEQ_LENGTH = 96
    EPOCHS = 50

    # 数据加载
    train_timestamps, train_data, scaler, features = load_data('train_data.csv')
    test_timestamps, test_data, _, _ = load_data('test_data.csv')

    # 数据集初始化
    train_dataset = SolarDataset(train_timestamps, train_data, SEQ_LENGTH)
    test_dataset = SolarDataset(test_timestamps, test_data, SEQ_LENGTH)

    # DataLoader配置
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

    # 模型初始化
    model = TransformerLSTM(
        input_dim=train_data.shape[1],
        hidden_dim=64,
        num_heads=4,
        num_layers=2
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 优化器配置
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.HuberLoss()

    # 训练与评估
    loss_history = train_model(model, train_loader, criterion, optimizer, EPOCHS)
    y_true, y_pred, timestamps = evaluate_model(model, test_loader, scaler, features)

    # 计算指标
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f'\n最终指标：')
    print(f'RMSE: {rmse:.2f} MW')
    print(f'MAE: {mae:.2f} MW')
    print(f'R²: {r2:.4f}')

    # 绘制训练曲线
    plt.figure()
    plt.plot(loss_history)
    plt.title('训练损失变化曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.savefig('训练损失图像.png')
