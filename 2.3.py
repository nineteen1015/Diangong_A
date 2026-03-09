import matplotlib
matplotlib.use('TkAgg')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return data.index, scaled_data, scaler, data.columns

class SolarDataset(Dataset):
    def __init__(self, timestamps, data, seq_length=96):
        self.timestamps = timestamps.astype(np.int64) // 10**9
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length, :]
        y = self.data[idx+self.seq_length, 0]
        ts = self.timestamps[idx+self.seq_length]
        return x, y, torch.tensor(ts, dtype=torch.float32)

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

def train_model(model, train_loader, criterion, optimizer, epochs):
    loss_history = []
    device = next(model.parameters()).device
    for epoch in range(epochs):
        model.train()
        total_loss = 0
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
        print(f'Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}')
    return loss_history

def evaluate_model(model, test_loader, scaler, feature_names):
    model.eval()
    y_true, y_pred = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(pred)

    # 反标准化
    dummy = np.zeros((len(y_true), len(feature_names)))
    dummy[:, 0] = y_true
    y_true = scaler.inverse_transform(dummy)[:, 0]
    dummy[:, 0] = y_pred
    y_pred = scaler.inverse_transform(dummy)[:, 0]

    # 可视化
    plt.figure(figsize=(15, 6))
    plt.plot(y_true[:24*4], label='实际功率', marker='o')
    plt.plot(y_pred[:24*4], label='预测功率', linestyle='--')
    plt.title('未来24小时功率预测对比')
    plt.xlabel('时间步 (15分钟间隔)')
    plt.ylabel('功率 (MW)')
    plt.legend()
    plt.savefig('短期预测对比.png', dpi=300)
    plt.close()

    return y_true, y_pred

if __name__ == "__main__":
    # 参数配置
    BATCH_SIZE = 128
    SEQ_LENGTH = 96  # 24小时历史数据
    EPOCHS = 50

    # 数据加载
    train_timestamps, train_data, scaler, features = load_data('train_data.csv')
    test_timestamps, test_data, _, _ = load_data('test_data.csv')

    # 数据集初始化
    train_dataset = SolarDataset(train_timestamps, train_data, SEQ_LENGTH)
    test_dataset = SolarDataset(test_timestamps, test_data, SEQ_LENGTH)

    # DataLoader
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

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
    y_true, y_pred = evaluate_model(model, test_loader, scaler, features)

    # 计算指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\nRMSE: {rmse:.2f} MW")
    print(f"MAE: {mae:.2f} MW")
    print(f"R²: {r2:.4f}")

    # 训练损失曲线
    plt.figure()
    plt.plot(loss_history)
    plt.title('训练损失变化曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('Huber损失值')
    plt.savefig('训练损失.png', dpi=300)