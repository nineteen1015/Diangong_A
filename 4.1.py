import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from pykrige.ok import OrdinaryKriging
import warnings

# 中文字体显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 加载数据
train_data = pd.read_csv('train_data.csv', parse_dates=True, index_col=0)
test_data = pd.read_csv('test_data.csv', parse_dates=True, index_col=0)
features = train_data.columns.drop('实际功率')


# 空间降尺度处理
def spatial_downscaling(df, target_col='ghi', resolution=0.01):
    grid_lon = np.arange(df['lon'].min(), df['lon'].max(), resolution)
    grid_lat = np.arange(df['lat'].min(), df['lat'].max(), resolution)

    ok = OrdinaryKriging(df['lon'], df['lat'], df[target_col], variogram_model='linear')
    z, _ = ok.execute('grid', grid_lon, grid_lat)

    df[f'{target_col}_high_res'] = z.ravel()[:len(df)]
    return df

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[features])
X_test = scaler.transform(test_data[features])
y_train = train_data['实际功率']
y_test = test_data['实际功率']

model_baseline = lgb.LGBMRegressor()
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)

# 评估指标
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
    return mae, rmse, r2

print("基准模型性能：")
mae_base, rmse_base, r2_base = evaluate(y_test, y_pred_baseline, "基准模型")

# 绘制预测结果对比图（示例）
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:200], label='实际功率', color='blue')
plt.plot(y_pred_baseline[:200], label='基准模型预测', color='red', linestyle='--')
plt.xlabel('时间步')
plt.ylabel('功率（标准化）')
plt.title('实际功率与预测值对比')
plt.legend()
plt.show()

# 误差分布图
errors_baseline = y_test - y_pred_baseline

plt.figure(figsize=(10, 5))
sns.histplot(errors_baseline, kde=True, label='基准模型误差', color='red')
plt.xlabel('预测误差')
plt.ylabel('频率')
plt.title('误差分布对比')
plt.legend()
plt.show()