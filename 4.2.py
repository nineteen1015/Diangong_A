import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import warnings

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 加载数据
train_data = pd.read_csv('train_data.csv', parse_dates=True, index_col=0)
test_data = pd.read_csv('test_data.csv', parse_dates=True, index_col=0)

# 定义特征列（根据数据列调整）
features = [
    'ghi', 'temp', 'cloud',
    'lag_96', 'lag_192',
    'hour_sin', 'hour_cos',
    'day_of_year_sin', 'day_of_year_cos',
    'ghi_rolling_24_mean', 'ghi_rolling_96_mean', 'ghi_rolling_672_mean',
    'temp_rolling_24_mean', 'temp_rolling_96_mean', 'temp_rolling_672_mean'
]

# 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[features])
X_test = scaler.transform(test_data[features])
y_train = train_data['实际功率']
y_test = test_data['实际功率']

# 训练模型
model = lgb.LGBMRegressor(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=200,
    reg_alpha=0.1
).fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
    return mae, rmse, r2

print("=== 模型性能 ===")
evaluate(y_test, y_pred, "光伏功率预测模型")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.title('特征重要性')
plt.tight_layout()
plt.savefig('特征重要性.png', dpi=300)

# 预测结果可视化样例
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:24*7], label='实际功率')
plt.plot(y_pred[:24*7], label='预测功率', linestyle='--')
plt.title('一周预测结果对比')
plt.xlabel('时间步长')
plt.ylabel('功率（MW）')
plt.legend()
plt.savefig('预测结果对比.png', dpi=300)