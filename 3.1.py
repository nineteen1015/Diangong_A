import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import warnings

# 中文字体显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 数据加载与预处理
def load_and_preprocess(train_path, test_path):
    # 加载数据
    train = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test = pd.read_csv(test_path, index_col=0, parse_dates=True)

    # 校验必要字段
    required_cols = ['实际功率', 'ghi', 'temp', 'cloud']
    for df in [train, test]:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"缺失必要字段: {missing}")

    # 列名标准化
    column_mapping = {
        '实际功率': 'power',
        'ghi': 'radiation',
        'temp': 'temperature',
        'cloud': 'cloud_cover'
    }
    train = train.rename(columns=column_mapping)
    test = test.rename(columns=column_mapping)

    # 基于训练集的动态阈值
    rad_low = train['radiation'].quantile(0.05)
    power_q1 = train['power'].quantile(0.01)
    power_q99 = train['power'].quantile(0.99)

    # 应用过滤条件（保持索引对齐）
    train = train[
        (train['radiation'] >= rad_low) &
        (train['power'].between(power_q1, power_q99))
        ].copy()
    test = test[
        (test['radiation'] >= rad_low) &
        (test['power'].between(power_q1, power_q99))
        ].copy()

    # 时间序列插值（带容错机制）
    for df in [train, test]:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.interpolate(method='time', inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    return train, test

def create_features(train, test):
    # 时间周期特征
    for df in [train, test]:
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    lag = 24 * 4  #15分钟间隔
    train['power_lag24h'] = train['power'].shift(lag).ffill()
    test['power_lag24h'] = test['power'].shift(lag).ffill().fillna(train['power'].median())

    #滚动特征
    train['power_roll24h'] = train['power'].rolling(24 * 4, min_periods=1).mean()
    test['power_roll24h'] = test['power'].expanding(min_periods=1).mean()

    return train.dropna(how='any'), test.dropna(how='any')


# 模型训练
def train_model(train, test, features):
    model = lgb.LGBMRegressor(
        num_leaves=63,
        learning_rate=0.02,
        n_estimators=2000,
        min_child_samples=100,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )

    model.fit(
        train[features],
        train['power'],
        eval_set=[(test[features], test['power'])],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    return model

# 稳健评估框架
def safe_evaluate(y_true, y_pred, test_radiation, train_radiation):
    try:
        # 数据对齐
        common_index = y_true.index.intersection(y_pred.index)
        if len(common_index) == 0:
            raise ValueError("预测值与真实值索引不匹配")

        test_rad = test_radiation.reindex(common_index)
        rad_threshold = train_radiation.quantile(0.1)
        valid_mask = (
                y_true[common_index].notna() &
                y_pred[common_index].notna() &
                test_rad.notna()
        )

        # 创建白天时段掩码（正确比较辐射值）
        daytime_mask = test_rad[valid_mask] > rad_threshold

        # 计算指标
        y_true_valid = y_true[common_index][valid_mask][daytime_mask]
        y_pred_valid = y_pred[common_index][valid_mask][daytime_mask]

        if len(y_true_valid) == 0:
            raise ValueError("无有效数据可用于评估")

        return {
            'RMSE': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
            'MAE': mean_absolute_error(y_true_valid, y_pred_valid),
            'R2': max(0, r2_score(y_true_valid, y_pred_valid))
    }

    except Exception as e:
        print(f"评估失败: {str(e)}")
    return {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}


if __name__ == '__main__':
    try:
        # 数据管道
        train, test = load_and_preprocess('train_data.csv', 'test_data.csv')
        train, test = create_features(train, test)

        # 定义特征集
        base_features = ['hour_sin', 'hour_cos', 'dayofyear_sin', 'dayofyear_cos']
        nwp_features = base_features + [
            'radiation', 'temperature', 'cloud_cover',
            'power_lag24h', 'power_roll24h'
        ]

        # 模型训练
        print("=== 训练基线模型 ===")
        baseline_model = train_model(train, test, base_features)
        print("\n=== 训练NWP模型 ===")
        nwp_model = train_model(train, test, nwp_features)

        # 生成预测
        test['baseline_pred'] = baseline_model.predict(test[base_features])
        test['nwp_pred'] = nwp_model.predict(test[nwp_features])

        # 结果评估（传递正确的辐射数据）
        baseline_metrics = safe_evaluate(
            test['power'], test['baseline_pred'],
            test['radiation'],  # 测试集辐射数据
            train['radiation']  # 训练集辐射数据
        )
        nwp_metrics = safe_evaluate(
            test['power'], test['nwp_pred'],
            test['radiation'],
            train['radiation']
        )

        # 结果展示
        print("\n=== 最终指标 ===")
        print(
            f"基线模型: RMSE={baseline_metrics['RMSE']:.3f}, MAE={baseline_metrics['MAE']:.3f}, R²={baseline_metrics['R2']:.3f}")
        print(f"NWP模型: RMSE={nwp_metrics['RMSE']:.3f}, MAE={nwp_metrics['MAE']:.3f}, R²={nwp_metrics['R2']:.3f}")

        # 可视化诊断
        plt.figure(figsize=(15, 6))
        plt.scatter(test['power'], test['nwp_pred'], alpha=0.3, label='预测值')
        plt.plot([0, test['power'].max()], [0, test['power'].max()], 'r--', label='理想线')
        plt.xlabel('实际功率 (kW)')
        plt.ylabel('预测功率 (kW)')
        plt.title('功率预测散点图')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"程序执行失败: {str(e)}")