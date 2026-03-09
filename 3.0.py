import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings

# 中文字体显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
# ------------------------------
# 数据加载与预处理
# ------------------------------
def load_dataset(train_path, test_path):
    """加载并验证数据集"""

    def _load_data(filepath):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna(axis=0, how='all')
        return df

    train = _load_data(train_path)
    test = _load_data(test_path)
    return train, test


def preprocess_data(train, test):
    """联合数据预处理"""
    # 列名标准化
    column_mapping = {
        '实际功率': 'power',
        'ghi': 'radiation',
        'temp': 'temperature',
        'cloud': 'cloud_cover'
    }
    train = train.rename(columns=column_mapping)
    test = test.rename(columns=column_mapping)

    # 合并数据集进行清洗
    full = pd.concat([train.assign(is_train=1), test.assign(is_train=0)])

    # 异常值处理
    full = full[
        (full['radiation'] >= 0) &
        (full['power'].between(0, 1000))
        ]

    # 缺失值处理
    full = full.interpolate(method='time').ffill().bfill()

    # 重新分割数据集
    train = full[full['is_train'] == 1].drop(columns='is_train')
    test = full[full['is_train'] == 0].drop(columns='is_train')
    return train, test


# ------------------------------
# 特征工程
# ------------------------------
def create_features(df):
    """生成时序特征"""
    # 基础时间特征
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month

    # 滞后特征（24小时前）
    df['power_lag24h'] = df['power'].shift(24 * 4)  # 15分钟间隔数据
    df['radiation_lag24h'] = df['radiation'].shift(24 * 4)

    # 滚动特征（24小时均值）
    df['power_roll24h'] = df['power'].rolling(24 * 4, min_periods=1).mean()
    return df.dropna()


# ------------------------------
# 场景划分
# ------------------------------
class SceneProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.scene_labels = {
            0: '晴天',
            1: '多云',
            2: '阴天'
        }

    def fit_transform(self, df):
        scaled = self.scaler.fit_transform(df[['radiation', 'cloud_cover']])
        df['scene'] = self.kmeans.fit_predict(scaled)
        df['scene_label'] = df['scene'].map(self.scene_labels)
        return df

    def transform(self, df):
        scaled = self.scaler.transform(df[['radiation', 'cloud_cover']])
        df['scene'] = self.kmeans.predict(scaled)
        df['scene_label'] = df['scene'].map(self.scene_labels)
        return df


# ------------------------------
# 模型训练评估
# ------------------------------
def train_evaluate_model(train, test, features, target='power'):
    model = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=500,
        random_state=42
    )
    model.fit(train[features], train[target])

    test = test.copy()
    test['pred'] = model.predict(test[features])

    # 白天时段筛选
    radiation_threshold = train['radiation'].quantile(0.1)
    is_daytime = test['radiation'] > radiation_threshold

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(
            test.loc[is_daytime, target],
            test.loc[is_daytime, 'pred']
        )),
        'MAE': mean_absolute_error(
            test.loc[is_daytime, target],
            test.loc[is_daytime, 'pred']
        ),
        'R2': r2_score(
            test.loc[is_daytime, target],
            test.loc[is_daytime, 'pred']
        )
    }
    return metrics, test


# ------------------------------
# 主程序
# ------------------------------
if __name__ == '__main__':
    # 数据管道
    train, test = load_dataset('train_data.csv', 'test_data.csv')
    train, test = preprocess_data(train, test)

    # 特征工程
    train = create_features(train)
    test = create_features(test)

    # 定义特征集
    baseline_features = ['hour', 'weekday', 'month', 'power_lag24h']
    nwp_features = baseline_features + [
        'radiation', 'temperature', 'cloud_cover',
        'radiation_lag24h', 'power_roll24h'
    ]

    # 场景划分
    scene_processor = SceneProcessor()
    train = scene_processor.fit_transform(train)
    test = scene_processor.transform(test)

    # 模型评估
    print("\n=== 基线模型评估 ===")
    baseline_metrics, test = train_evaluate_model(train, test, baseline_features)
    test['pred_baseline'] = test['pred']

    print("\n=== NWP模型评估 ===")
    nwp_metrics, test = train_evaluate_model(train, test, nwp_features)
    test['pred_nwp'] = test['pred']

    # 场景指标分析
    scene_metrics = {}
    for scene in test['scene_label'].unique():
        scene_data = test[test['scene_label'] == scene]
        is_daytime = scene_data['radiation'] > train['radiation'].quantile(0.1)

        scene_metrics[scene] = {
            'RMSE': np.sqrt(mean_squared_error(
                scene_data.loc[is_daytime, 'power'],
                scene_data.loc[is_daytime, 'pred_nwp']
            )),
            'MAE': mean_absolute_error(
                scene_data.loc[is_daytime, 'power'],
                scene_data.loc[is_daytime, 'pred_nwp']
            ),
            'R2': r2_score(
                scene_data.loc[is_daytime, 'power'],
                scene_data.loc[is_daytime, 'pred_nwp']
            )
        }

    # 结果展示
    print("\n=== 最终指标 ===")
    print(f"基线模型: {baseline_metrics}")
    print(f"NWP模型: {nwp_metrics}")

    print("\n=== 分场景指标 ===")
    for scene, metrics in scene_metrics.items():
        print(f"\n{scene}:")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"R²: {metrics['R2']:.2f}")