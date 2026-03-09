import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_gefcom():
    try:
        #数据加载
        gefcom = pd.read_excel(
            'hourly_solar_gen_2023.xlsx',
            sheet_name='HourlyData',
            usecols=['local_day', 'LOCAL_HOUR_END', 'tot_solar_mwh'],
            engine='openpyxl'
        ).rename(columns=lambda x: x.strip())

        # 时间处理
        gefcom = gefcom.dropna(subset=['LOCAL_HOUR_END', 'tot_solar_mwh'])
        gefcom['LOCAL_HOUR_END'] = pd.to_numeric(gefcom['LOCAL_HOUR_END'], errors='coerce')
        gefcom = gefcom.dropna(subset=['LOCAL_HOUR_END'])

        #时间索引
        gefcom['timestamp'] = (
                pd.to_datetime(gefcom['local_day'].astype(str).str.split().str[0],
                               format='%Y-%m-%d',
                               errors='coerce')
                + pd.to_timedelta(
            gefcom['LOCAL_HOUR_END'].replace(24, 0).astype(int),
            unit='h'
        )
        )

        #时间序列框架
        full_range = pd.date_range(
            start=gefcom['timestamp'].min().floor('D'),
            end=gefcom['timestamp'].max().ceil('D'),
            freq='15min'
        )

        return (
            gefcom.set_index('timestamp')['tot_solar_mwh']
            .reindex(full_range)
            .interpolate('time', limit_area='inside')
            .fillna(0)
            .mul(4)
            .clip(0, 2000)
            .rename('实际功率')
            .to_frame()
        )

    except Exception as e:
        print(f"加载发电数据失败：{str(e)}")
        sys.exit(1)


def load_era5():
    base_path = r'./ERA5_CSV'
    dfs = []

    for root, _, files in os.walk(base_path):
        for file in [f for f in files if f.endswith('.csv')]:
            try:
                df = pd.read_csv(
                    os.path.join(root, file),
                    parse_dates=['valid_time'],
                    usecols=['valid_time', 'solar_rad', 'temp_2m', 'cloud_cover'],
                    dtype={'solar_rad': float, 'temp_2m': float, 'cloud_cover': float}
                ).rename(columns={
                    'solar_rad': 'ghi',
                    'temp_2m': 'temp',
                    'cloud_cover': 'cloud'
                }).set_index('valid_time')

                # 数据清洗增强
                df['ghi'] = df['ghi'].clip(0) / 3600
                df['cloud'] = df['cloud'].clip(0, 100) / 100
                df['temp'] = df['temp'].clip(-40, 60)

                dfs.append(df)
            except Exception as e:
                print(f"处理文件 {file} 出错: {str(e)[:100]}")
                continue

    if not dfs:
        print("错误：未找到有效气象数据！")
        sys.exit(1)

    # 时间处理增强
    full_df = (
        pd.concat(dfs)
        .pipe(lambda df: df[~df.index.duplicated(keep='first')])
        .asfreq('h')
        .resample('15min')
        .interpolate('time')
        .bfill(limit=4)
        .dropna()
    )

    # 确保完整时间范围
    full_range = pd.date_range(
        start=full_df.index.min().floor('D'),
        end=full_df.index.max().ceil('D'),
        freq='15min'
    )

    return full_df.reindex(full_range).interpolate('time')


def validate_data(df):

    time_diffs = df.index.to_series().diff().dropna()
    time_tolerance = pd.Timedelta('1s')
    valid_time = ((time_diffs - pd.Timedelta('15min')).abs() <= time_tolerance).all()

    if not valid_time:
        anomalies = time_diffs[abs(time_diffs - pd.Timedelta('15min')) > time_tolerance]
        print(f"发现{len(anomalies)}处异常时间间隔：")
        print(anomalies.head(5))
        raise ValueError("时间连续性验证失败")

    # 数据范围检查
    checks = {
        '辐射非负': (df['ghi'] >= 0).all(),
        '温度范围': df['temp'].between(-40, 60).all(),
        '云量范围': df['cloud'].between(0, 1).all(),
        '功率范围': df['实际功率'].between(0, 2000).all()
    }

    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"数据验证失败: {', '.join(failed)}")

    print("数据验证通过")


def create_features(df):

    df = df.asfreq('15min')

    # 滞后特征（精确计算15分钟间隔）
    for periods in [96, 192]:  # 24h*4=96, 48h*4=192
        df[f'lag_{periods}'] = df['实际功率'].shift(periods)

    # 增强时间特征
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)

    # 天气特征工程
    for var in ['ghi', 'temp']:
        for window in [24, 96, 672]:  # 6h, 24h, 7d
            df[f'{var}_rolling_{window}_mean'] = (
                df[var]
                .rolling(window=window, min_periods=1)
                .mean()
            )

    return df.dropna()


if __name__ == "__main__":
    try:
        # 数据加载
        print("加载发电数据...")
        power = load_gefcom()
        print(f"发电数据范围：{power.index.min()} - {power.index.max()}")

        print("\n加载气象数据...")
        weather = load_era5()
        print(f"气象数据范围：{weather.index.min()} - {weather.index.max()}")

        merged = (
            pd.merge_asof(
                power.sort_index(),
                weather.sort_index(),
                left_index=True,
                right_index=True,
                tolerance=pd.Timedelta('15min'),
                direction='nearest'
            )
            .pipe(lambda df: df[~df.index.duplicated()])
            .resample('15min')
            .asfreq()
            .interpolate(method='time')
            .bfill(limit=4)
        )

        # 数据验证
        validate_data(merged)

        # 特征工程
        print("\n创建特征...")
        features = create_features(merged)

        # 数据拆分
        split_date = '2023-09-01'
        train = features.loc[:split_date].copy()
        test = features.loc[split_date:].copy()

        # 安全标准化
        feature_cols = [c for c in features.columns if c not in ['True', 'lag_96', 'lag_192']]
        scaler = StandardScaler()
        train.loc[:, feature_cols] = scaler.fit_transform(train[feature_cols])
        test.loc[:, feature_cols] = scaler.transform(test[feature_cols])

        # 保存结果
        train.to_csv('train_data.csv', index=True)
        test.to_csv('test_data.csv', index=True)

        print("\n预处理完成！")
        print(f"训练集样本：{len(train)} | 测试集样本：{len(test)}")

    except Exception as e:
        print(f"\n运行失败：{str(e)}")
        sys.exit(1)