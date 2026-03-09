import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pvlib.location import Location
import warnings

# 中文字体显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

def calculate_theoretical_power(df):
    # 光伏系统参数（北纬35.0°，东经115.0°）
    system_params = {
        'latitude': 35.0,
        'longitude': 115.0,
        'altitude': 50,
        'capacity': 10,  # MW
        'surface_tilt': 30,
        'surface_azimuth': 180,
        'loss_factor': 0.85,
        'min_solar_zenith': 85
    }

    # 创建位置对象
    location = Location(
        latitude=system_params['latitude'],
        longitude=system_params['longitude'],
        tz='Asia/Shanghai',
        altitude=system_params['altitude']
    )

    # 计算太阳位置与晴空辐照
    solar_position = location.get_solarposition(df.index)
    clearsky = location.get_clearsky(df.index)

    # 计算平面阵列辐照度（POA）
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=system_params['surface_tilt'],
        surface_azimuth=system_params['surface_azimuth'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi']
    )

    # 计算理论功率（MW）
    df['theory_power'] = (
        poa['poa_global'] * system_params['capacity'] * system_params['loss_factor'] / 1000
    )

    # 添加太阳高度角信息
    df['solar_zenith'] = solar_position['apparent_zenith']
    return df, system_params

def preprocessing(filepath):
    # 校验必要列
    required_cols = ['local_day', 'LOCAL_HOUR_END', 'tot_solar_mwh']
    df = pd.read_excel(filepath, sheet_name='HourlyData')
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"缺失必要列: {missing}")

    # 时间处理
    df['LOCAL_HOUR_END'] = pd.to_numeric(df['LOCAL_HOUR_END'], errors='coerce')
    df = df.dropna(subset=['LOCAL_HOUR_END'])
    df['LOCAL_HOUR_END'] = df['LOCAL_HOUR_END'].clip(1, 24).astype(int)

    # 生成时间索引
    df['time'] = pd.to_datetime(df['local_day']) + pd.to_timedelta(df['LOCAL_HOUR_END'] - 1, unit='h')
    df = df.set_index('time').sort_index()

    # 处理实际功率（单位转换为MW）
    df['tot_solar_mw'] = pd.to_numeric(df['tot_solar_mwh'], errors='coerce').fillna(0).clip(lower=0) / 1e3
    return df

def enhanced_analysis(df, system_params):
    # 过滤白天数据
    daytime_mask = df['solar_zenith'] < system_params['min_solar_zenith']
    valid_df = df[daytime_mask].copy()

    # 计算偏差率
    valid_df['deviation_ratio'] = valid_df['tot_solar_mw'] / valid_df['theory_power'].clip(lower=0.001)
    valid_df = valid_df[(valid_df['deviation_ratio'] >= 0) & (valid_df['deviation_ratio'] <= 5)]

    # 输出统计报告
    print("=" * 40)
    print(f"总数据量: {len(df)}")
    print(f"有效白天数据量: {len(valid_df)}")
    print(f"理论功率范围: {valid_df['theory_power'].min():.2f} - {valid_df['theory_power'].max():.2f} MW")
    print(f"实际功率范围: {valid_df['tot_solar_mw'].min():.2f} - {valid_df['tot_solar_mw'].max():.2f} MW")
    stats = valid_df['deviation_ratio'].describe()
    print("\n偏差率统计:")
    print(f"平均偏差率: {stats['mean']:.2%}")
    print(f"标准差: {stats['std']:.3f}")

    # 可视化分析
    fig, ax = plt.subplots(3, 2, figsize=(20, 18))

    # 日均功率对比
    valid_df['theory_power'].resample('D').mean().plot(ax=ax[0, 0], label='理论功率', alpha=0.7)
    valid_df['tot_solar_mw'].resample('D').mean().plot(ax=ax[0, 0], label='实际功率', alpha=0.7)
    ax[0, 0].set_title('日均功率对比曲线', fontsize=12)
    ax[0, 0].set_ylabel('功率 (MW)')

    # 月均功率对比（季节性分析）
    monthly_theory = valid_df['theory_power'].resample('M').mean()
    monthly_actual = valid_df['tot_solar_mw'].resample('M').mean()
    monthly_theory.plot(ax=ax[0, 1], marker='o', label='理论功率')
    monthly_actual.plot(ax=ax[0, 1], marker='s', label='实际功率')
    ax[0, 1].set_title('月均功率对比（季节性变化）', fontsize=12)
    ax[0, 1].set_ylabel('功率 (MW)')

    # 偏差率分布
    valid_df['deviation_ratio'].hist(bins=50, ax=ax[1, 0], alpha=0.7)
    ax[1, 0].set_title('偏差率分布直方图', fontsize=12)
    ax[1, 0].set_xlabel('实际功率/理论功率')

    # 小时级偏差率
    valid_df.groupby(valid_df.index.hour)['deviation_ratio'].mean().plot(ax=ax[1, 1], marker='o')
    ax[1, 1].set_title('小时平均偏差率变化', fontsize=12)
    ax[1, 1].set_ylabel('偏差率')

    # 实际vs理论散点图
    ax[2, 0].scatter(valid_df['theory_power'], valid_df['tot_solar_mw'], alpha=0.3, s=10)
    ax[2, 0].plot([0, valid_df['theory_power'].max()], [0, valid_df['theory_power'].max()], 'r--')
    ax[2, 0].set_title('实际功率 vs 理论功率散点图', fontsize=12)
    ax[2, 0].set_xlabel('理论功率 (MW)')
    ax[2, 0].set_ylabel('实际功率 (MW)')

    # 功率偏差时间序列
    valid_df['power_diff'] = valid_df['tot_solar_mw'] - valid_df['theory_power']
    valid_df['power_diff'].resample('D').mean().plot(ax=ax[2, 1], color='purple')
    ax[2, 1].set_title('日均功率偏差时序图', fontsize=12)
    ax[2, 1].set_ylabel('偏差 (MW)')

    plt.tight_layout()
    plt.savefig('光伏发电特性分析.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    df = preprocessing('hourly_solar_gen_2023.xlsx')
    df, params = calculate_theoretical_power(df)
    enhanced_analysis(df, params)
    print("\n分析完成")