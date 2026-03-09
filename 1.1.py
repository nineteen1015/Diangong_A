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
    # 光伏系统参数北纬30-40，东经110-120
    system_params = {
        'latitude': 35.0,
        'longitude': 115.0,
        'altitude': 50,
        'capacity': 10,  # MW
        'surface_tilt': 30,  # 倾角
        'surface_azimuth': 180,  # 方位角
        'loss_factor': 0.85,  # 系统效率
        'min_solar_zenith': 85  # 太阳高度角阈值
    }

    # 创建位置对象
    location = Location(
        latitude=system_params['latitude'],
        longitude=system_params['longitude'],
        tz='Asia/Shanghai',
        altitude=system_params['altitude']
    )

    # 计算太阳位置
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
            poa['poa_global']
            * system_params['capacity']
            * system_params['loss_factor']/ 1000  # W/m² -> kW/m²
    )

    # 添加太阳高度角信息
    df['solar_zenith'] = solar_position['apparent_zenith']

    return df, system_params

# 数据预处理
def preprocessing(filepath):
    # 加载数据
    df = pd.read_excel(filepath, sheet_name='HourlyData')

    # 时间处理
    df['LOCAL_HOUR_END'] = pd.to_numeric(df['LOCAL_HOUR_END'], errors='coerce')
    df = df.dropna(subset=['LOCAL_HOUR_END'])
    df['LOCAL_HOUR_END'] = df['LOCAL_HOUR_END'].clip(1, 24).astype(int)

    df['time'] = pd.to_datetime(df['local_day']) + pd.to_timedelta(
        df['LOCAL_HOUR_END'] - 1,
        unit='h'
    )
    df = df.set_index('time').sort_index()

    # 处理实际功率
    df['tot_solar_mwh'] = (
        pd.to_numeric(df['tot_solar_mwh'], errors='coerce')
        .fillna(0)
        .clip(lower=0)
    )

    return df

def enhanced_analysis(df, system_params):
    # 过滤有效数据（白天时段）
    daytime_mask = df['solar_zenith'] < system_params['min_solar_zenith']
    valid_df = df[daytime_mask].copy()

    # 计算偏差率（添加极小值防止除零）
    valid_df['deviation_ratio'] = (
            valid_df['tot_solar_mwh']
            / valid_df['theory_power'].clip(lower=0.001)
    )

    # 过滤异常偏差率
    valid_df = valid_df[
        (valid_df['deviation_ratio'] >= 0)
        & (valid_df['deviation_ratio'] <= 5)
        ]

    # 生成分析报告
    print("增强型分析报告".center(40, '='))
    print(f"总数据量: {len(df)}")
    print(f"有效白天数据量: {len(valid_df)}")
    print(f"理论功率范围: {valid_df['theory_power'].min():.2f} - {valid_df['theory_power'].max():.2f} MW")
    print(f"实际功率范围: {valid_df['tot_solar_mwh'].min():.2f} - {valid_df['tot_solar_mwh'].max():.2f} MWh")

    # 关键统计指标
    stats = valid_df['deviation_ratio'].describe()
    print("\n偏差率统计:")
    print(f"平均偏差率: {stats['mean']:.2%}")
    print(f"标准差: {stats['std']:.3f}")

    # 可视化分析
    fig, ax = plt.subplots(2, 2, figsize=(18, 12))

    # 功率对比曲线
    valid_df['theory_power'].resample('D').mean().plot(
        ax=ax[0, 0], label='理论功率', alpha=0.7
    )
    valid_df['tot_solar_mwh'].resample('D').mean().plot(
        ax=ax[0, 0], label='实际功率', alpha=0.7
    )
    ax[0, 0].set_title('日均功率对比')
    ax[0, 0].set_ylabel('功率 (MW)')
    ax[0, 0].legend()

    # 偏差率分布
    valid_df['deviation_ratio'].hist(bins=50, ax=ax[0, 1], alpha=0.7)
    ax[0, 1].set_title('偏差率分布')
    ax[0, 1].set_xlabel('实际/理论功率比')

    # 小时级模式
    valid_df.groupby(valid_df.index.hour)['deviation_ratio'].mean().plot(
        ax=ax[1, 0], marker='o'
    )
    ax[1, 0].set_title('小时平均偏差率')
    ax[1, 0].set_ylabel('偏差率')

    # 散点图分析
    ax[1, 1].scatter(
        valid_df['theory_power'],
        valid_df['tot_solar_mwh'],
        alpha=0.3,
        s=10
    )
    ax[1, 1].plot(
        [0, valid_df['theory_power'].max()],
        [0, valid_df['theory_power'].max()],
        'r--'
    )
    ax[1, 1].set_title('实际 vs 理论功率')
    ax[1, 1].set_xlabel('理论功率 (MW)')
    ax[1, 1].set_ylabel('实际功率 (MWh)')

    plt.tight_layout()
    plt.savefig('analysis.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    # 数据预处理
    df = preprocessing('hourly_solar_gen_2023.xlsx')

    # 理论功率计算
    df, params = calculate_theoretical_power(df)

    # 执行增强分析
    enhanced_analysis(df, params)

    # 保存处理数据
    df.to_csv('data.csv')
    print("\n分析完成")
