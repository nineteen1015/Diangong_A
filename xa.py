import xarray as xr
import pandas as pd
import os
from glob import glob


def merge_era5_files(input_dir):

    accum_files = glob(os.path.join(input_dir, "**/*stepType-accum.nc"), recursive=True)
    instant_files = glob(os.path.join(input_dir, "**/*stepType-instant.nc"), recursive=True)

    if not accum_files or not instant_files:
        raise FileNotFoundError("缺少必要文件类型（accum/instant）")

    print("加载accum数据...")
    ds_accum = xr.open_mfdataset(accum_files, combine='nested', concat_dim='time')

    print("加载instant数据...")
    ds_instant = xr.open_mfdataset(instant_files, combine='nested', concat_dim='time')

    # 合并数据集（使用ssrd替代ssr）
    print("合并变量...")
    merged_ds = xr.merge([ds_accum[['ssrd']], ds_instant[['t2m', 'tcc']]])

    return merged_ds

def process_data(ds):

    #北纬30-40，东经110-120
    ds = ds.sel(
        latitude=slice(40, 30),
        longitude=slice(110, 120)
    ).mean(['latitude', 'longitude'])

    ds['ssrd'] = ds['ssrd'] / 3600  # J/m²->W/m²
    ds['t2m'] = ds['t2m'] - 273.15  # K->℃
    ds['tcc'] = ds['tcc'] * 100  # 比->百分比

    # 重命名
    return ds.rename_vars({
        'ssrd': 'solar_rad',
        't2m': 'temp_2m',
        'tcc': 'cloud_cover'
    })


if __name__ == "__main__":
    try:
        era5_dir = r"D:\DiangongB\text\ERA5\1"#1-12
        output_csv = "1.csv"

        merged_ds = merge_era5_files(era5_dir)
        processed_ds = process_data(merged_ds)

        df = processed_ds.to_dataframe().reset_index()

        df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

        df.to_csv(output_csv, index=False)
        print(f"成功保存至：{output_csv}")

    except Exception as e:
        print(f"处理失败：{str(e)}")