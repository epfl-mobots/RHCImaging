'''
A general library used in all imaging scripts.

Author: Cyril Monette
Initial date: 18/07/2025
'''

import pandas as pd
import os, sys
from tqdm import tqdm
from dask import delayed, compute
from HiveOpenings.libOpenings import * # To filter out invalid datetimes

RPiCamV3_img_shape = (2592, 4608)   # Height, Width
RPiCamV3_img_shape_RGB = (2592, 4608, 3)   # Height, Width, Channels

@delayed
def _fetch_single_datetime(dt:pd.Timestamp, paths:list[str], hive_nb:int):
    dt = dt.tz_convert('UTC')  # Ensure the datetime is in UTC. Will fail if not tz-aware.
    dt_result = {}
    for path in paths:
        rpi_name = os.path.basename(path)[:4]
        rpi_num = path.split('/')[-1][3]
        filename = f"hive{hive_nb}_rpi{rpi_num}_{dt.strftime('%y%m%d-%H%M')}"
        files = os.listdir(path)
        img_path = next((os.path.join(path, f) for f in files if filename in f), None)
        dt_result[rpi_name] = img_path
    return dt, dt_result

@delayed
def _fetch_single_datetime_rounded(dt:pd.Timestamp, paths:list[str], hive_nb:int, max_time_diff:int=15):
    '''
    Fetches the images path for a specific datetime, finding the closest images to the given datetime.
    :param dt: pd.Timestamp, datetime for which we want the image. Needs to be tz-aware.
    :param paths: list of str, list of paths to search for the images.
    :param hive_nb: int, hive number (e.g., 1, 2, etc.)
    :param max_time_diff: int, maximum time difference in minutes to consider for rounding.
    :return: dict, containing the image paths for each RPi. If no image is found within the max_time_diff, the value will be None for that RPi.
    '''
    dt = dt.tz_convert('UTC')  # Ensure the datetime is in UTC. Will fail if not tz-aware.
    dt_result = {}
    for path in paths:
        rpi_name = os.path.basename(path)[:4]
        rpi_num = path.split('/')[-1][3]
        prefix = f"hive{hive_nb}_rpi{rpi_num}_"
        files = os.listdir(path)
        best_file = None
        best_delta = None
        for f in files:
            if not f.startswith(prefix):
                continue
            ts_part = f[len(prefix):].split('.')[0].rstrip('Z')
            try:
                file_dt = pd.to_datetime(ts_part, format='%y%m%d-%H%M%S').tz_localize('UTC')
                delta = abs((file_dt - dt).total_seconds())
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_file = f
            except ValueError:
                continue
        if best_delta is not None and best_delta <= max_time_diff * 60:
            dt_result[rpi_name] = os.path.join(path, best_file)
        else:
            dt_result[rpi_name] = None
    return dt, dt_result


def fetchImagesPaths(rootpath_imgs:str, datetimes:list[pd.Timestamp], hive_nb:int, invalid_recovery_time:int = None, images_fill_limit:int = None, rpis:list[int]=[1,2,3,4], exact_image:bool=True, verbose=False):
    '''
    Fetches the images' paths for a specific hive at specific datetimes using Dask for parallel processing.

    :param rootpath_imgs: str, root path to the images
    :param datetimes: list of pd.Timestamps, datetimes for which we want the images. Precision at minute level. Needs to be tz-aware.
    :param hive_nb: int, hive number (e.g., 1, 2, etc.)
    :param invalid_recovery_time: int, if specified, will filter out invalid datetimes including the given recovery time in minutes (when the hives were being opened + recovery time [min]).
    :param images_fill_limit: int, if provided, maximum number of images to fill the gaps with the previous images. If not provided, will not fill gaps (None in df).
    :param rpis: list of int, list of RPi numbers to consider. Default is [1,2,3,4].
    :param exact_image: bool, if True, will look for an exact match of the datetime. If False, will use _fetch_single_datetime_rounded to find the closest image.
    :return imgs_paths_filtered: pd.DataFrame, containing the image paths. Each row is a datetime, each column is a RPi. If validity is checked, the last column will indicate whether the datetime is valid or not (bool).
    '''

    if not all(dt.tzinfo is not None for dt in datetimes):
        raise ValueError("All datetimes must be tz-aware.")

    paths = [os.path.join(rootpath_imgs, f) for f in os.listdir(rootpath_imgs) if os.path.isdir(os.path.join(rootpath_imgs, f))]
    paths = [p for p in paths if f"h{hive_nb}" in p and int(os.path.basename(p)[3]) in rpis]
    paths.sort()
    if verbose:
        print(f"Using image paths: {paths}")

    columns = [os.path.basename(p)[:4] for p in paths]

    if invalid_recovery_time is not None:
        # Filter out datetimes that are not valid (i.e., when the hives were being opened)
        valid_datetimes = filter_timestamps(datetimes, hive_nb, invalid_recovery_time)

    validity = [dt in valid_datetimes for dt in datetimes] if invalid_recovery_time is not None else None

    if verbose:
        print(f"Datetimes: {datetimes}")
        if invalid_recovery_time is not None:
            print(f"Valid datetimes: {valid_datetimes}")

    # Delayed processing
    if exact_image:
        delayed_results = [_fetch_single_datetime(dt, paths, hive_nb) for dt in datetimes]
    else:
        delayed_results = [_fetch_single_datetime_rounded(dt, paths, hive_nb) for dt in datetimes]
    results = compute(*delayed_results)

    # Build final DataFrame
    imgs_paths = pd.DataFrame(index=datetimes, columns=columns)
    
    for dt, dt_result in results:
        for rpi in columns:
            imgs_paths.loc[dt, rpi] = dt_result[rpi]
    
    if verbose:
        print("Non-null counts per column:")
        print(imgs_paths.notna().sum())
        print("Total non-nulls:", imgs_paths.notna().sum().sum())

    if imgs_paths.isna().all().all():
        raise ValueError("No images found for the given datetimes and hive number. " \
                         "There might be a timestamp mismatch.")
    
    if validity is not None:
        imgs_paths['valid'] = validity # Add a column for validity if it is checked

    if images_fill_limit is not None and images_fill_limit > 0:
        valid_imgs = imgs_paths[imgs_paths['valid'] == True].drop(columns=['valid']) if 'valid' in imgs_paths.columns else imgs_paths
        print(f"Missing images before filtering: {valid_imgs.isnull().sum().sum()} out of {valid_imgs.shape[0] * valid_imgs.shape[1]}")

        imgs_paths_filtered = imgs_paths.ffill(limit=images_fill_limit, axis=0) if images_fill_limit > 0 else imgs_paths

        valid_imgs_filtered = imgs_paths_filtered[imgs_paths_filtered['valid'] == True].drop(columns=['valid']) if 'valid' in imgs_paths_filtered.columns else imgs_paths_filtered
        missing_after = valid_imgs_filtered.isnull().sum().sum()
        if missing_after > 0:
            print(f"[W]: There are still {missing_after} missing images after filling with limit {images_fill_limit}.")
        
        return imgs_paths_filtered

    return imgs_paths