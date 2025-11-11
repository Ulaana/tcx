import time
from datetime import datetime
import numpy as np
import pandas as pd
from lxml import etree
import geopandas as gpd
from shapely.geometry import Point

pd.set_option('display.max_columns', None)

def parse_tcx(file):
    ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
          'ns3': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'}
    tree = etree.parse(file)
    trackpoints = tree.findall('.//tcx:Trackpoint', ns)
    data = []
    for trackpoint in trackpoints:
        time_elem = trackpoint.find('tcx:Time', ns)
        hr_elem = trackpoint.find('tcx:HeartRateBpm/tcx:Value', ns)
        dist_elem = trackpoint.find('tcx:DistanceMeters', ns)
        ele_elem = trackpoint.find('tcx:AltitudeMeters', ns)
        lat_elem = trackpoint.find('tcx:Position/tcx:LatitudeDegrees', ns)
        lon_elem = trackpoint.find('tcx:Position/tcx:LongitudeDegrees', ns)
        run_cadence_elem = trackpoint.find('tcx:Extensions/ns3:TPX/ns3:RunCadence', ns)
        if time_elem is not None:
            time = datetime.fromisoformat(time_elem.text.replace("Z", "+00:00"))
        else:
            time = None
        if hr_elem is not None and hr_elem.text is not None:
            hr = int(hr_elem.text)
        else:
            hr = None
        if dist_elem is not None and dist_elem.text is not None:
            dist = float(dist_elem.text)
        else:
            dist = None
        if ele_elem is not None and ele_elem.text is not None:
            elevation = float(ele_elem.text)
        else:
            elevation = None
        if lat_elem is not None and lat_elem.text is not None:
            lat = float(lat_elem.text)
        else:
            lat = None
        if lon_elem is not None and lon_elem.text is not None:
            lon = float(lon_elem.text)
        else:
            lon = None
        if run_cadence_elem is not None and run_cadence_elem.text is not None:
            run_cadence = int(run_cadence_elem.text)
        else:
            run_cadence = None
        if run_cadence is not None:
            total_cadence = run_cadence * 2
        else:
            total_cadence = None
        if lon is not None and lat is not None:
            geometry = Point(lon, lat)
        else:
            geometry = None
        if time and dist is not None:
            data.append({
                'time': time,
                'heart_rate': hr,
                'distance': dist,
                'elevation': elevation,
                "lat": lat,
                "lon": lon,
                'run_cadence': run_cadence,
                'total_cadence': total_cadence,
                'geometry': geometry
            })
    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    gdf = gdf.dropna(subset=['time', 'distance', 'elevation'])
    gdf = gdf.sort_values(by='time').reset_index(drop=True)
    return gdf

def calculate_pace(df):
    df['time_diff'] = df['time'].diff().dt.total_seconds()
    df['dist_diff'] = df['distance'].diff()
    pace_data = []
    for time, distance in zip(df['time_diff'], df['dist_diff']):
        if time and distance and distance > 0:
            pace = (time / 60) / (distance / 1000)
            if pace < 20:
                pace_data.append(pace)
            else:
                pace_data.append(None)
        else:
            pace_data.append(None)
    df['pace'] = pace_data
    return df

def smoothing(df, smoothing_window_size):
    pace_smoothed = []
    cadence_smoothed = []
    for i in range(len(df)):
        start = max(0, i - smoothing_window_size // 2)
        end = min(len(df), i + smoothing_window_size // 2 + 1)

        pace_sum = 0
        pace_count = 0
        for j in range(start, end):
            pace_value = df.iloc[j]['pace']
            if pace_value is not None and not np.isnan(pace_value):
                pace_sum += pace_value
                pace_count += 1
        if pace_count > 0:
            pace_avg = pace_sum / pace_count
        else:
            pace_avg = None
        pace_smoothed.append(pace_avg)

        cadence_sum = 0
        cadence_count = 0
        for j in range(start, end):
            cadence_value = df.iloc[j]['total_cadence']
            if cadence_value is not None and not np.isnan(cadence_value):
                cadence_sum += cadence_value
                cadence_count += 1
        if cadence_count > 0:
            cadence_avg = cadence_sum / cadence_count
        else:
            cadence_avg = None
        cadence_smoothed.append(cadence_avg)
    df['pace_smoothed'] = pace_smoothed
    df['cadence_smoothed'] = cadence_smoothed
    return df

def calculate_pace_numpy(df):
    time_diff = df['time'].diff().dt.total_seconds().to_numpy()
    dist_diff = df['distance'].diff().to_numpy()

    with np.errstate(divide='ignore', invalid='ignore'):
        pace = (time_diff / 60) / (dist_diff / 1000)
        pace[(dist_diff <= 0) | (pace > 20)] = np.nan

    df['pace_np'] = pace
    return df

def smoothing_numpy(df, smoothing_window_size):
    pace = df['pace_np'].to_numpy()
    cadence = df['total_cadence'].to_numpy()

    def moving_window(array, window):
        mask = np.isfinite(array)
        array_filled = np.nan_to_num(array, nan=0.0)
        cumsum = np.convolve(array_filled, np.ones(window), mode='same')
        counts = np.convolve(mask.astype(int), np.ones(window), mode='same')
        avg = cumsum / counts
        avg[counts == 0] = np.nan
        return avg

    pace_smoothed = moving_window(pace, smoothing_window_size)
    cadence_smoothed = moving_window(cadence, smoothing_window_size)

    df['pace_smoothed_np'] = pace_smoothed
    df['cadence_smoothed_np'] = cadence_smoothed
    return df

def comparing(df, smoothing_window_size=5):
    df_copy1 = df.copy()
    df_copy2 = df.copy()

    start1 = time.perf_counter()
    df_copy1 = calculate_pace(df_copy1)
    df_copy1 = smoothing(df_copy1, smoothing_window_size)
    end1 = time.perf_counter()

    start2 = time.perf_counter()
    df_copy2 = calculate_pace_numpy(df_copy2)
    df_copy2 = smoothing_numpy(df_copy2, smoothing_window_size)
    end2 = time.perf_counter()

    print(f"Czas obliczeń (oryginał): {end1 - start1:.6f} s")
    print(f"Czas obliczeń (numpy): {end2 - start2:.6f} s")

    return df_copy1, df_copy2

if __name__ == "__main__":
    gdf = parse_tcx("prawie10.tcx")
    df1, df2 = comparing(gdf)
    #print(df1)
    #print(df2)
