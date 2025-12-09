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
        cadence_elem = trackpoint.find('tcx:Extensions/ns3:TPX/ns3:RunCadence', ns)
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
        if cadence_elem is not None and cadence_elem.text is not None:
            cadence = int(cadence_elem.text) * 2
        else:
            cadence = None
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
                'cadence': cadence,
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
            cadence_value = df.iloc[j]['cadence']
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

def parse_tcx_numpy(file):
    ns = {
        'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
        'ns3': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'
    }
    tree = etree.parse(file)
    trackpoints = tree.findall('.//tcx:Trackpoint', ns)
    time_list, dist_list, ele_list = [], [], []
    hr_list, cad_list, lat_list, lon_list = [], [], [], []

    for trackpoint in trackpoints:
        time_elem = trackpoint.find('tcx:Time', ns)
        dist_elem = trackpoint.find('tcx:DistanceMeters', ns)
        ele_elem = trackpoint.find('tcx:AltitudeMeters', ns)

        if time_elem is None or dist_elem is None or ele_elem is None:
            continue
        try:
            time = datetime.fromisoformat(time_elem.text.replace("Z", "+00:00")).timestamp()
        except:
            continue
        time_list.append(time)
        dist_list.append(float(dist_elem.text))
        ele_list.append(float(ele_elem.text))

        hr_elem = trackpoint.find('tcx:HeartRateBpm/tcx:Value', ns)
        hr_list.append(float(hr_elem.text) if hr_elem is not None else np.nan)

        cad_elem = trackpoint.find('tcx:Extensions/ns3:TPX/ns3:RunCadence', ns)
        cad_list.append(float(cad_elem.text) * 2 if cad_elem is not None else np.nan)

        lat_elem = trackpoint.find('tcx:Position/tcx:LatitudeDegrees', ns)
        lon_elem = trackpoint.find('tcx:Position/tcx:LongitudeDegrees', ns)
        lat_list.append(float(lat_elem.text) if lat_elem is not None else np.nan)
        lon_list.append(float(lon_elem.text) if lon_elem is not None else np.nan)

    return (
        np.array(time_list, dtype=np.float64),
        np.array(dist_list, dtype=np.float64),
        np.array(ele_list, dtype=np.float64),
        np.array(hr_list, dtype=np.float64),
        np.array(cad_list, dtype=np.float64),
        np.array(lat_list, dtype=np.float64),
        np.array(lon_list, dtype=np.float64)
    )

def calculate_pace_numpy(time_arr, dist_arr):
    time_diff = np.diff(time_arr)
    dist_diff = np.diff(dist_arr)

    with np.errstate(divide='ignore', invalid='ignore'):
        pace = time_diff * (1000/60) / dist_diff
        pace[(dist_diff <= 0) | (pace > 20)] = np.nan

    pace = np.insert(pace, 0, np.nan)
    return pace

def smoothing_numpy(array, window_size):
    mask = np.isfinite(array)
    arr_filled = np.nan_to_num(array, nan=0.0)
    cumsum = np.convolve(arr_filled, np.ones(window_size), mode='same')
    counts = np.convolve(mask.astype(float), np.ones(window_size), mode='same')
    smoothed = np.divide(cumsum, counts, out=np.full_like(cumsum, np.nan), where=counts!=0)
    smoothed[counts == 0] = np.nan
    return smoothed

def parse_tcx_list(file):
    ns = {
        'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
        'ns3': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'
    }
    tree = etree.parse(file)
    trackpoints = tree.findall('.//tcx:Trackpoint', ns)

    t, d, e, hr, cad, lat, lon = [], [], [], [], [], [], []

    for pt in trackpoints:
        time_elem = pt.find('tcx:Time', ns)
        dist_elem = pt.find('tcx:DistanceMeters', ns)
        ele_elem = pt.find('tcx:AltitudeMeters', ns)

        if time_elem is None or dist_elem is None or ele_elem is None:
            continue

        try:
            ts = datetime.fromisoformat(time_elem.text.replace("Z", "+00:00")).timestamp()
        except:
            continue

        t.append(ts)
        d.append(float(dist_elem.text))
        e.append(float(ele_elem.text))

        hr_elem = pt.find('tcx:HeartRateBpm/tcx:Value', ns)
        cad_elem = pt.find('tcx:Extensions/ns3:TPX/ns3:RunCadence', ns)
        lat_elem = pt.find('tcx:Position/tcx:LatitudeDegrees', ns)
        lon_elem = pt.find('tcx:Position/tcx:LongitudeDegrees', ns)

        hr.append(float(hr_elem.text) if hr_elem is not None else None)
        cad.append(float(cad_elem.text) * 2 if cad_elem is not None else None)
        lat.append(float(lat_elem.text) if lat_elem is not None else None)
        lon.append(float(lon_elem.text) if lon_elem is not None else None)

    return t, d, e, hr, cad, lat, lon

def calculate_pace_list(time, dist):
    pace = [None]
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        dd = dist[i] - dist[i - 1]
        if dd <= 0:
            pace.append(None)
        else:
            p = dt * (1000/60) / dd
            if p > 20:
                pace.append(None)
            else:
                pace.append(p)
    return pace

def smoothing_list(arr, window):
    out = []
    half = window // 2

    for i in range(len(arr)):
        start = max(0, i - half)
        end = min(len(arr), i + half + 1)

        values = [x for x in arr[start:end] if x is not None]
        out.append(sum(values) / len(values) if values else None)

    return out

def comparing(file, smoothing_window_size=5):
    start1 = time.perf_counter()
    gdf = parse_tcx(file)
    gdf = calculate_pace(gdf)
    gdf = smoothing(gdf, smoothing_window_size)
    end1 = time.perf_counter()

    start2 = time.perf_counter()
    t, d, e, hr, cad, lat, lon = parse_tcx_numpy(file)
    pace_np = calculate_pace_numpy(t, d)
    smooth_pace_np = smoothing_numpy(pace_np, smoothing_window_size)
    smooth_cad_np = smoothing_numpy(cad, smoothing_window_size)
    end2 = time.perf_counter()

    start3 = time.perf_counter()
    tl, dl, el, hrl, cadl, latl, lonl = parse_tcx_list(file)
    pace_l = calculate_pace_list(tl, dl)
    smooth_pace_l = smoothing_list(pace_l, smoothing_window_size)
    smooth_pace_l = smoothing_list(cadl, smoothing_window_size)
    end3 = time.perf_counter()

    time1 = end1 - start1
    time2 = end2 - start2
    time3 = end3 - start3

    print(f"Czas obliczeń (oryginalna): {time1:.6f} s")
    print(f"Czas obliczeń (listy): {time3:.6f} s")
    print(f"Czas obliczeń (NumPy): {time2:.6f} s")
    print(f"Różnica w czasie (NumPy vs oryginalna): {time1 / time2:.2f}x szybciej")
    print(f"Różnica w czasie (NumPy vs listy): {time3 / time2:.2f}x szybciej")

if __name__ == "__main__":
    file = ("prawie10.tcx")
    comparing(file)
    #print(gdf)
    #print(array)
