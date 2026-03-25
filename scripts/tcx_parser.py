from tcxreader.tcxreader import TCXReader
import numpy as np

def parser_py(file):
    tcx_reader = TCXReader()
    tcx = tcx_reader.read(file)
    points = []
    for trackpoint in tcx.trackpoints:
        lat = trackpoint.latitude if trackpoint.latitude else 0.0
        lon = trackpoint.longitude if trackpoint.longitude else 0.0
        if lat != 0.0 and lon != 0.0:
            ele = trackpoint.elevation if trackpoint.elevation else 0.0
            hr = trackpoint.hr_value if trackpoint.hr_value else 0.0
            cadence = trackpoint.cadence if trackpoint.cadence else 0.0
            points.append((lat, lon, ele, hr, cadence))
    return points

def parser_np(file):
    points = parser_py(file)
    return np.array(points)