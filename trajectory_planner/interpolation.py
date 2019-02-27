import numpy as np 
import matplotlib.pyplot as plt
def interpolate(time, points, stab_time=0):
    offset = 0
    if stab_time != 0:
        ts = time[1] - time[0]
        offset = int(stab_time/ts)
        
    t = np.linspace(0, time[-1-offset], len(points))
    points_interp = np.interp(time , t, points)

    return points_interp