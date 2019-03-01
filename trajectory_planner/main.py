import numpy as np
import sys
import csv
from time import gmtime, strftime
from matplotlib import pyplot as plt
from trajectory_planner.RRT import Point, find_path
from trajectory_planner.interpolation import interpolate
from trajectory_planner.trajectoryplanner import trajectory, yawtrajectory

#time for stabilization at the end of trajectory
stab_time = 4

size_of_map = Point(20,20)
max_length = 0.5
max_steps = 2000


def getTrajectoryFromRRT(T,ts, init_point, goal_point, save=True, to_plot=True):
    time = np.linspace(0, T, T/ts+1)

    init_p = Point(init_point[0],init_point[1]) #for RRT for x,y
    goal_p = Point(goal_point[0],goal_point[1]) #for RRT for x,y

    tree_x, tree_y = find_path(init_p, goal_p, size_of_map, max_length, max_steps, to_plot=to_plot)
    if tree_x == [] or tree_y == []:
        sys.exit("Error in the RRT")

    z = np.linspace(init_point[2], goal_point[2], num=len(time))
    psi = np.linspace(init_point[3], goal_point[3], num=len(time))

    interp_x = interpolate(time, tree_x, stab_time=stab_time)
    interp_y = interpolate(time, tree_y, stab_time=stab_time)
    interp_z = interpolate(time, z, stab_time=stab_time)
    interp_psi = interpolate(time, psi, stab_time=stab_time)
    
    if save == True : writeToCSV(interp_x, interp_y, interp_z, interp_psi, T, ts)

    x = trajectory(time, interp_x, init_point[0])
    y = trajectory(time, interp_y, init_point[1])
    z = trajectory(time, interp_z, init_point[2])
    yaw = yawtrajectory(time, interp_psi, init_point[3])
    
    if to_plot==True : plotPath(T,ts, x, y, z, yaw)

    p, p_dot, p_2dot, p_3dot ,p_4dot = repack(x,y,z)
    psi, psi_dot, psi_2dot = yaw[0,:], yaw[1,:], yaw[2,:]

    return p, p_dot, p_2dot, p_3dot ,p_4dot, psi, psi_dot, psi_2dot

def getTrajectoryFromFile(T, ts, filename, to_plot=True):
    time = np.linspace(0, T, T/ts+1)

    data = np.loadtxt(filename, delimiter=',')
    x_path = data[:,0]
    y_path = data[:,1]
    z_path = data[:,2]
    yaw_path = data[:,3]

    x = trajectory(time, x_path, x_path[0])
    y = trajectory(time, y_path, y_path[0])
    z = trajectory(time, z_path, z_path[0])
    yaw = yawtrajectory(time, yaw_path, yaw_path[0])
    
    if to_plot==True:
        plt.figure(1)
        plt.title("Loaded path")
        plt.scatter(x_path[0], y_path[0])
        plt.scatter(x_path[-1], x_path[-1], c='r')
        plt.plot(x_path,y_path, c='b', label='desired path')
        plt.show(block=False)

        plotPath(T,ts, x, y, z, yaw)

    p, p_dot, p_2dot, p_3dot ,p_4dot = repack(x,y,z)
    psi, psi_dot, psi_2dot = yaw[0,:], yaw[1,:], yaw[2,:]

    return p, p_dot, p_2dot, p_3dot ,p_4dot, psi, psi_dot, psi_2dot

def getTrajectoryFromStep(T,ts, init_point, to_plot=True):
    time = np.linspace(0, T, T/ts+1)

    # uncomment one of two parts below
    ##########################################################################
    # with trajectory planner, using linear filter
    # x = trajectory(time, np.ones_like(time), init_point[0]) 
    # y = trajectory(time, np.ones_like(time), init_point[1])
    # z = trajectory(time, np.ones_like(time), init_point[2])
    # yaw = yawtrajectory(time, np.ones_like(time), init_point[3])

    # if to_plot==True: plotPath(T,ts, x, y, z, yaw)

    # p, p_dot, p_2dot, p_3dot, p_4dot = repack(x,y,z)
    # psi, psi_dot, psi_2dot = yaw[0,:], yaw[1,:], yaw[2,:]

    ##########################################################################
    # without trajectory planning - just hard step function // minus in z axis becouse it's down directed
    p = np.array([np.ones_like(time),np.ones_like(time),-np.ones_like(time)]).transpose()
    p_dot = np.array([np.zeros_like(time),np.zeros_like(time),np.zeros_like(time)]).transpose()
    p_2dot = np.array([np.zeros_like(time),np.zeros_like(time),np.zeros_like(time)]).transpose()
    p_3dot = np.array([np.zeros_like(time),np.zeros_like(time),np.zeros_like(time)]).transpose()
    p_4dot = np.array([np.zeros_like(time),np.zeros_like(time),np.zeros_like(time)]).transpose()
    psi = np.ones_like(time)
    psi_dot = np.zeros_like(time)
    psi_2dot =np.zeros_like(time)

    return p, p_dot, p_2dot, p_3dot ,p_4dot, psi, psi_dot, psi_2dot

def getTrajectoryFromPoints(T, ts, points, to_plot=True):
    time = np.linspace(0, T, T/ts+1)

    interp_x = interpolate(time, points[0,:], stab_time=stab_time)
    interp_y = interpolate(time, points[1,:], stab_time=stab_time)
    interp_z = interpolate(time, points[2,:], stab_time=stab_time)
    interp_psi = interpolate(time, points[3,:], stab_time=stab_time)

    x = trajectory(time, interp_x, points[0,0])
    y = trajectory(time, interp_y, points[1,0])
    z = trajectory(time, interp_z, points[2,0])
    yaw = yawtrajectory(time, interp_psi, points[3,0])
    
    if to_plot==True : 
        plt.figure(1)
        plt.title("Loaded path")
        plt.scatter(points[0,-1], points[1,-1], c='r')
        plt.scatter(points[0,0], points[1,0])
        plt.plot(points[0],points[1], label='desired path')
        plt.show(block=False)
        plotPath(T,ts, x, y, z, yaw)

    p, p_dot, p_2dot, p_3dot ,p_4dot = repack(x,y,z)
    psi, psi_dot, psi_2dot = yaw[0,:], yaw[1,:], yaw[2,:]

    return p, p_dot, p_2dot, p_3dot ,p_4dot, psi, psi_dot, psi_2dot

def plotPath(T,ts, trajectory_x, trajectory_y, trajectory_z, trajectory_psi):
    plt.ioff()
    plt.figure(1)
    plt.title("Desired trajectory")
    plt.axis([0, size_of_map.x, 0, size_of_map.y])
    plt.plot(trajectory_x[0,:],trajectory_y[0,:], 'm', label="filtered path")
    plt.show(block=False)

def repack(x,y,z):
    p = np.vstack([np.vstack([x[0,:], y[0,:]]), z[0,:]])
    p_dot = np.vstack([np.vstack([x[1,:], y[1,:]]), z[1,:]])
    p_2dot = np.vstack([np.vstack([x[2,:], y[2,:]]), z[2,:]])
    p_3dot = np.vstack([np.vstack([x[3,:], y[3,:]]), z[3,:]])
    p_4dot = np.vstack([np.vstack([x[4,:], y[4,:]]), z[4,:]])

    return np.transpose(p), np.transpose(p_dot), np.transpose(p_2dot), np.transpose(p_3dot), np.transpose(p_4dot)

def writeToCSV(x, y, z, yaw, T, ts):
    filename = strftime("%Y%m%d_%H%M%S", gmtime()) + "_T" + str(T) + "ts" + str(ts).replace('.', '_') + ".csv"
    np.savetxt(filename, np.transpose([x,y,z, yaw]), fmt='%1.8e', delimiter=',')

if __name__ == "__main":
    getTrajectoryFromRRT(10, 0.1, np.array([0,0,0,0]), np.array([15,15,0,0]))
