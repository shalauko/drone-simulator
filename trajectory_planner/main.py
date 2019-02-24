import numpy as np
import sys
import csv
from time import gmtime, strftime
from matplotlib import pyplot as plt
from trajectory_planner.RRT import Point, find_path
from trajectory_planner.interpolation import interpolate
from trajectory_planner.trajectoryplanner import trajectory, yawtrajectory

#time for stabilization at the end of trajectory
stab_time = 3

init_p = Point(0,0)
goal_p = Point(15,15)
init_p_z = 0
goal_p_z = 1
size_of_map = Point(20,20)
max_length = 0.5
max_steps = 2000

init_psi = 0
goal_psi = 0

def getTrajectoryFromRRT(T,ts, savePath=True, plotTrajectories=True):
    time = np.linspace(0, T, T/ts+1)

    tree_x, tree_y = find_path(init_p, goal_p, size_of_map, max_length, max_steps)
    if tree_x == [] or tree_y == []:
        sys.exit("Error in the RRT")

    z = np.linspace(init_p_z,goal_p_z, num=len(time))
    psi = np.linspace(init_psi,goal_psi, num=len(time))

    interp_x = interpolate(time, tree_x, stab_time=stab_time)
    interp_y = interpolate(time, tree_y, stab_time=stab_time)
    interp_z = interpolate(time, z, stab_time=stab_time)
    interp_psi = interpolate(time, psi, stab_time=stab_time)
    
    if savePath == True : writeToCSV(interp_x, interp_y, interp_z, interp_psi)

    x = trajectory(time, interp_x, init_p.x)
    y = trajectory(time, interp_y, init_p.y)
    z = trajectory(time, interp_z, init_p_z)
    yaw = yawtrajectory(time, interp_psi, init_psi)
    
    if plotTrajectories==True : plottrajectories(T,ts, x, y, z, yaw)

    p, p_dot, p_2dot, p_3dot ,p_4dot = repack(x,y,z)
    psi, psi_dot, psi_2dot = yaw[0,:], yaw[1,:], yaw[2,:]

    return p, p_dot, p_2dot, p_3dot ,p_4dot, psi, psi_dot, psi_2dot

def getTrajectoryFromFile(T, ts, filename, plot=True):
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
    
    if plot==True:
        plt.figure(1)
        plt.title("Loaded path")
        plt.scatter(x_path[0], y_path[0])
        plt.scatter(x_path[-1], x_path[-1], c='r')
        plt.plot(x_path,y_path)
        plt.show(block=False)

        plottrajectories(T,ts, x, y, z, yaw)

    p, p_dot, p_2dot, p_3dot ,p_4dot = repack(x,y,z)
    psi, psi_dot, psi_2dot = yaw[0,:], yaw[1,:], yaw[2,:]

    return p, p_dot, p_2dot, p_3dot ,p_4dot, psi, psi_dot, psi_2dot

def getTrajectoryFromStep(T,ts):
    time = np.linspace(0, T, T/ts+1)

    x = trajectory(time, np.ones_like(time), init_p.x) 
    y = trajectory(time, np.ones_like(time), init_p.y)
    z = trajectory(time, np.ones_like(time), init_p_z)
    yaw = yawtrajectory(time, np.ones_like(time), init_psi)

    plottrajectories(T,ts, x, y, z, yaw)

    p, p_dot, p_2dot, p_3dot ,p_4dot = repack(x,y,z)
    psi, psi_dot, psi_2dot = yaw[0,:], yaw[1,:], yaw[2,:]

    return p, p_dot, p_2dot, p_3dot ,p_4dot, psi, psi_dot, psi_2dot

def plottrajectories(T,ts, trajectory_x, trajectory_y, trajectory_z, trajectory_psi):
    time = np.linspace(0, T, T/ts+1)

    plt.ioff()
    plt.figure(1)
    plt.title("Rapidly-exploring random tree with desired trajectory")
    plt.axis([0, size_of_map.x, 0, size_of_map.y])
    plt.plot(trajectory_x[0,:],trajectory_y[0,:], 'm')
    plt.show(block=False)

    plt.figure(101)
    plt.title("desired trajectory for x")
    plt.plot(time, trajectory_x[0,:])
    plt.plot(time, trajectory_x[1,:])
    plt.plot(time, trajectory_x[2,:])
    plt.plot(time, trajectory_x[3,:])
    plt.plot(time, trajectory_x[4,:])
    plt.show(block=False)

    plt.figure(102)
    plt.title("desired trajectory for y")
    plt.plot(time, trajectory_y[0,:])
    plt.plot(time, trajectory_y[1,:])
    plt.plot(time, trajectory_y[2,:])
    plt.plot(time, trajectory_y[3,:])
    plt.plot(time, trajectory_y[4,:])
    plt.show(block=False)

    plt.figure(104)
    plt.title("desired trajectory for z")
    plt.plot(time, trajectory_z[0,:])
    plt.plot(time, trajectory_z[1,:])
    plt.plot(time, trajectory_z[2,:])
    plt.plot(time, trajectory_z[3,:])
    plt.plot(time, trajectory_z[4,:])
    plt.show(block=False)

    plt.figure(105)
    plt.title("desired trajectory for yaw")
    plt.plot(time, trajectory_psi[0,:])
    plt.plot(time, trajectory_psi[1,:])
    plt.plot(time, trajectory_psi[2,:])
    plt.show(block=False)

def repack(x,y,z):
    p = np.vstack([np.vstack([x[0,:], y[0,:]]), z[0,:]])
    p_dot = np.vstack([np.vstack([x[1,:], y[1,:]]), z[1,:]])
    p_2dot = np.vstack([np.vstack([x[2,:], y[2,:]]), z[2,:]])
    p_3dot = np.vstack([np.vstack([x[3,:], y[3,:]]), z[3,:]])
    p_4dot = np.vstack([np.vstack([x[4,:], y[4,:]]), z[4,:]])

    return np.transpose(p), np.transpose(p_dot), np.transpose(p_2dot), np.transpose(p_3dot), np.transpose(p_4dot)

def writeToCSV(x, y, z, yaw):
    filename = strftime("%Y-%m-%d_%H:%M:%S.csv", gmtime())
    np.savetxt(filename, np.transpose([x,y,z, yaw]), fmt='%1.8e', delimiter=',')

if __name__ == "__main":
    getTrajectoryFromRRT(10, 0.1)
