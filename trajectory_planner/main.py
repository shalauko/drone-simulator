import numpy as np
import sys
from matplotlib import pyplot as plt
from trajectory_planner.RRT import Point, find_path
from trajectory_planner.interpolation import interpolate
from trajectory_planner.trajectoryplanner import trajectory 

init_p = Point(0,0)
goal_p = Point(6,6)
init_p_z = 0
goal_p_z = 1
size_of_map = Point(10,10)
max_length = 0.5
max_steps = 2000

def gettrajectoryXYZ(T, ts):

    return do_RRT(T,ts)
    ## in case of step; make sure that init_p == (0,0) and init_p_z == 0
    # return do_step(T,ts)

def gettrajectoryDIV(T, ts):

    x,y,z = do_RRT(T,ts)
    ## in case of step; make sure that init_p == (0,0) and init_p_z == 0
    # x,y,z = do_step(T,ts)

    return repack(x,y,z)

def do_RRT(T,ts):
    time = np.linspace(0, T, T/ts+1)

    tree_x, tree_y = find_path(init_p, goal_p, size_of_map, max_length, max_steps)
    if tree_x == [] or tree_y == []:
        sys.exit("Error in the RRT")

    z = np.linspace(init_p_z,goal_p_z, num=len(time))

    interp_x = interpolate(time, tree_x, stab_time=2)
    interp_y = interpolate(time, tree_y, stab_time=2)
    interp_z = interpolate(time, z, stab_time=2)
    

    trajectory_x = trajectory(time, interp_x, init_p.x)
    trajectory_y = trajectory(time, interp_y, init_p.y)
    trajectory_z = trajectory(time, interp_z, init_p_z)

    plottrajectories(T,ts, trajectory_x, trajectory_y, trajectory_z)

    repack(trajectory_x, trajectory_y, trajectory_z)

    return trajectory_x, trajectory_y, trajectory_z

def do_step(T,ts):
    time = np.linspace(0, T, T/ts+1)

    trajectory_x = trajectory(time, np.ones_like(time), init_p.x) 
    trajectory_y = trajectory(time, np.ones_like(time), init_p.y)
    trajectory_z = trajectory(time, np.ones_like(time), init_p_z)

    plottrajectories(T,ts, trajectory_x, trajectory_y, trajectory_z)

    repack(trajectory_x, trajectory_y, trajectory_z)

    return trajectory_x, trajectory_y, trajectory_z

def plottrajectories(T,ts, trajectory_x, trajectory_y, trajectory_z):
    time = np.linspace(0, T, T/ts+1)

    plt.ioff()
    plt.figure(1)
    plt.axis([0, size_of_map.x, 0, size_of_map.y])
    plt.plot(trajectory_x[0,:],trajectory_y[0,:], 'm')
    plt.show(block=False)

    plt.figure(2)
    plt.plot(time, trajectory_x[0,:])
    plt.plot(time, trajectory_x[1,:])
    plt.plot(time, trajectory_x[2,:])
    plt.plot(time, trajectory_x[3,:])
    plt.plot(time, trajectory_x[4,:])
    plt.show(block=False)

    plt.figure(3)
    plt.plot(time, trajectory_y[0,:])
    plt.plot(time, trajectory_y[1,:])
    plt.plot(time, trajectory_y[2,:])
    plt.plot(time, trajectory_y[3,:])
    plt.plot(time, trajectory_y[4,:])
    plt.show(block=False)

    plt.figure(4)
    plt.plot(time, trajectory_z[0,:])
    plt.plot(time, trajectory_z[1,:])
    plt.plot(time, trajectory_z[2,:])
    plt.plot(time, trajectory_z[3,:])
    plt.plot(time, trajectory_z[4,:])
    plt.show()

def repack(x,y,z):
    p = np.vstack([np.vstack([x[0,:], y[0,:]]), z[0,:]])
    p_dot = np.vstack([np.vstack([x[1,:], y[1,:]]), z[1,:]])
    p_2dot = np.vstack([np.vstack([x[2,:], y[2,:]]), z[2,:]])
    p_3dot = np.vstack([np.vstack([x[3,:], y[3,:]]), z[3,:]])
    p_4dot = np.vstack([np.vstack([x[4,:], y[4,:]]), z[4,:]])

    return p, p_dot, p_2dot, p_3dot, p_4dot
    

if __name__ == "__main":
    gettrajectoryXYZ(10, 0.1)
