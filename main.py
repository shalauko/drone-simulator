import sys
import numpy as np 
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import trajectory_planner.main as tp
import dynamics
from gtcontroller import gt_controller as gt
from nhcontroller import nh_controller as nh
def main():
    #############################################
    # which controller do you want to use (GT = 1 or NH = 2); set what do you want
    controller = 1
    if controller == 1: print('GEOMETRIC TRACKING CONTROLLER')
    elif controller == 2: print('NEAR-HOVERING CONTROLLER')
    #############################################
    # Which GT_CTRL do you want to use. For full calculate_omega_d=True, for simplified =False
    calculate_omega_d = False

    #############################################
    # time step, time of simulation and timevector
    ts = 0.01
    T = 50
    t = np.linspace(0, T, num=int(T/ts)+1)

    ################################################
    # parameters of drone and gravity
    m = 1.2
    J = np.diag([0.015, 0.015, 0.026])
    g = 9.81

    ##########################################
    #  parameters of GT-controller
    if controller == 1:
        Kp = 3
        Kv = 1.4*(4*Kp)**0.5
        K_R = 0.6
        K_omega = 0.2

    ###############################################
    # parameters of NH-controller
    elif controller == 2:
        kp=4                                                            # P for PID // thrust ctrl
        kd=1.4*np.sqrt(4*kp)                                               # D for PID // thrust ctrl
        ki=1                                                             # I for PID (in fact is used only for z axis) // thrust ctrl
        k_psi = kp                                                         # P in thrust controll
        K_eta_dot = np.array([20, 50, 10]).reshape(3,1)                                  # D for RPY in torque controll 
        K_eta = np.array([100, 150, 0]).reshape(3,1)                                              # P for PRY in torque controll //for yaw must be 0!

        # bounding
        max_pos_err_norm = 10
        max_tilt    = 45/180 * np.pi

        # initial value for position error and position error integral
        e_p = np.array([np.zeros(3)])
        e_I = np.array([np.zeros(3)])

    ######################################################
    # initial conditions as firsts values in output arrays; wired view of arrays demand stacking :)
    R = np.eye(3)
    R_dot = np.zeros(9).reshape(3,3)

    omega = np.array([np.zeros(3)])
    omega_dot = np.array([np.zeros(3)])

    eta = np.array([np.zeros(3)])

    p_2dot = np.array([np.zeros(3)])
    p_dot = np.array([np.zeros(3)])
    p = np.array([np.zeros(3)])

    u = np.array([np.zeros(1)])
    tau = np.array([np.zeros(3)])

    ######################################################
    # points in the space for path and trajectory planning
    init_point = np.array([p[0,0], p[0,1], p[0,2], eta[0,2]])                       # init x, y, z, yaw
    goal_point = np.array([15, 15, -1, 1])                                           #  target x, y, z(directed down), yaw

    # for generator from points - use it for fourth opsion in trajectory planning //
    # init and goal point are requared, rest of values could be pasted randomly //
    # quantity of points in each path should be equal
    x_path = np.array([init_point[0], 3, 6, 15, 3, 7, goal_point[0]])
    y_path = np.array([init_point[1], 6, 2, 10, 12 , 1.5, goal_point[1]])
    z_path = np.array([init_point[2], -2, -3, 0, 1, -3, goal_point[2]])                       # z axis is directed down
    psi_path = np.array([init_point[3], 0.5, 1, 0.4, 0, 0, goal_point[3]])
    pathPoints = np.vstack((x_path, np.vstack((y_path, np.vstack((z_path, psi_path))))))

    #############################################################
    # begin of calculations

    ############################################################ 
    print("Trajectory calculations...")
    # calculate path and trajectory // uncomment one of four options
    # when path is loaded from a file, note that time and time step are showed at the end of file name - T and ts in simulation at the begin of code shoud be same

    # p_d, p_d_dot, p_d_2dot, p_d_3dot ,p_d_4dot, psi_d, psi_d_dot, psi_d_2dot = tp.getTrajectoryFromStep(T, ts, init_point, to_plot=True)
    p_d, p_d_dot, p_d_2dot, p_d_3dot ,p_d_4dot, psi_d, psi_d_dot, psi_d_2dot = tp.getTrajectoryFromRRT(T, ts, init_point, goal_point, save=False, to_plot=True)
    # p_d, p_d_dot, p_d_2dot, p_d_3dot ,p_d_4dot, psi_d, psi_d_dot, psi_d_2dot = tp.getTrajectoryFromFile(T, ts, 'trajectories/20190227_212533_T50ts0_01.csv', to_plot=True)
    # p_d, p_d_dot, p_d_2dot, p_d_3dot ,p_d_4dot, psi_d, psi_d_dot, psi_d_2dot = tp.getTrajectoryFromPoints(T, ts, pathPoints, to_plot=True)

    print("Trajectory is found")
    ##############################################################
    print("Start a simulation")
    for i in range(0, int(T/ts)):

        # helped parameters
        R_now = R[3*i:3*i+3]
        omega_now = omega[i]

        if controller == 1:
            # geometric tracking controller // if full GT-CTRL calculate_omega_d = True, if simlified GT-CTRL calculate_omega_d = False
            u_now, tau_now = gt(p[i], p_dot[i],\
                p_d[i], p_d_dot[i], p_d_2dot[i], p_d_3dot[i], p_d_4dot[i], \
                psi_d[i], psi_d_dot[i], psi_d_2dot[i], \
                omega_now, R_now, m, J, g, Kp, Kv, K_R, K_omega, calculate_omega_d=calculate_omega_d)
        
        elif controller == 2:
            # near-hovering conntroller and integration of position error
            u_now, tau_now, e_p_now = nh(p[i], p_dot[i],\
                p_d[i], p_d_dot[i], p_d_2dot[i], psi_d[i],\
                    eta[i], omega[i], e_I[i], \
                        m, J, g, kd, kp, ki, k_psi, K_eta, K_eta_dot,\
                            max_pos_err_norm, max_tilt)
            
            # stack position error; integrate position error; stack integral 
            e_p = np.vstack([e_p,e_p_now.reshape(1,3)])
            e_I_next = e_I[i] + np.trapz(np.array([e_p[i],e_p[i+1]]), dx=ts, axis = 0)
            e_I = np.vstack((e_I, e_I_next))

        # stack thrust and torque
        u = np.append(u,u_now)
        tau = np.vstack([tau,tau_now])

        # dynamics symulation // computating values for the next step
        p_2dot_next, eta_next, R_dot_next, omega_dot_next = dynamics.simulateDynamics(u_now, tau_now, R_now, omega_now, m, J)

        ## stack Euler angles
        eta = np.vstack((eta, eta_next.reshape(1,3)))

        ## displacement, velocity, acceleration
        # stack acceleration
        p_2dot = np.vstack((p_2dot, p_2dot_next.reshape(1,3)))

        # integration of acceleration, i.e get velocity; stack velocity
        p_dot_next = p_dot[i] + np.trapz(np.array([p_2dot[i],p_2dot[i+1]]), dx=ts, axis = 0)
        p_dot = np.vstack((p_dot, p_dot_next))

        # integration of velocity, i.e get position; stack position
        p_next = p[i] + np.trapz(np.array([p_dot[i],p_dot[i+1]]), dx=ts, axis = 0)
        p = np.vstack((p, p_next))

        ## omega
        # stack omega_dot
        omega_dot = np.vstack((omega_dot, omega_dot_next))
        # integration of omega_dot, i.e. get omega; stack omega
        omega_next = omega[i] + np.trapz(np.array([omega_dot[i],omega_dot[i+1]]), dx=ts, axis = 0)
        omega = np.vstack((omega, omega_next))

        ## R - rotation matrixs
        # stack R_dot
        R_dot = np.vstack((R_dot, np.array(R_dot_next)))

        #integration of R_dot, i.e. get R; stack R
        R_next = R[3*i:3*i+3].reshape(9) + np.trapz(np.array([R_dot[3*i:3*i+3].reshape(9), R_dot[3*(i+1):3*(i+1)+3].reshape(9)]), dx=ts, axis = 0)
        R_next = R_next.reshape(3,3)
        R = np.vstack((R, np.array(R_next)))

        # show progress in terminal
        five_percent = 0.05*int(T/ts)
        if i%five_percent == 0:
            title = str(i*5/five_percent)  + "% done"
            print(title, end="\r")

    print("Simulation completed")

    ################################################
    # square errors
    p_error = np.sqrt(np.square(p_d - p))
    psi_error = np.sqrt(np.square(psi_d - eta[:,2]))

    # cheking quality
    std_dev = np.sqrt(np.sum(np.square(p_d - p))/len(p))
    print('std_dev =', std_dev)
    print('median velocity =', np.median(np.sqrt(np.square(p_dot[:,0]) + np.square(p_dot[:,1]) + np.square(p_dot[:,2]))))

    ################################################
    ## ploting // note, axes in world and  body frames are directed in same direction (NED), 
    # but for esthetic reason at plots are inverted
    plotPath(t, p)                                                          # plot actual path (it shows above desired path, 
                                                                            # when ploting in grajectory generator in swiched on)
    plotPosition(t, p, p_dot, p_2dot)                                       # plot actual path in 3D; separated displacement in x,y,z; 3D velocity; 3D acceleration
    plot3DErrorAndYaw(t, p_error, psi_error)                                # plot 3D and yaw errors
    plotDesiredVsActual(t, p, p_d, eta[:,2],  psi_d)                        # plot position and yaw, desired vs actual

    ##  some additional plots
    # plotErrors(t, p_error, psi_error)                                       # plot errors on separated axes
    # plotTrajectories(t, p_d, p_d_dot, p_d_2dot, p_d_3dot, p_d_4dot, psi_d, psi_d_dot, psi_d_2dot)
    # plotPosVelAcc(t, p, p_dot, p_2dot, p_d, p_d_dot, p_d_2dot)              # plot displacement, velocity, aceleration at x,y,z; actual values vs desired
    # plotTorques(t,tau)
    # plotThrust(t,u)
    # plotRollAndPith(t,eta)
    plt.show()

#######################################################
# plot functions
def plotPath(t, p):
    plt.figure(1)
    plt.title('Path')
    plt.plot(p[:,0],p[:,1], '-y', label='real path')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(loc='upper left')


def plotPosition(t, p, p_dot, p_2dot):
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(221, projection='3d')
    ax.plot3D( p[:,1], p[:,0], -p[:,2], 'b')              # order is mixed becoue of axes inversion NED->classic world frame (y_wf = x_ned, x_wf = y_ned, z_wf = -z_ned)
    plt.title('3D-path')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plt.subplot(222)
    plt.title('position')
    plt.plot(t, p[:,0], 'b', label="x") 
    plt.plot(t, p[:,1], 'g', label="y")
    plt.plot(t, -p[:,2], 'r', label="z")
    plt.xlabel('time [s]')
    plt.ylabel('[m]')
    plt.legend(loc='upper left')
    plt.subplot(223)
    plt.title('3D velocity')
    plt.plot(t, np.sqrt(np.square(p_dot[:,0]) + np.square(p_dot[:,1]) + np.square(p_dot[:,2])), 'b') 
    plt.xlabel('time [s]')
    plt.ylabel('[m/s]')
    plt.subplot(224)
    plt.title('3D acceleration')
    plt.plot(t, np.sqrt(np.square(p_2dot[:,0]) + np.square(p_2dot[:,1]) + np.square(p_2dot[:,2])), 'b') 
    plt.xlabel('time [s]')
    plt.ylabel('[m/s^2]')
    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.96, wspace=0.28, hspace=0.5)

def plotErrors(t, p_error, psi_error):
    plt.figure(3)
    plt.subplot(221)
    plt.title('x-error')
    plt.plot(t, p_error[:,0])
    plt.xlabel('time [s]')
    plt.ylabel('[m]')
    plt.subplot(222)
    plt.title('y-error')
    plt.plot(t, p_error[:,1])
    plt.xlabel('time [s]')
    plt.ylabel('[m]')
    plt.subplot(223)
    plt.title('z-error')
    plt.plot(t, p_error[:,2])
    plt.xlabel('time [s]')
    plt.ylabel('[m]')
    plt.subplot(224)
    plt.title('3D-error and yaw error')
    plt.plot(t, np.sqrt(np.sum(np.square(p_error), axis=1)), 'b', label="position error [m]")
    plt.plot(t, psi_error * 180/np.pi, 'r', label="yaw error [deg]")
    plt.xlabel('time [s]')
    plt.ylabel('[m] or [deg]')
    plt.legend(loc='upper right')
    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.96, wspace=0.28, hspace=0.5)

def plot3DErrorAndYaw(t, p_error, psi_error):
    plt.figure(4)
    plt.title('3D-error and yaw error')
    plt.plot(t, np.sqrt(np.sum(np.square(p_error), axis=1)), 'b', label="position error [m]")
    plt.plot(t, psi_error * 180/np.pi, 'r', label="yaw error [deg]")
    plt.xlabel('time [s]')
    plt.ylabel('[m] or [deg]')
    plt.legend(loc='upper right')

def plotDesiredVsActual(t, p, p_d, psi, psi_d):
    fig5 = plt.figure(5)
    plt.subplot(221)
    plt.title('x-position')
    plt.plot(t, p_d[:,0], 'g', label="desired value")
    plt.plot(t, p[:,0], 'r', label="real value")
    plt.xlabel('time [s]')
    plt.ylabel('[m]')
    plt.subplot(222)
    plt.title('y-position')
    plt.plot(t, p_d[:,1], 'g')
    plt.plot(t, p[:,1], 'r')
    plt.xlabel('time [s]')
    plt.ylabel('[m]')
    plt.subplot(223)
    plt.title('z-position')
    plt.plot(t, -p_d[:,2], 'g')
    plt.plot(t, -p[:,2], 'r')
    plt.xlabel('time [s]')
    plt.ylabel('[m]')
    plt.subplot(224)
    plt.title('yaw')
    plt.plot(t, psi_d * 180/np.pi, 'g')
    plt.plot(t, psi * 180/np.pi, 'r')
    plt.xlabel('time [s]')
    plt.ylabel('[deg]')

    fig5.legend(loc='upper left')
    plt.subplots_adjust(wspace=0.28, hspace=0.5)

def plotTrajectories(t, p_d, p_d_dot, p_d_2dot, p_d_3dot, p_d_4dot, psi_d, psi_d_dot, psi_d_2dot):
    fig = plt.figure(100)
    plt.subplot(221)
    plt.title("desired trajectory for x")
    plt.plot(t, p_d[:,0], 'b', label='position')
    plt.plot(t, p_d_dot[:,0], 'g', label='velocity')
    plt.plot(t, p_d_2dot[:,0], 'r', label='acceleration')
    plt.plot(t, p_d_3dot[:,0], 'm', label='jerk')
    plt.plot(t, p_d_4dot[:,0], 'c', label='snap')
    plt.subplot(222)
    plt.title("desired trajectory for y")
    plt.plot(t, p_d[:,1], 'b')
    plt.plot(t, p_d_dot[:,1], 'g')
    plt.plot(t, p_d_2dot[:,1], 'r')
    plt.plot(t, p_d_3dot[:,1], 'm')
    plt.plot(t, p_d_4dot[:,1], 'c')
    plt.subplot(223)
    plt.title("desired trajectory for z")
    plt.plot(t, -p_d[:,2], 'b')
    plt.plot(t, -p_d_dot[:,2], 'g')
    plt.plot(t, -p_d_2dot[:,2], 'r')
    plt.plot(t, -p_d_3dot[:,2], 'm')
    plt.plot(t, -p_d_4dot[:,2], 'c')
    plt.subplot(224)
    plt.title("desired trajectory for yaw")
    plt.plot(t, psi_d, 'b')
    plt.plot(t, psi_d_dot, 'g')
    plt.plot(t, psi_d_2dot, 'r')
    fig.legend(loc='upper left')
    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.96, wspace=0.28, hspace=0.5)

def plotPosVelAcc(t, p, p_dot, p_2dot, p_d, p_d_dot, p_d_2dot):
    fig = plt.figure(10)
    plt.subplot(331)
    plt.title('x-position')
    plt.plot(t, p_d[:,0], 'g', label="desired value")
    plt.plot(t, p[:,0], 'r', label="real value")
    plt.xlabel('time [s]')
    plt.ylabel('[m]')
    plt.subplot(332)
    plt.title('x-velocity')
    plt.plot(t, p_d_dot[:,0], 'g')
    plt.plot(t, p_dot[:,0], 'r')
    plt.xlabel('time [s]')
    plt.ylabel('[m/s]')
    plt.subplot(333)
    plt.title('x-acceleration')
    plt.plot(t, p_d_2dot[:,1], 'g')
    plt.plot(t, p_2dot[:,1], 'r')
    plt.xlabel('time [s]')
    plt.ylabel('[m/s^2]')
    plt.subplot(334)
    plt.title('y-position')
    plt.plot(t, p_d[:,1], 'g')
    plt.plot(t, p[:,1], 'r')
    plt.xlabel('time [s]')
    plt.ylabel('[m]')
    plt.subplot(335)
    plt.title('y-velocity')
    plt.plot(t, p_d_dot[:,1], 'g')
    plt.plot(t, p_dot[:,1], 'r')
    plt.xlabel('time [s]')
    plt.ylabel('[m/s]')
    plt.subplot(336)
    plt.title('y-acceleration')
    plt.plot(t, p_d_2dot[:,1], 'g')
    plt.plot(t, p_2dot[:,1], 'r')
    plt.xlabel('time [s]')
    plt.ylabel('[m/s^2]')
    plt.subplot(337)
    plt.title('z-position')
    plt.plot(t, -p_d[:,2], 'g')
    plt.plot(t, -p[:,2], 'r')
    plt.xlabel('time [s]')
    plt.ylabel('[m]')
    plt.subplot(338)
    plt.title('z-velocity')
    plt.plot(t, -p_d_dot[:,2], 'g')
    plt.plot(t, -p_dot[:,2], 'r')
    plt.xlabel('time [s]')
    plt.ylabel('[m/s]')
    plt.subplot(339)
    plt.title('z-acceleration')
    plt.plot(t, -p_d_2dot[:,2], 'g')
    plt.plot(t, -p_2dot[:,2], 'r')
    plt.xlabel('time [s]')
    plt.ylabel('[m]')
    fig.legend(loc='upper left')
    plt.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.96, wspace=0.28, hspace=0.5)

def plotTorques(t,tau):
    fig =  plt.figure(11)
    plt.title('steering torques')
    plt.plot(t, tau[:,0], 'b', label = 'x')
    plt.plot(t, tau[:,1], 'g', label = 'y')
    plt.plot(t, tau[:,2], 'r', label = 'z')
    plt.xlabel('time [s]')
    plt.ylabel('[Nm]')
    fig.legend(loc='upper left')

def plotThrust(t,u):
    plt.figure(12)
    plt.title('thrust')
    plt.plot(t, u, 'b')
    plt.xlabel('time [s]')
    plt.ylabel('[N]')

def plotRollAndPith(t,eta):
    plt.figure(13)
    plt.title('Roll and pitch')
    plt.plot(t, eta[:,0] * 180/np.pi, 'b', label = 'roll')
    plt.plot(t, eta[:,1] * 180/np.pi, 'g', label = 'pith')
    plt.xlabel('time [s]')
    plt.ylabel('[deg]')

if __name__ == "__main__":
    main()