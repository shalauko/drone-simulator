import numpy as np 
from matplotlib import pyplot as plt 
import trajectory_planner.main as tp
import dynamics
from gtcontroller import gt_controller as gt

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
#  parameters of controller
Kp = 1
Kv = 1.4*(4*Kp)**0.5
K_R = 0.6
K_omega = 0.2

######################################################
# initial conditions as firsts values in output arrays
R = np.eye(3)
R_dot = np.zeros(9).reshape(3,3)

omega = np.array([np.zeros(3)])
omega_dot = np.array([np.zeros(3)])

eta = np.zeros(3).reshape(3,1)              # not used in calculations - only for stats // for now is not availibale

p_2dot = np.array([np.zeros(3)])
p_dot = np.array([np.zeros(3)]) 
p = np.array([np.zeros(3)]) 

u = np.array([0])
tau = np.array([0,0,0])

######################################################
# points in the space for path and trajectory planning
init_point = np.array([p[0,0], p[0,1], p[0,2], eta[2]])         # init x, y, z, yaw
goal_point = np.array([10, 10, 1, 0])                             # target x, y, z, yaw

# for generator from points - use it for fourth opsion in trajectory planning
x_path = np.array([init_point[0], 4, 5, 2, 12, 15, 3, 4, goal_point[0]])
y_path = np.array([init_point[1], 2, 10, 14, 8, 7, 4, 3, goal_point[1]])
z_path = np.array([init_point[2], 0, 0, 0, 0, 0, 0, 0, goal_point[2]])
psi_path = np.array([init_point[3], 0, 0, 0, 0, 0, 0, 0, goal_point[3]])
pathPoints = np.vstack((x_path, np.vstack((y_path, np.vstack((z_path, psi_path))))))

########################################################
# begin of calculations

######################## 
print("Trajectory calculations...")
# calculate path and trajectory // uncomment one of four options
# when path is loaded from a file, note that time and time step are showed at the end of file name - T and ts in simulation at the begin of code shoud be same

# p_d, p_d_dot, p_d_2dot, p_d_3dot ,p_d_4dot, psi_d, psi_d_dot, psi_d_2dot = tp.getTrajectoryFromStep(T, ts, init_point, plots=True)
p_d, p_d_dot, p_d_2dot, p_d_3dot ,p_d_4dot, psi_d, psi_d_dot, psi_d_2dot = tp.getTrajectoryFromRRT(T, ts, init_point, goal_point, save=False, plots=True)
# p_d, p_d_dot, p_d_2dot, p_d_3dot ,p_d_4dot, psi_d, psi_d_dot, psi_d_2dot = tp.getTrajectoryFromFile(T, ts, '20190225_185806_T50ts0_01.csv', plots=True)
# p_d, p_d_dot, p_d_2dot, p_d_3dot ,p_d_4dot, psi_d, psi_d_dot, psi_d_2dot = tp.getTrajectoryFromPoints(T, ts, pathPoints, plots=True)

print("Trajectory is found...")
#########################
print("Simulation...")
for i in range(0, int(T/ts)):

    R_now = R[3*i:3*i+3]
    omega_now = omega[i]

    # geometric tracking controller
    u_now, tau_now = gt(p[i], p_dot[i],\
        p_d[i], p_d_dot[i], p_d_2dot[i], p_d_3dot[i], p_d_4dot[i], \
        psi_d[i], psi_d_dot[i], psi_d_2dot[i], \
        omega_now, R_now, m, J, g, Kp, Kv, K_R, K_omega, calculate_omega_d=True)

    # stack thrust and torque
    u = np.append(u,u_now)
    tau = np.vstack([tau,tau_now])

    # dynamics symulation // computating values for the next step // for now eta is not used
    p_2dot_next, eta, R_dot_next, omega_dot_next = dynamics.simulateDynamics(u_now, tau_now, R_now, omega_now, m, J)

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
    omega_dot = np.vstack((omega_dot, omega_dot_next.reshape(1,3).squeeze()))

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
    
# cheking quality
p_error = np.sqrt(np.square(p_d - p))
RMS = np.sqrt(np.sum(np.square(p_d - p))/len(p))
print('RMS =', RMS)

################################################
#  ploting    
plt.figure(1)
plt.title('Path')
plt.plot(p[:,0],p[:,1], '-y', label='real path')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(loc='upper left')

plt.figure(2)
plt.subplot(221)
plt.title('xy-path')
plt.plot(p[:,0],p[:,1], '-y')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.subplot(222)
plt.title('position')
plt.plot(t, p[:,0], 'b', label="x") 
plt.plot(t, p[:,1], 'g', label="y")
plt.plot(t, p[:,2], 'r', label="z")
plt.xlabel('time [s]')
plt.ylabel('[m]')
plt.legend(loc='upper left')
plt.subplot(223)
plt.title('velocity')
plt.plot(t, np.sqrt(np.square(p_dot[:,0]) + np.square(p_dot[:,1]) + np.square(p_dot[:,2])), 'b') 
plt.xlabel('time [s]')
plt.ylabel('[m/s]')
plt.subplot(224)
plt.title('acceleration')
plt.plot(t, np.sqrt(np.square(p_2dot[:,0]) + np.square(p_2dot[:,1]) + np.square(p_2dot[:,2])), 'b') 
plt.xlabel('time [s]')
plt.ylabel('[m/s^2]')
plt.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.96, wspace=0.28, hspace=0.5)

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
plt.title('3D-error')
plt.plot(t, np.sqrt(np.sum(np.square(p_error)/3, axis=1)))
plt.xlabel('time [s]')
plt.ylabel('[m]')
plt.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.96, wspace=0.28, hspace=0.5)

fig = plt.figure(4)
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
plt.plot(t, p_d[:,2], 'g')
plt.plot(t, p[:,2], 'r')
plt.xlabel('time [s]')
plt.ylabel('[m]')
plt.subplot(338)
plt.title('z-velocity')
plt.plot(t, p_d_dot[:,2], 'g')
plt.plot(t, p_dot[:,2], 'r')
plt.xlabel('time [s]')
plt.ylabel('[m/s]')
plt.subplot(339)
plt.title('z-acceleration')
plt.plot(t, p_d_2dot[:,2], 'g')
plt.plot(t, p_2dot[:,2], 'r')
plt.xlabel('time [s]')
plt.ylabel('[m]')
fig.legend(loc='upper left')
plt.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.96, wspace=0.28, hspace=0.5)

# plt.figure(11)
# plt.title('steering torques')
# plt.plot(t, tau[:,0], 'b')
# plt.plot(t, tau[:,1], 'g')
# plt.plot(t, tau[:,2], 'r')

# plt.figure(12)
# plt.title('thrust')
# plt.plot(t, u, 'b')
 
plt.show()
