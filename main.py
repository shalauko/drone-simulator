import numpy as np 
from matplotlib import pyplot as plt 
import trajectory_planner.main as tp
import dynamics

# time step, time of simulation and timevector
ts = 0.1
T = 10
t = np.linspace(0, T, num=int(T/ts)+1)

# parameters of drone
m = 1.2
J = np.diag([0.015, 0.015, 0.026])

# initial conditions
R0 = np.eye(3)
R_dot0 = np.zeros(9).reshape(3,3)

omega0 = np.array([np.zeros(3)])        # that view of array we need for use vstack and iteration
omega_dot0 = np.array([np.zeros(3)])

eta0 = np.zeros(3).reshape(3,1)

p_2dot0 = np.array([np.zeros(3)])
p_dot0 = np.array([np.zeros(3)])        # that view of array we need for use vstack and iteration
p0 = np.array([np.zeros(3)])            # that view of array we need for use vstack and iteration

# construct output values
R = R0
R_dot = R_dot0

omega = omega0
omega_dot = omega_dot0

p_2dot = p_2dot0
p_dot = p_dot0
p = p0

# calculate path and trajectory
tp.gettrajectoryDIV(T,ts)

u = np.linspace(0, 35, num=int(T/ts)+1) # 9.81 * 1.2 * np.ones(int(T/ts)+1)
tau = np.array([[0],[0],[0]])

for i in range(1, int(T/ts+1)):
    # dynamics symulation // _t means temporary
    R_previous = R[3*(i-1):3*(i-1)+3]
    p_2dot_t, eta, R_dot_t, omega_dot_t = dynamics.simulateDynamics(u[i-1], tau, R_previous, omega[i-1].reshape(3,1), m, J)

    ## displacement, velocity, acceleration
    # stack acceleration
    p_2dot = np.vstack((p_2dot, p_2dot_t.reshape(1,3)))

    # integration of acceleration, i.e get velocity; stack velocity
    p_dot_t = p_dot[i-1] + np.trapz(np.array([p_2dot[i-1],p_2dot[i]]), dx=0.1, axis = 0)
    p_dot = np.vstack((p_dot, p_dot_t))

    # integration of velocity, i.e get position; stack position
    p_t = p[i-1] + np.trapz(np.array([p_dot[i-1],p_dot[i]]), dx=0.1, axis = 0)
    p = np.vstack((p, p_t))

    ## omega
    # stack omega_dot
    omega_dot = np.vstack((omega_dot, omega_dot_t.reshape(1,3).squeeze()))

    # integration of omega_dot, i.e. get omega; stack omega
    omega_t = omega[i-1] + np.trapz(np.array([omega_dot[i-1],omega_dot[i]]), dx=0.1, axis = 0)
    omega = np.vstack((omega, omega_t))

    ## R
    # stack R_dot
    R_dot = np.vstack((R_dot, np.array(R_dot_t)))

    #integration of R_dot, i.e. get R; stack R
    R_t = R[3*(i-1):3*(i-1)+3].reshape(9) + np.trapz(np.array([R_dot[3*(i-1):3*(i-1)+3].reshape(9), R_dot[3*i:3*i+3].reshape(9)]), dx=0.1, axis = 0)
    R_t = R_t.reshape(3,3)
    R = np.vstack((R, np.array(R_t)))
    

# print("p \n", p)
# print("p_dot \n", p_dot)
# print("p_2dot \n", p_2dot)
# print(eta)
# print("R_dot \n", R_dot)
# print("R \n", R)
# print("omega_dot \n", omega_dot)
# print("omega \n", omega)
# plt.plot(t, p[:,0]) # position at x axis
# plt.plot(t, p[:,1]) # position at y axis

plt.plot(t, p[:,2]) 
plt.plot(t, p_dot[:,2])
plt.plot(t, p_2dot[:,2])
# plt.show(block=False)

# plt.figure(2)
# plt.plot(t, omega[:,0]) 
# plt.plot(t, omega[:,1]) 
# plt.plot(t, omega[:,2]) 
plt.show()



