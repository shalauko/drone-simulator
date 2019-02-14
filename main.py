import numpy as np 
from matplotlib import pyplot as plt 
import trajectory_planner.main as tp
import dynamics
from gtcontroller import gt_controller as gt

# time step, time of simulation and timevector
ts = 0.1
T = 10
t = np.linspace(0, T, num=int(T/ts)+1)

# parameters of drone and gravity
m = 1.2
J = np.diag([0.015, 0.015, 0.026])
g = 9.81

# parameters of controller
Kp = 0.5
Kv = 0.8
K_R = 1
K_omega = 1

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
p_d, p_d_dot, p_d_2dot, p_d_3dot ,p_d_4dot, psi_d, psi_d_dot, psi_d_2dot = tp.gettrajectoryDIV(T,ts)

u = np.linspace(0, 35, num=int(T/ts)+1) # 9.81 * 1.2 * np.ones(int(T/ts)+1)
tau = np.array([0,0,0])

for i in range(0, int(T/ts)):
    # dynamics symulation // _t means temporary // for now eta is not used
    # print('p', p[i-1], 'p_dot', p_dot[i-1])
    # print('p_d', p_d[i-1], 'p_d_dot', p_d_dot[i-1], 'p_d_2dot', p_d_2dot[i-1], 'p_d_3dot', p_d_3dot[i-1], 'p_d_4dot', p_d_4dot[i-1])
    R_now = R[3*i:3*i+3]

    u_now, tau_now = gt(p[i], p_dot[i], p_d[i], \
        p_d_dot[i], p_d_2dot[i], p_d_3dot[i], p_d_4dot[i], \
        psi_d[i], psi_d_dot[i], psi_d_2dot[i], \
        omega[i], R_now, m, J, g, Kp, Kv, K_R, K_omega)

    u = np.append(u,u_now)
    tau = np.vstack([tau,tau_now])

    # dynamics symulation // _t means temporary // for now eta is not used
    p_2dot_t, eta, R_dot_t, omega_dot_t = dynamics.simulateDynamics(u_now, tau_now, R_now, omega[i], m, J)

    ## displacement, velocity, acceleration
    # stack acceleration
    p_2dot = np.vstack((p_2dot, p_2dot_t.reshape(1,3)))

    # integration of acceleration, i.e get velocity; stack velocity
    p_dot_t = p_dot[i] + np.trapz(np.array([p_2dot[i],p_2dot[i+1]]), dx=0.1, axis = 0)
    p_dot = np.vstack((p_dot, p_dot_t))

    # integration of velocity, i.e get position; stack position
    p_t = p[i] + np.trapz(np.array([p_dot[i],p_dot[i+1]]), dx=0.1, axis = 0)
    p = np.vstack((p, p_t))

    ## omega
    # stack omega_dot
    omega_dot = np.vstack((omega_dot, omega_dot_t.reshape(1,3).squeeze()))

    # integration of omega_dot, i.e. get omega; stack omega
    omega_t = omega[i] + np.trapz(np.array([omega_dot[i],omega_dot[i+1]]), dx=0.1, axis = 0)
    omega = np.vstack((omega, omega_t))

    ## R
    # stack R_dot
    R_dot = np.vstack((R_dot, np.array(R_dot_t)))

    #integration of R_dot, i.e. get R; stack R
    R_t = R[3*i:3*i+3].reshape(9) + np.trapz(np.array([R_dot[3*i:3*i+3].reshape(9), R_dot[3*(i+1):3*(i+1)+3].reshape(9)]), dx=0.1, axis = 0)
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

plt.figure(5)
plt.plot(t, p[:,2]) 
plt.plot(t, p_dot[:,2])
plt.plot(t, p_2dot[:,2])
# plt.show(block=False)

# plt.plot(t, omega[:,0]) 
# plt.plot(t, omega[:,1]) 
# plt.plot(t, omega[:,2]) 
plt.show()
