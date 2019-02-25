import numpy as np 
from matplotlib import pyplot as plt 
import trajectory_planner.main as tp
import dynamics
from gtcontroller import gt_controller as gt

# time step, time of simulation and timevector
ts = 0.01
T = 50
t = np.linspace(0, T, num=int(T/ts)+1)

# parameters of drone and gravity
m = 1.2
J = np.diag([0.015, 0.015, 0.026])
g = 9.81

# parameters of controller
Kp = 1
Kv = 1.4*(4*Kp)**0.5
K_R = 0.6
K_omega = 0.2

# initial conditions as firsts values in output arrays
R = np.eye(3)
R_dot = np.zeros(9).reshape(3,3)

omega = np.array([np.zeros(3)])
omega_dot = np.array([np.zeros(3)])

eta = np.zeros(3).reshape(3,1)              # not used in calculations - only for stats

p_2dot = np.array([np.zeros(3)])
p_dot = np.array([np.zeros(3)]) 
p = np.array([np.zeros(3)]) 

u = np.array([0])
tau = np.array([0,0,0])

print("Trajectory calculations...")
# calculate path and trajectory // uncomment one of three options
# p_d, p_d_dot, p_d_2dot, p_d_3dot ,p_d_4dot, psi_d, psi_d_dot, psi_d_2dot = tp.getTrajectoryFromStep(T, ts)
p_d, p_d_dot, p_d_2dot, p_d_3dot ,p_d_4dot, psi_d, psi_d_dot, psi_d_2dot = tp.getTrajectoryFromRRT(T, ts, save=True, plots=True)
    # as arguments are time of simulation, time step, name of file and optional boolean ploting
# p_d, p_d_dot, p_d_2dot, p_d_3dot ,p_d_4dot, psi_d, psi_d_dot, psi_d_2dot = tp.getTrajectoryFromFile(100, 0.01, '20190225_121043.csv', plots=False)

print("Trajectory is found...")

print("Simulation...")
for i in range(0, int(T/ts)):

    R_now = R[3*i:3*i+3]
    omega_now = omega[i]

    # geometric tracking controller
    u_now, tau_now = gt(p[i], p_dot[i], p_d[i], \
        p_d_dot[i], p_d_2dot[i], p_d_3dot[i], p_d_4dot[i], \
        psi_d[i], psi_d_dot[i], psi_d_2dot[i], \
        omega_now, R_now, m, J, g, Kp, Kv, K_R, K_omega)

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

    ## R
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

## ploting    
# plt.figure(1)
# plt.title('Executed trajectory')
# plt.plot(p[:,0],p[:,1], '-y')

plt.figure(5)
plt.title('position')
plt.plot(t, p[:,0], 'b') 
plt.plot(t, p[:,1], 'g')
plt.plot(t, p[:,2], 'r')

plt.figure(6)
plt.title('velocity')
plt.plot(t, np.sqrt(np.square(p_dot[:,0]) + np.square(p_dot[:,1]) + np.square(p_dot[:,2])), 'b') 

plt.figure(7)
plt.title('acceleration')
plt.plot(t, np.sqrt(np.square(p_2dot[:,0]) + np.square(p_2dot[:,1]) + np.square(p_2dot[:,2])), 'b') 

# plt.figure(7)
# plt.title('angle velocity')
# plt.plot(t, omega[:,0],'b') 
# plt.plot(t, omega[:,1],'g')
# plt.plot(t, omega[:,2],'r')

# plt.figure(8)
# plt.title('steering torque')
# plt.plot(t, tau[:,0], 'b')
# plt.plot(t, tau[:,1], 'g')
# plt.plot(t, tau[:,2], 'r')

# plt.figure(8)
# plt.title('thrust')
# plt.plot(t, u, 'b')

# plt.figure(9)
# plt.plot(t, p_error[:,0], 'b')
# plt.plot(t, p_error[:,1], 'g')
# plt.plot(t, p_error[:,2], 'r')
 
plt.show()
