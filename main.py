import numpy as np 
from matplotlib import pyplot as plt 
import dynamics

# time step, time of simulation and timevector
ts = 0.1
T = 1
t = np.linspace(0, T, num=int(T/ts)+1)

# parameters of drone
m = 1.2
J = np.diag([0.015, 0.015, 0.026])

# initial conditions
R0 = np.eye(3)
omega0 = np.zeros(3).reshape(3,1)
eta0 = np.zeros(3).reshape(3,1)

p_2dot0 = np.zeros(3).reshape(3,1)
p_dot0 = np.array([[[0]],[[0]],[[0]]])  # that view of array we need for use dstack
p0 = np.array([[[0]],[[0]],[[0]]])      # that view of array we need for use dstack

# construct output values
R = R0

p_2dot = p_2dot0
p_dot = p_dot0
p = p0

u = 9.81 * m * np.ones(int(T/ts)+1)
tau = np.array([[1],[1],[1]])

for i in range(1, int(T/ts+1)):
    #dynamics symulation // _t means temporary
    p_2dot_t, eta, R_dot, omega_dot = dynamics.simulateDynamics(u[i-1], tau, R0, omega0, m, J)

    #collecting values
    p_2dot = np.dstack((p_2dot, p_2dot_t))

    # integration of acceleration, i.e get velocity
    p_dot_t = p_dot[:,0,i-1] + np.trapz(np.array([p_2dot[:,0,i-1],p_2dot[:,0,i]]), dx=0.1, axis = 0)
    p_dot_t = p_dot_t.reshape(3,1)

    # collecting values
    p_dot = np.dstack((p_dot, p_dot_t))

    # integration of velocity, i.e get position
    p_t = p[:,0,i-1] + np.trapz(np.array([p_dot[:,0,i-1],p_dot[:,0,i]]), dx=0.1, axis = 0)
    p_t = p_t.reshape(3,1)

    # collecting values
    p = np.dstack((p, p_t))

p = np.squeeze(p)
p_dot = np.squeeze(p_dot)
p_2dot = np.squeeze(p_2dot)
print("p \n", p[2,:])
print("p_dot \n", p_dot)
print("p_2dot \n", p_2dot)
# print(eta)
# print(R_dot)
# print(omega_dot)
plt.plot(t, p[2,:])
plt.show()

