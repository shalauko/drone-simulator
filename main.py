import numpy as np 
import dynamics

# time step and time of simulation
ts = 0.1
T = 1

# parameters of drone
m = 1.2
J = np.diag([0.015, 0.015, 0.026])

# initial conditions
R0 = np.eye(3)
omega0 = np.zeros(3).reshape(3,1)
eta0 = np.zeros(3).reshape(3,1)

p_2dot0 = np.zeros(3).reshape(3,1)
p_dot0 = np.array([[[0]],[[0]],[[0]]]) # np.zeros(3).reshape(3,1)
p0 = np.zeros(3).reshape(3,1)

# construct output values
R = R0

p_2dot = p_2dot0
p_dot = p_dot0
p = p0

u = np.arange(11)
tau = np.array([[1],[1],[1]])

for i in range(1, int(T/ts+1)):
    #dynamics symulation
    p_2dot_t, eta, R_dot, omega_dot = dynamics.simulateDynamics(u[i-1], tau, R0, omega0, m, J)

    #collecting values
    p_2dot = np.dstack((p_2dot, p_2dot_t))

    # first integral of acceleration
    p_dot_t = p_dot[:,0,i-1] + np.trapz(np.array([p_2dot[:,0,i-1],p_2dot[:,0,i]]), dx=0.1, axis = 0)
    p_dot_t = p_dot_t.reshape(3,1)

    #collecting values
    p_dot = np.dstack((p_dot, p_dot_t))

print("p_dot \n", p_dot)
print("p_2dot \n", p_2dot)
print(eta)
print(R_dot)
print(omega_dot)

