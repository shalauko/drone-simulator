import numpy as np 
import sys

def simulateDynamics(u, tau, R, omega, m, J, g=9.81, \
    tau_ext=np.zeros(3).reshape(3,1), f_ext=np.zeros(3).reshape(3,1)):

    # temporary helped vulues
    e3 = np.array([0,0,1]).reshape(3,1)
    skew_omega = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])

    # dynamics equations
    p_2dot = g*e3 + (1/m)*(np.matmul((-u*R),e3) + f_ext)
        # simplify omega_dot = inv(J)*(-skew_omega*J*omega + tau + tau_ext)
    omega_dot = np.matmul(np.linalg.inv(J),(np.matmul(np.matmul((-1)*skew_omega,J),omega) + tau + tau_ext))
    R_dot = np.matmul(R,skew_omega)

    # construct Euler angles
    phi = np.arctan2(R[2][1],R[2][2])
    theta = np.arctan2(-R[2][0],np.sqrt(R[2][1]**2+R[2][2]**2))
    psi = np.arctan2(R[1][0],R[0][0])
    eta=np.array([[phi], [theta], [psi]])

    return p_2dot, eta, R_dot, omega_dot

if __name__ == "__main__":
    simulateDynamics(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])