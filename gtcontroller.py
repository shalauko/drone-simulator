import numpy as np 
from numpy import linalg as LA
from matplotlib import pyplot as plt 

def gt_controller(p, p_dot, p_d, p_d_dot, p_d_2dot, p_d_3dot, p_d_4dot, \
    psi_d, psi_d_dot, psi_d_2dot, omega, R, m, J, g, Kp, Kv, K_R, K_omega):

    e3 = np.array([0,0,1]).reshape(3,1)
    p = p.reshape(3,1) 
    p_dot = p_dot.reshape(3,1)
    p_d = p_d.reshape(3,1)
    p_d_dot = p_d_dot.reshape(3,1)
    p_d_2dot = p_d_2dot.reshape(3,1)
    p_d_3dot = p_d_3dot.reshape(3,1)
    p_d_4dot = p_d_4dot.reshape(3,1)
    omega = omega.reshape(3,1)

    e_p = p - p_d

    e_p_dot = p_dot - p_d_dot

    x_c = np.array([np.cos(psi_d), np.sin(psi_d), 0]).reshape(3,1)

    # t1 = np.array([-np.sin(psi_d), np.cos(psi_d), 0]).reshape(3,1)
    # t2 = np.array([-np.cos(psi_d), np.sin(psi_d), 0]).reshape(3,1)
    # x_c_dot = psi_d_dot * t1
    # x_c_2dot = psi_d_2dot * t1 + (psi_d_dot**2) * t2

    A = -Kp*e_p - Kv*e_p_dot - m*g*e3 + m*p_d_2dot                      # temporary variable
    z_d_B = - A/LA.norm(A)
    B = np.cross(z_d_B.reshape(1,3), x_c.reshape(1,3)).reshape(3,1)     # temporary variable                         # temporary variable
    x_d_B = np.cross(B.reshape(1,3), z_d_B.reshape(1,3)).reshape(3,1) / LA.norm(B)
    y_d_B = np.cross(z_d_B.reshape(1,3), x_d_B.reshape(1,3)).reshape(3,1)
    R_d = np.dstack([x_d_B, np.dstack([y_d_B, z_d_B])]).squeeze()

    u = - A.reshape(1,3) @ (R@e3)

    omega_d = np.zeros(3).reshape(3,1)
    omega_d_dot = np.zeros(3).reshape(3,1)

    t3 = (R_d.transpose() @ R - R.transpose() @ R_d)
    t3_vec = np.array([t3[2,1], t3[0,2], t3[1,0]]).reshape(3,1)
    e_R = 0.5 * t3_vec

    e_omega = omega - R.transpose() @ R_d @ omega_d

    omega_s =np.array([0, -omega[2], omega[1], omega[2], 0, -omega[0], -omega[1], omega[0], 0]).reshape(3,3)
    tau = - K_R*e_R - K_omega*e_omega + omega_s @ J @ omega \
        - J @ (omega_s @ R.transpose() @ R_d @ omega_d - R.transpose() @ R_d @ omega_d_dot)
        
    return u, tau.reshape(1,3)
    