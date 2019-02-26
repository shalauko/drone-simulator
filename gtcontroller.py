import numpy as np 
from numpy import linalg as LA
from matplotlib import pyplot as plt 

def gt_controller(p, p_dot, p_d, p_d_dot, p_d_2dot, p_d_3dot, p_d_4dot, \
    psi_d, psi_d_dot, psi_d_2dot, omega, R, m, J, g, Kp, Kv, K_R, K_omega, calculate_omega_d=True):

    e3 = np.array([0,0,1])
    omega_s =np.array([0, -omega[2], omega[1], omega[2], 0, -omega[0], -omega[1], omega[0], 0]).reshape(3,3)

    # error in position and velosity
    e_p = p - p_d
    e_p_dot = p_dot - p_d_dot

    x_c = np.array([np.cos(psi_d), np.sin(psi_d), 0])

    A = -Kp*e_p - Kv*e_p_dot - m*g*e3 + m*p_d_2dot                                  # temporary variable
    z_d_B = - A/LA.norm(A)
    B = np.cross(z_d_B, x_c)               # temporary variable
    x_d_B = np.cross(B, z_d_B) / LA.norm(B)
    y_d_B = np.cross(z_d_B, x_d_B)
    R_d = np.vstack([x_d_B, np.vstack([y_d_B, z_d_B])]).transpose()
    
    # steering thrust
    u = - A @ (R@e3)

    ## computations for steering torques
    if calculate_omega_d == False:
        omega_d = np.zeros(3)
        omega_d_dot = np.zeros(3)

    elif calculate_omega_d == True:
        e3 = np.array([0,0,1])
        t1 = np.array([-np.sin(psi_d), np.cos(psi_d), 0])
        t2 = np.array([-np.cos(psi_d), np.sin(psi_d), 0])
        x_c_dot = psi_d_dot * t1
        x_c_2dot = psi_d_2dot * t1 + (psi_d_dot**2) * t2

        R_dot = R @ omega_s
        e_p_2dot = g*e3 - (1/m) * u * R @ e3 - p_d_2dot
        A_dot = -Kp*e_p_dot - Kv*e_p_2dot + m*p_d_3dot
        u_dot = -A_dot @ (R@e3) + A @ (R_dot@e3)
        e_p_3dot = -(1/m)*(u_dot * R@e3 + u * R_dot@e3 + m*p_d_3dot)
        A_2dot = -Kp*e_p_2dot - Kv*e_p_3dot + m*p_d_4dot

        normA = LA.norm(A)
        normA_dot = (A@A_dot)/normA
        normA_2dot = (1/normA)*(A@A_dot + A@A_2dot) - (1/normA**2)*A@A_dot*normA_dot
        
        z_d_B_dot = - (A_dot/normA - A*normA_dot/normA**2)
        z_d_B_2dot = - (A_2dot/normA - 2*A_dot*normA_dot/normA**2 - (A/normA**2)*(normA_2dot - 2*normA_dot**2/normA))

        B_dot = np.cross(z_d_B_dot, x_c) + np.cross(z_d_B, x_c_dot)
        B_2dot = np.cross(z_d_B_2dot, x_c) + 2*np.cross(z_d_B_dot, x_c_dot) + np.cross(z_d_B,x_c_2dot)

        normB = LA.norm(B)
        normB_dot = (B@B_dot)/normB
        normB_2dot = (1/normB)*(B_dot@B_dot + B@B_2dot) - (1/normB**2)*B@B_dot*normB_dot
        
        x_d_B_dot = (np.cross(B_dot,z_d_B) + np.cross(B,z_d_B_dot))*(1/normB) - np.cross(B,z_d_B)*normB_dot/normB**2
        temp = (np.cross(B_dot,z_d_B) + np.cross(B,z_d_B_dot))/normB**2
        x_d_B_2dot = (np.cross(B_2dot,z_d_B) + 2*np.cross(B_dot,z_d_B_dot) \
            + np.cross(B,z_d_B_2dot))*(1/normB) - normB_dot*temp \
                - np.cross(B,z_d_B)*normB_2dot/normB**2 + (temp \
                    - 2*np.cross(B,z_d_B)*normB_dot/normB**3)*normB_dot

        y_d_B_dot = np.cross(z_d_B_dot, x_d_B) + np.cross(z_d_B, x_d_B_dot)
        y_d_B_2dot = np.cross(z_d_B_2dot, x_d_B) + 2*np.cross(z_d_B_dot, x_d_B_dot) + np.cross(z_d_B, x_d_B_2dot)

        R_d_dot = np.vstack([x_d_B_dot, np.vstack([y_d_B_dot, z_d_B_dot])]).transpose()
        R_d_2dot = np.vstack([x_d_B_2dot, np.vstack([y_d_B_2dot, z_d_B_2dot])]).transpose()

        omega_d_matrix = R_d.transpose() @ R_d_dot
        omega_d = np.array([omega_d_matrix[2,1], omega_d_matrix[0,2], omega_d_matrix[1,0]])

        omega_d_dot_matrix = R_d.transpose()@R_d_2dot - omega_d_matrix**2
        omega_d_dot = np.array([omega_d_dot_matrix[2,1], omega_d_dot_matrix[0,2], omega_d_dot_matrix[1,0]])

    # error in rotation matrix
    t3 = (R_d.transpose() @ R - R.transpose() @ R_d)                # temporary
    t3_vec = np.array([t3[2,1], t3[0,2], t3[1,0]])                  # temporary
    e_R = 0.5 * t3_vec

    # error in angular velocity
    e_omega = omega - R.transpose() @ R_d @ omega_d

    # steering torques
    tau = - K_R*e_R - K_omega*e_omega + omega_s @ J @ omega \
        - J @ (omega_s @ R.transpose() @ R_d @ omega_d - R.transpose() @ R_d @ omega_d_dot)
        
    return u, tau
    