import numpy as np 
from numpy import linalg as LA
from matplotlib import pyplot as plt 

def nh_controller(p, p_dot, p_d, p_d_dot, p_d_2dot, psi_d, eta, omega,  e_I, m, J, g, kd, kp, ki, k_psi, K_eta, K_eta_dot, max_pos_err_norm, max_tilt):
    
    p = p.reshape(3,1)
    p_dot = p_dot.reshape(3,1)
    eta = eta.reshape(3,1)
    omega = omega.reshape(3,1)
    p_d = p_d.reshape(3,1)
    p_d_dot = p_d_dot.reshape(3,1)
    p_d_2dot = p_d_2dot.reshape(3,1)
    e_I = e_I.reshape(3,1)

    
    phi = eta[0]
    theta = eta[1]
    psi = eta[2]

    e_p = p_d - p

    ## saturations  
    sat_norm = (2 * max_pos_err_norm / np.pi) * np.arctan(np.pi * LA.norm(e_p) / (2*max_pos_err_norm))
    if LA.norm(e_p) > 0.0001:
        e_p = e_p/LA.norm(e_p) * sat_norm

    max_sin_theta_phi = np.array([np.sin(max_tilt), np.sin(max_tilt)]).reshape(2,1)

    ## thrust controll
    e_p_dot = p_d_dot - p_dot

    PD = kd*e_p_dot[:2] + kp*e_p[:2]                                  # in fact we use PD only for x,y axes
    PID = kd*e_p_dot[2] + kp*e_p[2] + ki*e_I[2]                      # in fact we use PID only for z axis

    u = (-m / (np.cos(phi)*np.cos(theta)))*(-g + p_d_2dot[2] +  PID)

    Q = np.array([np.cos(phi)*np.cos(psi), np.sin(psi), np.cos(phi)*np.sin(psi), -np.cos(psi)]).reshape(2,2)       # inv(Q) = [cos(psi)/cos(phi) sin(psi)/cos(phi); sin(psi) -cos(psi)]
        
    sin_theta_phi_d = - m/u * LA.inv(Q) @ PD                           # pay attention theta first, and then in phi                                                          

    sin_theta_phi_d = (2 * max_sin_theta_phi / np.pi) * np.arctan( np.pi * sin_theta_phi_d / (2 * max_sin_theta_phi))       # soft saturation
    
    eta_d = np.append([np.arcsin(sin_theta_phi_d[1]), np.arcsin(sin_theta_phi_d[0])], psi_d).reshape(3,1) 
    psi_d_dot = k_psi * np.arctan2(np.sin(psi_d - psi), np.cos(psi_d - psi))

    ## torgue controll
    T = np.array([1, np.tan(theta)*np.sin(phi), np.tan(theta)*np.cos(phi), 0, np.cos(phi), -np.sin(phi), 0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]).reshape(3,3)

    eta_dot = T@omega
    eta_d_dot = np.array([0, 0, psi_d_dot]).reshape(3,1)

    e_eta = eta_d - eta  
    e_eta_dot = eta_d_dot - eta_dot

    omega_s =np.array([0, -omega[2], omega[1], omega[2], 0, -omega[0], -omega[1], omega[0], 0]).reshape(3,3)
    
    tau = omega_s@J@omega + J@LA.inv(T)@(K_eta_dot*e_eta_dot + K_eta*e_eta)
 
    return u, tau.reshape(1,3), e_p