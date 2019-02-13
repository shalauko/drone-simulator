#! usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import control

def trajectory(time, ref_array, init_p):
    zeros = np.zeros_like(time)
    x_ref = np.array([ref_array, zeros, zeros,zeros])
    X0 = np.array([init_p,0,0,0])

    A_p = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],[0, 0, 0, 0]])
    B_p = np.array([[0],[0],[0],[1]])
    C_p = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    D_p = np.array([[0],[0],[0],[0]])
    poles_p = np.array([-2,-2.1,-2.2,-2.3])

    K_p = signal.place_poles(A_p, B_p, poles_p).gain_matrix
    B_p_new = np.matmul(B_p,K_p)                                # here is "+"" because of negative feedback loop and positive input
    D_p_new = np.matmul(D_p,K_p)

    sys = control.StateSpace(A_p,B_p_new,C_p,D_p_new)

        # workaround for control lib - just transition input to output one to one; its plant on the feedback fork
    sys2= control.StateSpace([],[],[],np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])) 
    
    sys_f = control.feedback(sys,sys2)
    tout, yout, states = control.forced_response(sys_f, T=time, U=x_ref, X0=X0)

    # snap; but it's shifted on 1 point forward 
    p_4dot = np.matmul(-B_p_new,yout-x_ref)
    # adding initial snap == 0 and cuttin last point    
    x = np.vstack([yout, np.insert(p_4dot[3,0:-1], 0, 0)])

    return x

def yawtrajectory(time, ref_array, init_psi):
    zeros = np.zeros_like(time)
    psi_ref = np.array([ref_array, zeros])
    psi0 = np.array([init_psi,0])

    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0],[1]])
    C = np.array([[1,0],[0,1]])
    D = np.array([[0],[0]])
    poles = np.array([-2,-3])

    K = signal.place_poles(A, B, poles).gain_matrix
    B_new = np.matmul(B,K)                             
    D_new = np.matmul(D,K)

    sys = control.StateSpace(A, B_new, C, D_new)
    sys2= control.StateSpace([],[],[],np.array([[1,0], [0,1]])) 

    sys_f = control.feedback(sys,sys2)
    tout, yout, states = control.forced_response(sys_f, T=time, U=psi_ref, X0=psi0)

    psi_2dot = np.matmul(-B_new, yout - psi_ref)
    psi = np.vstack([yout, np.insert(psi_2dot[1,0:-1], 0, 0)])
    
    return psi