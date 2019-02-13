#! usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
import control

def trajectory(time, ref_array, init_p):
    zeros = np.zeros_like(time)
    x_ref = np.array([ref_array, zeros, zeros,zeros])
    X0 = np.array([init_p,0,0,0])
    # print(x_ref)

    A_p = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],[0, 0, 0, 0]])
    B_p = np.array([[0],[0],[0],[1]])
    C_p = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    D_p = np.array([[0],[0],[0],[0]])
    poles_p = np.array([-2,-3,-4,-5])

    K_p = signal.place_poles(A_p, B_p, poles_p).gain_matrix
    B_p_new = np.matmul(B_p,K_p)                                # here is "+"" because of negative feedback loop and positive input
    D_p_new = np.matmul(D_p,K_p)

    # print(B_p_new)
    sys = control.StateSpace(A_p,B_p_new,C_p,D_p_new)
    sys2= control.StateSpace([],[],[],np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])) 
    
    sys_f = control.feedback(sys,sys2)
    tout, yout, states = control.forced_response(sys_f, T=time, U=x_ref, X0=X0)

    p_4dot = np.matmul(-B_p_new,yout-x_ref)
    
    x = np.vstack([yout, p_4dot[3,:]])

    return x