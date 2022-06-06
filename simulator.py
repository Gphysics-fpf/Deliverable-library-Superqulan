import numpy as np 
from scipy import sparse
import itertools, operator
import numpy as np
from numpy import floor, sqrt
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from math import pi as π
import math
import copy
import sympy as sym




def Trotter_solver_dynamics(time,v0,control,g1=None,g2=None):
    vt = [v0]
    last_t = time[0]
    for t in time[1:]:
        dt = t - last_t
        v0 = scipy.sparse.linalg.expm_multiply((-1j*dt) * control.Hamiltonian(g1=g1(t), g2=g2(t)), v0)
        vt.append(v0)
        last_t = t
    return np.array(vt).T

'''We have developed a lot of controls over the past months. Think where and how to accomodate it. '''

def gt_sech_mod(t, κ, σ):
    
    η = 2*σ

    if η != 1:
        print('ERROR, we have not upgraded this function yet.')
        # th = np.tanh(t*σ)
        ex = np.exp(-2*t*σ)

        return (κ - 2*σ*th)/np.sqrt(2*(κ*(1+ex)-2*σ)/σ)
    
    else:
        return κ / 2 / np.cosh(κ*t/2)