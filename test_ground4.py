import numpy as np
from cmath import cos,sin,exp
from math import acos, asin, atan, isclose

def angles(psi):
    c = complex(psi[0]).real
    theta = 2*acos(c)
    if (theta == 0) | (isclose(theta.real,np.pi)): 
        return (theta,0)
    #k = sin(theta/2).real
    a = complex(psi[1]).real
    b = -complex(psi[1]).imag
    if (a == 0): 
        return (theta,0)
    if a > 0:
        phi = atan(b/a)
    else:
        phi = atan(b/a) + np.pi
    if theta.real > np.pi:
        theta = 2*np.pi - theta
    return (theta.real,phi.real)

def nbin(psi):
    theta = angles(psi)[0]
    phi = angles(psi)[1]
    if theta > 0:
        i = np.floor(30*theta/np.pi)
    else:
        i = 0
    if phi > 0:
        j = np.floor(30*phi/np.pi)
    else:
        j = 0
    return int(20*i + j)

# allowed states
s_theta = np.linspace(0,np.pi,10)
s_phi = np.linspace(0,2*np.pi,20)

states = []
for theta in s_theta:
    for phi in s_phi:
        state = [cos(theta/2), exp(phi*1j)*sin(theta/2)]
        if not (isclose(angles(state)[1],phi)):
            print("angles:",theta, phi)
            print("state cos(theta/2), exp(phi*1j)*sin(theta/2):", state)
            print("angles from state:", angles(state))
        #print(nbin(state))
        state = [cos(theta/2), -exp(phi*1j)*sin(theta/2)]
        if  not (isclose(angles(state)[1],phi)):
            print("angles:",theta, phi)
            print("state cos(theta/2), -exp(phi*1j)*sin(theta/2):", state)
            print("angles from state:", angles(state))
        #print(nbin(state))
