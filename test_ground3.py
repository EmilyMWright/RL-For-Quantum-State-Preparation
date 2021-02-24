import random
from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from cmath import cos,sin,exp
from math import acos, asin, isclose
from scipy.linalg import expm
import pandas as pd

from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

# Plotting points


def state_to_pt(state):
    a = state.full()[0]
    b = state.full()[1]
    if a == 0:
        return [0,-1*float(b.imag),-1*float(b.real)]
    else:
        u = b/a
        ux = u.real
        uy = u.imag

    Px = float(2*ux/(1 + ux**2 + uy**2))
    Py = float(2*uy/(1 + ux**2 + uy**2))
    Pz = float((1 - ux**2 - uy**2)/(1 + ux**2 + uy**2))
    return [Px,Py,Pz]

def points(theta,phi):
    x = sin(theta)*cos(phi)
    y = sin(theta)*sin(phi)
    z = cos(theta)
    return [x,y,z]

# def angles(psi):
#     a = complex(psi[0]).real
#     theta = 2*acos(a)
#     b = complex(psi[1]).real
#     if (theta == 0) | (isclose(theta.real,np.pi)):
#         return (theta,0)
#     k = sin(theta/2).real
#     if b/k > 1:
#         k = b
#     if b/k < -1:
#         k = -b
#     phi = acos(b/k)
#     if (complex(psi[1]).imag < 0):
#         phi = 2*np.pi - phi
#     if theta.real > np.pi:
#         theta = 2*np.pi - theta
#     return (theta.real,phi.real)

def angles(psi):
    a = complex(psi[0]).real
    theta = 2*acos(a)
    b = complex(psi[1]).real
    c = complex(psi[1]).imag
    if theta.real > np.pi:
        theta = 2*np.pi - theta
    if (theta == 0) | (isclose(theta.real,np.pi)):
        return (theta,0)
    
    k = sin(theta/2).real
    
    if b/k > 1:
        k = b
    if b/k < -1:
        k = -b
    phi = acos(b/k)
    if c<0:
        phi = 2*np.pi-phi
    return (theta.real,phi.real)

def nbin(psi):
    theta = angles(psi)[0]
    phi = angles(psi)[1]
    if theta > 0:
        i = np.floor(10*theta/np.pi)
    else:
        i = 0
    if phi > 0:
        j = np.floor(10*phi/np.pi)
    else:
        j = 0
    return int(20*i + j)

# allowed states
s_theta = np.linspace(0,np.pi,10)
s_phi = np.linspace(0,2*np.pi,20)


q0 = basis(2,0)
q1 = basis(2,1)

states = []
for theta in s_theta:
    for phi in s_phi:
        print("angles:",theta, phi)
        state = [cos(theta/2), exp(phi*1j)*sin(theta/2)]
        print("state cos(theta/2), exp(phi*1j)*sin(theta/2):", state)
        print("angles from state:", angles(state))
        print(nbin(state))
        print("\n")

# points = [state_to_pt(s) for s in states]

# # Create a sphere
# r = 1
# pi = np.pi
# cos = np.cos
# sin = np.sin
# phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
# x = r*sin(phi)*cos(theta)
# y = r*sin(phi)*sin(theta)
# z = r*cos(phi)

# #Set colours and render
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(
#     x, y, z,  rstride=1, cstride=1, alpha = 0.2, linewidth=0)

# xx = [p[0] for p in points]
# yy = [p[1] for p in points]
# zz = [p[2] for p in points]

# ax.scatter(xx,yy,zz,color="k",s=20)

# ax.set_xlim([-1,1])
# ax.set_ylim([-1,1])
# ax.set_zlim([-1,1])
# #ax.set_aspect("equal")
# plt.tight_layout()
# plt.show()