import random
from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from cmath import cos,sin,exp
from math import acos, asin, isclose
from scipy.linalg import expm

SIGMA_X = np.matrix([[0,1],[1,0]])
SIGMA_Z = np.matrix([[1,0],[0,-1]])

def F(psi,psi0):
    a = np.vdot(psi,psi0)
    return float(abs(a)**2)

def reward(state, target):
    f = F(state, target)
    if f < 0.999:
        return 100*f**3
    else:
        return 5000

def evolve(psi0,h,dt):
    H = -SIGMA_X - h*SIGMA_Z
    U = expm(dt*(-1j)*H)
    return np.dot(U,psi0)

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
        i = np.floor(30*theta/np.pi)
    else:
        i = 0
    if phi > 0:
        j = np.floor(30*phi/np.pi)
    else:
        j = 0
    return int(30*i + j)

# parameters
N = 10; T = np.pi; dt = T/N

# initial state and target
psi0 = np.array([1,0])
psit = np.array([0,1])

h = 0
psi = psi0
rewards = [reward(psi,psit)]
visited = [psi]

h = [-1,0,1]
for i in range(3):
    psi = evolve(psi0,h[i],dt)
    print(psi)
    print(angles(psi))
    print(nbin(psi))
    r = reward(psi,psit)
    rewards.append(r)
    visited.append(psi)
    if 1-F(psi, psit) < 10**(-3):
        break

print(rewards)

print(visited)

q0 = basis(2,0)
q1 = basis(2,1)

s = [complex(vis[0])*q0+complex(vis[1])*q1 for vis in visited]

# ##### PLOTTING ######
fig = plt.figure()
b = Bloch(fig=fig)
b.add_states(s)
#b.add_states(psit)
#b.add_points([0,0,0])
b.render(fig=fig)
plt.show()