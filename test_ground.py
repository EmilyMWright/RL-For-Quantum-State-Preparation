from random import seed
import random
from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from cmath import cos,sin,exp,acos,asin
from scipy.linalg import expm


def angles(psi):
    a = complex(psi[0]).real
    theta = 2*acos(a)
    if theta.real > np.pi:
        theta = 2*np.pi - theta
    b = psi[1].real
    k = sin(theta/2).real
    if k == 0:
        return (theta,0)
    phi = acos(b/k)
    return (theta.real,phi.real)

def points(theta,phi):
    x = sin(theta)*cos(phi)
    y = sin(theta)*sin(phi)
    z = cos(theta)
    return [x,y,z]

def angles(psi):
    a = complex(psi[0]).real
    theta = 2*acos(a)
    if theta.real > np.pi:
        theta = 2*np.pi - theta
    b = psi[1].real
    k = sin(theta/2).real
    if k == 0:
        return (theta,0)
    phi = acos(b/k)
    return (theta.real,phi.real)


# parameters
N = 10; T = np.pi; dt = T/N

# initial state and target
psi0 = np.array([1,0])
psit = np.array([0,1])

h = 0
psi = psi0
rewards = [reward(psi,psit)]
visited = [psi]

for i in range(N):
    psi = evolve(psi,0,dt)
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