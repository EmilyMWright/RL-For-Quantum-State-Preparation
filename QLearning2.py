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
alpha = 0.99; gamma = 0.6; epsilon = 0.9
iters = 10000; N = 10; T = np.pi; dt = T/N
h_min = -1; h_max = 1; M = 3

# initial state and target
psi0 = np.array([complex(1,0),complex(0,0)])
psit = np.array([complex(0,0),complex(1,0)])

# action space
actions = np.linspace(h_min,h_max,M)

# Q-table
Q = np.zeros(shape=(900,len(actions)))

# count iterations
count = 0

for j in range(iters):
    # initialize
    psi = psi0
    state = nbin(psi)
    count = count + 1
    #print(count)
    #rewards = [reward(psi,psit)]
    #visited = [psi]
    for i in range(N):
        # choose action
        random.seed()
        # explore
        if random.random() > epsilon:
            k = random.randint(0,M-1)
        # exploit
        else:
            row = Q[state, :]
            k = np.argwhere(row == max(row))
            if len(k) > 1:
                k =random.choice(k)
        k = int(k)
        h = actions[k]

        # evolve state
        psi = evolve(psi,h,dt)

        # break if reached target state
        if 1-F(psi,psit) < 10**(-3):
            break

        prev_state = state
        state = nbin(psi)
        #visited.append(psi)

        # calculate reward
        r = reward(psi,psit)
        #rewards.append(r)

        # update Q-table
        Q[prev_state,k] = Q[prev_state,k] + alpha*(r + gamma*np.max(Q[state,:]) - Q[prev_state,k])

# test
controls = []
visited = [psi0]
psi = psi0
state = nbin(psi)
for j in range(N):
    # choose action
    random.seed()
    # explout
    row = Q[state, :]

    k = np.argwhere(row == max(row))
    if len(k) > 1:
        k =random.choice(k)
    k = int(k)
    h = actions[k]
    
    # track
    controls.append(h)

    # new state
    psi = evolve(psi,int(h),dt)
    visited.append(psi)

    # get closest allowed state
    state = nbin(psi)
        
    # break if reached target state
    if 1-F(psi,psit) < 10**(-3):
        break

print(Q)
print(controls)
print(visited)

q0 = basis(2,0)
q1 = basis(2,1)

s = [vis[0]*q0+vis[1]*q1 for vis in visited]

# ##### PLOTTING ######
fig = plt.figure()
b = Bloch(fig=fig)
b.add_states(s)
#b.add_states(psit)
#b.add_points([0,0,0])
b.render(fig=fig)
plt.show()