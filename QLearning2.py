
################################################################################
# QLearning2.py - Q-learning algorithm for quantum state preparation           #
#                                                                              #
# Reinforcement learning algorithm to discover a sequence of discrete controls #
# to bring one qubit from an initial state toward a desired state.             #
# The algorithm will maximize fidelity between the final state achieved by     # 
# the control and the desired state.                                           # 
#                                                                              #
# Author: Emily M. Wright                                                      #
# Date: February 2021                                                          #
#                                                                              #
################################################################################

import random
from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from cmath import cos,sin,exp
from math import acos, asin, isclose
from scipy.linalg import expm

# Pauli matrices
SIGMA_X = np.matrix([[0,1],[1,0]])
SIGMA_Z = np.matrix([[1,0],[0,-1]])

################################################################################
# Function: F                                                                  #
#                                                                              #
# Purpose: Calculate fidelity between two quantum states                       #
#                                                                              #
# Arguments:                                                                   #
#   psi      list of two complex numbers    quantum state                      #
#   psi0     list of two complex numbers    quantum state                      #
#                                                                              #
# Returns: fidelity |<psi*|psi0>|^2                                            #
#                                                                              #
################################################################################
def F(psi,psi0):
    a = np.vdot(psi,psi0)
    return float(abs(a)**2)

################################################################################
# Function: reward                                                             #
#                                                                              #
# Purpose: Calculate reward for current state given target state               #
#                                                                              #
# Arguments:                                                                   #
#   state      list of two complex numbers    current quantum state            #
#   target     list of two complex numbers    target quantum state             #
#                                                                              #
# Returns: Reward based on fidelity between states                             #
#                                                                              #
################################################################################
def reward(state, target):
    # fidelity
    f = F(state, target)
    if f < 0.999:
        return 100*f**3
    else:
        return 5000

################################################################################
# Function: evolve                                                             #
#                                                                              #
# Purpose: Evolve a quantum state according to the generalized Shrodinger      #
#          equation                                                            #
#                                                                              #
# Arguments:                                                                   #
#   psi0      list of two complex numbers    quantum state                     #
#   h         int                            control magnetic field            #
#   dt        float                          time step                         #
#                                                                              #
# Returns: New quantum state                                                   #
#                                                                              #
################################################################################
def evolve(psi0,h,dt):
    # noise
    sigma = np.random.normal(loc=0.0, scale=0.1, size=None)
    #sigma = 0
    # Hamiltonian
    H = -(1+sigma)*SIGMA_X - h*SIGMA_Z
    # unitary operator
    U = expm(dt*(-1j)*H)
    return np.dot(U,psi0)


################################################################################
# Function: angles                                                             #
#                                                                              #
# Purpose: Converts a state a|0> + b|1> to a pair of angles (theta,phi)        #
#                                                                              #
# The conversion arises from the representation                                #
# cos(theta/2)|0> + e^(i*phi)sin(theta/2)|1>                                   #
# theta in [0,pi], phi in [0,2pi]                                              #
#                                                                              #
# Arguments:                                                                   #
#   psi      list of two complex numbers    quantum state                      #
#                                                                              #
# Returns: Angles theta, phi                                                   #
#                                                                              #
################################################################################
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
    # account for rounding errors
    if b/k > 1:
        k = b
    if b/k < -1:
        k = -b
    phi = acos(b/k)
    # phase
    if c<0:
        phi = 2*np.pi-phi
    return theta.real,phi.real

################################################################################
# Function: nbin                                                               #
#                                                                              #
# Purpose: Quantizes the state space                                           #
#                                                                              #
# Slices state space into intervals of length pi/K for both angles             #
# Returns indices of Q-table for corresponding state "bin"                     #
#                                                                              #
# Arguments:                                                                   #
#   psi      list of two complex numbers    quantum state                      #
#   K        int                            slice size                         #
#                                                                              #
# Returns: i,j                                                                 #
#                                                                              #
################################################################################
def nbin(psi,K):
    theta, phi = angles(psi)
    if theta > 0:
        i = np.floor(K*theta/np.pi)
    else:
        i = 0
    if phi > 0:
        j = np.floor(K*phi/np.pi)
    else:
        j = 0
    return int(i),int(j)

################################## Q-learning ##################################

################################### Training ###################################

# parameters
# alpha - learning rate | gamma - discount factor | epsilon - exploitation probability | 
alpha = 0.99; gamma = 0.6; epsilon = 0.9
# iters - iterations | N - number of controls | T - total time | dt - length of time each control is applied
iters = 5000; N = 30; T = np.pi; dt = T/N
# h_min - smallest control | h_max largest control | M = number of control values
h_min = -1; h_max = 1; M = 3
# K - length of interval for quantization is pi/K
K = 30

# initial state and target
psi0 = np.array([complex(1,0),complex(0,0)])
psit = np.array([complex(0,0),complex(1,0)])

# action space
actions = np.linspace(h_min,h_max,M)

# Q-table
Q = np.zeros(shape=(62,62,len(actions)))

for j in range(iters):
    # initialize
    psi = psi0
    st,sp = nbin(psi,K)
    for i in range(N):
        # choose action
        random.seed()
        # explore
        if random.random() > epsilon:
            k = random.randint(0,M-1)
        # exploit
        else:
            row = Q[st,sp,:]
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

        prev_sp = sp
        prev_st = st
        sp, st = nbin(psi,K)

        # calculate reward
        r = reward(psi,psit)

        # update Q-table
        Q[prev_st,prev_sp,k] = Q[prev_st,prev_sp,k] + alpha*(r + gamma*np.max(Q[st,sp,:]) - Q[prev_st,prev_sp,k])

#################################### Testing ###################################

controls = []
visited = [psi0]
psi = psi0
st,sp = nbin(psi,K)
for j in range(N):
    # choose action
    random.seed()
    
    # exploit
    row = Q[st,sp, :]
    k = np.argwhere(row == max(row))
    if len(k) > 1:
        k =random.choice(k)
    h = actions[int(k)]
    
    # track controls
    controls.append(h)

    # new state
    psi = evolve(psi,int(h),dt)

    # track states
    visited.append(psi)
        
    # break if reached target state
    if 1-F(psi,psit) < 10**(-3):
        break

# print(Q)
print(controls)
# print(visited)

################################# Visualization ################################

# Visualize states
q0 = basis(2,0)
q1 = basis(2,1)

s = [vis[0]*q0+vis[1]*q1 for vis in visited]

###### PLOTTING ######
fig = plt.figure()
b = Bloch(fig=fig)
b.add_states(s)
b.render(fig=fig)
plt.show()