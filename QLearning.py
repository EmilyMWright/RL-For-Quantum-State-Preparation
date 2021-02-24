from random import seed
import random
from qutip import *
import matplotlib.pyplot as plt
import numpy as np
from cmath import cos,sin, exp

def closest_state(state, allowed_states):
    #return max(allowed_states, key=lambda x:fidelity(state,x))
    return np.argmax([fidelity(state,x) for x in allowed_states])

def reward(state, target):
    F = fidelity(state, target)
    if F < 0.999:
        return 100*(F**3)
    else:
        return 5000

def evolve(psi0,h,t):
    H = -sigmax() - h*sigmaz()
    times = np.linspace(0,t,100)
    result = mesolve(H, psi0, times, [])
    return result.states[-1]

q0 = basis(2,0)
q1 = basis(2,1)

# parameters
alpha = 0.99; gamma = 0.8; epsilon = 0.5
iters = 200; N = 10; T = np.pi; dt = T/N
h_min = -1; h_max = 1; M = 3

# allowed states
s_theta = np.linspace(0,np.pi,10)
s_phi = np.linspace(0,np.pi,10)

#states = [i for i in itertools.product(*[s_theta,s_phi])]

# action and state space
actions = np.linspace(h_min,h_max,M)

states = [q0]
for theta in s_theta[1:]:
    for phi in s_phi:
        states.append(cos(theta/2)*q0 + exp(phi*1j)*sin(theta/2)*q1)
        states.append(cos(theta/2)*q0 - exp(phi*1j)*sin(theta/2)*q1)

count = 0

# initial state and control
psi0 = q0

# target state
psit = q1

# Q-learning
Q = np.zeros(shape=(len(states),len(actions)))

for i in range(iters):
    print(i)
    exp = np.floor(i/20) + 1
    # alpha = 0.9**(exp/20)
    epsilon = 1-0.5**exp
    count = count + 1
    psi = psi0
    state = closest_state(psi0,states)
    control = []
    rewards = []
    #visited = [psi0]
    for j in range(N):
        # select action
        seed()
        if random.random() > epsilon:
            act = random.randint(0,len(actions))
        else:
            acts = Q[state, :]
            act = np.argwhere(acts == max(acts))
            if len(act) > 1:
                act = random.choice(act)
            else:
                act = act[0]
        h = act - 1
        
        # track
        control.append(int(h))

        # new state
        psi = evolve(psi,int(h),dt)
        #visited.append(psi)
        
        # break if reached target state
        if 1-fidelity(psi,psit) < 10**(-3):
            break

        # calculate reward
        r = reward(psi,psit)
        rewards.append(r)
        
        prev_state = state

        # get closest allowed state
        state = closest_state(psi,states)

        # update Q-table
        Q[prev_state,act] = Q[prev_state,act] + alpha*(r + gamma*np.max(Q[state,:]) - Q[prev_state,act])

# test
control = []
visited = [psi0]
state = closest_state(psi0,states)
psi = psi0
for j in range(N):
        # select action
        seed()
        acts = Q[state, :]
        act = np.argwhere(acts == max(acts))
        if len(act) > 1:
            act = random.choice(act)
        else:
            act = act[0]
        h = actions[act]
        
        # track
        control.append(int(h))

        # new state
        psi = evolve(psi,int(h),dt)
        visited.append(psi)

        # get closest allowed state
        state = closest_state(psi,states)
        
        # break if reached target state
        if 1-fidelity(psi,psit) < 10**(-3):
            break

print(Q)
#print("Rewards -", rewards)
#print("Total iterations -", count)
print("Final state -", psi.full())
print("Control sequence -", control)

#x=rsinθcosϕ, y=rsinθsinϕ, z=rcosθ

##### PLOTTING ######
fig = plt.figure()
b = Bloch(fig=fig)
b.add_states(visited)
#b.add_states(psit)
#b.add_points([0,0,0])
b.render(fig=fig)
plt.show()