from typing import List

from holonomic_agent import Agent, mktr, mkrot, atr
import numpy as np
import random as rand
from network import Controller
from dataset import Trace
import sys


class Run:

    def __init__(self, L,controller: Controller, dt: float = 0.1):
        self.controller = controller
        self.dt = dt
        self.L = L

    def __call__(self, state_c, agents_list, goal_list,  mas_vel, state_size,epsilon: float = 0.01, T: float = np.inf
                 ) -> Trace:
        t = 0.0
        dt = 0.1
        steps: List[Trace] = []
        L = self.L
        e = 10
#       while (e > epsilon and t < T) or t == 0:
        while (t < T) or t == 0:
            state = state_c[:,0].tolist()
            sensing = state_c[:,3:].tolist()

            control, *communication = self.controller(state, sensing)
            if communication:
                communication = communication[0]

            e = np.mean(abs(np.array(state)-np.array(goal_list)))


            # update
            for i,v in enumerate(control):
                v = np.clip(v, -mas_vel, +mas_vel)
                agents_list[i].step(v,dt)
                control[i]=v

                #check collisions
                if not agents_list[i].check_collisions(agents_list[:i]+agents_list[i+1:],L):
                    agents_list[i].setx(state[i])
                    agents_list[i].velocity= 0.0
                    control[i]=0.0

            for i in range( 0, len(agents_list)):
                agents_list[i].state, agents_list[i].vels= agents_list[i].observe(agents_list,L, i)

            steps.append(Trace(t, state, communication, sensing, control, e))
            state_c = save_state(agents_list,state_size)
            t += 1
        return Trace(*[np.array(x) for x in zip(*steps)])


def save_state(agents_list, state_size):

    n_agents=len(agents_list)
    state = np.zeros((n_agents, state_size))
    for i in range(n_agents):
        state[i,0]= agents_list[i].getxy()[0]
        state[i,1]= agents_list[i].getxy()[1]
        state[i,2]= agents_list[i].gettheta()
        for j in range(state_size-3):
            state[i,3+j]= np.concatenate((agents_list[i].state,agents_list[i].vels))[j]
    return state
