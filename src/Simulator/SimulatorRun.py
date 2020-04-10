from typing import List
from abc import abstractmethod, ABCMeta
import numpy as np
from network import Controller
from dataset import Trace


class Run(metaclass=ABCMeta):

    def __init__(self, L,controller: Controller, dt: float = 0.1):
        self.controller = controller
        self.dt = dt
        self.L = L

    def __call__(self, state_c, agents_list, goalList, mas_vel, state_size, epsilon: float = 0.01, T: float = np.inf
                 ) -> Trace:
        t = 0.0
        steps: List[Trace] = []


        while (t < T) or t == 0:
            state = state_c[:,0].tolist()
            sensing = state_c[:,3:].tolist()

            control, *communication = self.controller(state, sensing)
            if communication:
                communication = communication[0]

            e = self.runStep(control, goalList, agents_list, state, mas_vel)


            for i in range( 0, len(agents_list)):
                agents_list[i].state, agents_list[i].vels= agents_list[i].observe(agents_list,self.L, i)

            steps.append(Trace(t, state, communication, sensing, control, e))
            state_c = save_state(agents_list,state_size)
            t += 1
        return Trace(*[np.array(x) for x in zip(*steps)])


    @abstractmethod
    def runStep(self, control, goalList, agents_list, state, mas_vel)  -> float:
       pass



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
