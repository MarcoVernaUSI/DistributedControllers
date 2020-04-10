from typing import List

import numpy as np
from SimulatorRun import Run


class RunTask1(Run):

    def runStep(self, control, goalList, agents_list, state, mas_vel)  -> float:
        e = np.mean(abs(np.array(state) - np.array(goalList)))
        # update

        for i,v in enumerate(control):
            v = np.clip(v, -mas_vel, +mas_vel)
            agents_list[i].step(v,self.dt)
            control[i]=v

            #check collisions
            if not agents_list[i].check_collisions(agents_list[:i]+agents_list[i+1:],self.L):
                agents_list[i].setx(state[i])
                agents_list[i].velocity= 0.0
                control[i]=0.0
        return e