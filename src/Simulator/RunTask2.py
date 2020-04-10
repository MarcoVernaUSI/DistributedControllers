from typing import List

import numpy as np

from SimulatorRun import Run

class RunTask2 (Run):

    def runStep(self, control, goalList, agents_list, state, mas_vel)  -> float:
        e = []
        for i in range(len(control)):
            e.append(self.Cross_Entropy(control[i], goalList[i]))
        return np.mean(np.array(e))

    def Cross_Entropy(self, y_hat, y):
        if y == 1:
            return -np.log(y_hat)
        else:
            return -np.log(1 - y_hat)