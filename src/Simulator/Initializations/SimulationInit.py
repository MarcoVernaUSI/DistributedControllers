import sys
from abc import abstractmethod, ABCMeta
import numpy as np
from holonomic_agent import Agent, mktr


class SimulationInit(metaclass=ABCMeta):
    def create(self, n: int, l: float, agentwidth: float = 0.06, bufferdistance: float = 0.01):
        init = []
        tmp_agent_list = []

        initialpositions = self.calcolatePositions(n, l, agentwidth, bufferdistance)

        for j in range(n):
            while True:
                new_agent = Agent(np.matmul(np.eye(3), mktr(initialpositions[j], 0), width=agentwidth),
                                  len(tmp_agent_list))
                if new_agent.check_collisions(tmp_agent_list, l):
                    tmp_agent_list.append(new_agent)
                    init.append(new_agent.pose)
                    break
                else:
                    print('inizializzazione agenti fallita')
                    sys.exit()
        del tmp_agent_list
        return init

    @abstractmethod
    def calcolatePositions(self, n: int, l: float, agentwidth: float, bufferdistance: float):
        pass
