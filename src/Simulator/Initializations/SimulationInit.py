import sys
from abc import abstractmethod, ABCMeta
import numpy as np
from holonomic_agent import Agent, mktr


class SimulationInit(metaclass=ABCMeta):

    def create(self, N: int, L: float, agentWidth: float = 0.06, bufferDistance: float = 0.01):
        init = []
        tmp_agent_list = []

        initialPositions = self.calcolatePositions(N, L,agentWidth, bufferDistance)

        for j in range(N):
            while True:
                new_agent = Agent(np.matmul(np.eye(3), mktr(initialPositions[j], 0)),
                                  len(tmp_agent_list))
                if new_agent.check_collisions(tmp_agent_list, L):
                    tmp_agent_list.append(new_agent)
                    init.append(new_agent.pose)
                    break
                    # assert che non ci siano collisioni
                else:
                    print('inizializzazione agenti fallita')
                    sys.exit()

        del tmp_agent_list
        return init

    @abstractmethod
    def calcolatePositions(self, N: int, L: float, agentWidth: float, bufferDistance: float):
        pass
