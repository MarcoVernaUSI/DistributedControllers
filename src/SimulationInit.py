import sys
from abc import abstractmethod, ABCMeta
import numpy as np
from holonomic_agent import Agent, mktr


class SimulationInit(metaclass=ABCMeta):
    def __init__(self, N: int, L: float, agentWidth: float, bufferDistance: float = 0.01):
        self.N: int = N
        self.L: float = L
        self.agentWidth: float = agentWidth
        self.bufferDistance: float = bufferDistance

    def create(self):
        init = []
        tmp_agent_list = []

        initialPositions = self.calcolatePositions()

        for j in range(self.N):
            while True:
                new_agent = Agent(np.matmul(np.eye(3), mktr(initialPositions[j], 0)),
                                  len(tmp_agent_list))
                if new_agent.check_collisions(tmp_agent_list, self.L):
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
    def calcolatePositions(self):
        pass
