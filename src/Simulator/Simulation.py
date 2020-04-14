import numpy as np
from abc import abstractmethod, ABCMeta
from SimulatorRun import save_state

class Simulation(metaclass=ABCMeta):
    def __init__(self, nAgents, actualL, dt, masVel, agentsList, timesteps, parameter):
        self.nAgents: int = nAgents
        self.actualL: float = actualL
        self.dt = dt
        self.masVel = masVel
        self.agentsList = agentsList  # in realà basta definire le caratteristiche dell'agent
        self.timesteps = timesteps
        self.states, self.targets, self.errors, self.stateSize = self.initializeParameters(parameter)
        self.goalList = self.initializeTargets()

    @abstractmethod
    def initializeTargets(self):
        pass

    def initializeParameters(self, parameter):
        if parameter == 'add_vel':
            state_size = len(self.agentsList[0].state) + len(self.agentsList[0].vels) + 3
        else:
            state_size = len(self.agentsList[0].state) + 3  # x, y, theta + stato
        states = np.zeros((self.timesteps, self.nAgents, state_size))
        targets = np.zeros((self.timesteps, self.nAgents))
        errors = np.zeros((self.timesteps,), dtype=np.float32)

        return states, targets, errors, state_size

    def saveInitialState(self):
        self.states[0] = save_state(self.agentsList, self.stateSize)

    @abstractmethod
    def runWithExpert(self):
        pass

    @abstractmethod
    def controllerRun(self, controller):
        pass

    def runWithNet(self, controller):
        net_run = self.controllerRun(controller)
        trace = net_run(self.states[0], self.agentsList, self.goalList, self.masVel,
                        self.stateSize, epsilon=0.01, T=self.timesteps)
        self.states[:trace.state.shape[0], :, 0] = trace.state
        self.states[:trace.sensing.shape[0], :, 3:] = trace.sensing
        self.targets[:trace.control.shape[0]] = trace.control
        self.errors[:trace.error.shape[0]] = trace.error
        comms = trace.communication

        return self.states, self.targets, self.errors, comms
