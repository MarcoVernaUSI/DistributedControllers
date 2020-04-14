import numpy as np
from abc import abstractmethod, ABCMeta
from dataset import Trace
from typing import Callable, List
from network import Controller

SimulationRun = Callable[[Controller], Trace]

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
        self.states[0] = self.save_state(self.agentsList, self.stateSize)

    @abstractmethod
    def runWithExpert(self):
        pass

    def runWithNet(self, controller):
        net_run = self.run()
        trace = net_run(controller)

        self.states[:trace.state.shape[0], :, 0] = trace.state
        self.states[:trace.sensing.shape[0], :, 3:] = trace.sensing
        self.targets[:trace.control.shape[0]] = trace.control
        self.errors[:trace.error.shape[0]] = trace.error
        comms = trace.communication

        return self.states, self.targets, self.errors, comms

    def run(self) -> SimulationRun:
        def f(controller: Controller) -> Trace:
            state_c = self.states[0]
            agents_list =  self.agentsList
            goalList = self.goalList
            mas_vel = self.masVel
            state_size = self.stateSize
            T = self.timesteps

            t = 0.0
            steps: List[Trace] = []

            while (t < T) or t == 0:
                state = state_c[:,0].tolist()
                sensing = state_c[:,3:].tolist()

                control, *communication = controller(state, sensing)
                if communication:
                    communication = communication[0]

                e = self.runStep(control, goalList, agents_list, state, mas_vel)

                for i in range( 0, len(agents_list)):
                    agents_list[i].state, agents_list[i].vels= agents_list[i].observe(agents_list,self.actualL, i)

                steps.append(Trace(t, state, communication, sensing, control, e))
                state_c = self.save_state(agents_list,state_size)
                t += 1
            return Trace(*[np.array(x) for x in zip(*steps)])
        return f

    @abstractmethod
    def runStep(self, control, goalList, agents_list, state, mas_vel)  -> float:
       pass


    def save_state(self, agents_list, state_size):

        n_agents=len(agents_list)
        state = np.zeros((n_agents, state_size))
        for i in range(n_agents):
            state[i,0]= agents_list[i].getxy()[0]
            state[i,1]= agents_list[i].getxy()[1]
            state[i,2]= agents_list[i].gettheta()
            for j in range(state_size-3):
                state[i,3+j]= np.concatenate((agents_list[i].state,agents_list[i].vels))[j]
        return state

