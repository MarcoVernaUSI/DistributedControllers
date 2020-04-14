####################
# Simulator for task 1
####################
from typing import List

from RandomInit import RandomInit
from SimulationTask1 import SimulationTask1
from SimulationTask2 import SimulationTask2
from SimulationInit import SimulationInit
from UniformInit import UniformInit
from holonomic_agent import Agent
import random as rand



class Simulator:
    def __init__(self, timesteps: int, n: int, l: float, masvel: float, initMode: SimulationInit):
        self.timesteps: int = timesteps
        self.N: int = n
        self.L: float = l
        self.masVel: float = masvel
        self.initMode: SimulationInit = initMode
        self.dt = 0.1

    def defineL(self):
        return self.L

    def defineN(self):
        return self.N

    def setTask(self, n_agents, actualL, agents_list, timesteps, parameter):
        return SimulationTask1(n_agents, actualL, self.dt, self.masVel, agents_list, timesteps, parameter)

    def run(self, init=None, control=None, parameter=None, Linit=None):
        if init is not None:
            n_agents = len(init)
            actualL = Linit
        else:
            n_agents = self.defineN()
            actualL = self.defineL()
        timesteps = self.timesteps

        # Create list of agents
        if init is None:
            init = self.initMode.create(n_agents, actualL)

        agents_list = []
        for i in range(n_agents):
            agents_list.append(Agent(init[i], i))
        # li ordino in base alla posizione
        agents_list.sort(key=lambda x: x.getxy()[0], reverse=False)

        # initialize agents and simulation state
        for i in range(n_agents):
            agents_list[i].state, agents_list[i].vels = agents_list[i].observe(agents_list, actualL, i)

        sim = self.setTask(n_agents, actualL, agents_list, timesteps, parameter)
        sim.saveInitialState()

        if control is None:
            states, targets, errors, comms = sim.runWithExpert()
        else:
            net_controller = control.controller()
            states, targets, errors, comms = sim.runWithNet(net_controller)

        return states, targets, errors, comms


class SimulatorR(Simulator):
    def __init__(self, timesteps, N, mL, ML, masVel, initMode=RandomInit()):
        self.timesteps = timesteps
        self.N = N
        self.mL = mL
        self.ML = ML
        self.mas_vel = masVel
        self.dt = 0.1
        self.initMode = initMode

    def defineL(self):
        L = round(rand.random.uniform(self.mL, self.ML), 2)
        return L


class Simulator2(Simulator):
    def __init__(self, timesteps: int, n: int, l: float, masvel: float, initMode: SimulationInit = RandomInit()):
        super(Simulator2, self).__init__(timesteps, n, l, masvel, initMode)

    def setTask(self, n_agents, actualL, agents_list, timesteps, parameter):
        return SimulationTask2(n_agents, actualL, self.dt, self.masVel, agents_list, timesteps, parameter)


class SimulatorN(Simulator2):
    def __init__(self, timesteps, Ns, L, masVel, initMode=UniformInit()):
        self.timesteps = timesteps
        self.Ns = Ns
        self.L = L
        self.masVel = masVel
        self.dt = 0.1
        self.initMode = initMode

    def defineN(self):
        N = rand.choice(self.Ns)
        return N
