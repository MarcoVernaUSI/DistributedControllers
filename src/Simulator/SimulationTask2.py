import numpy as np
from SimulatorRun import save_state
from RunTask2 import RunTask2


class SimulationTask2():
    def __init__(self, nAgents, actualL, dt, masVel, agentsList, timesteps, parameter):
        self.nAgents: int = nAgents
        self.actualL: float= actualL
        self.dt = dt
        self.masVel = masVel
        self.agentsList = agentsList
        self.timesteps = timesteps
        self.states, self.targets, self.errors, self.stateSize = self.initializeParameters(parameter)
        self.goalList = self.initializeTargets()

    def initializeTargets(self):
        goal_list = [None]*self.nAgents
        for i in range(self.nAgents//2):
            goal_list[i]=0
        i=i+1
        for j in range(len(goal_list)-i):
            goal_list[i+j]=1

        return goal_list

    def initializeParameters(self, parameter):
        if parameter == 'add_vel':
            state_size = len(self.agentsList[0].state)+len(self.agentsList[0].vels)+3
        else:
            state_size = len(self.agentsList[0].state)+3 # x, y, theta + stato
        states = np.zeros((self.timesteps,self.nAgents,state_size))
        targets = np.zeros((self.timesteps, self.nAgents))
        errors = np.zeros((self.timesteps,), dtype=np.float32)

        return states, targets, errors, state_size

    def saveInitialState(self):
        self.states[0] = save_state(self.agentsList, self.stateSize)

    def runWithExpert(self):
        comms = None
        for t in range (0, self.timesteps):
            for i in range( 0, len(self.agentsList)):
                self.agentsList[i].state,  self.agentsList[i].vels= self.agentsList[i].observe(self.agentsList,self.actualL, i)
                self.agentsList[i].color = self.goalList[i]
                self.targets[t,i] = self.goalList[i]
            # save state

            self.states[t] = save_state(self.agentsList, self.stateSize)
            self.errors[t] = 0.0

        return self.states, self.targets, self.errors, comms

    def runWithNet(self, controller, masVel):
        net_run = RunTask2(self.actualL, controller=controller, dt=0.1)
        trace = net_run(self.states[0], self.agentsList, self.goalList, masVel,
                            self.stateSize, epsilon=0.01, T=self.timesteps)
        self.states[:trace.state.shape[0],:,0]=trace.state
        self.states[:trace.sensing.shape[0],:,3:]=trace.sensing
        self.targets[:trace.control.shape[0]]=trace.control
        self.errors[:trace.error.shape[0]]=trace.error
        comms= trace.communication

        return self.states, self.targets, self.errors, comms