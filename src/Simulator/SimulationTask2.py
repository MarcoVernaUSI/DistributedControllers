from Simulation import Simulation
from SimulatorRun import save_state
from RunTask2 import RunTask2


class SimulationTask2(Simulation):
    def initializeTargets(self):
        goal_list = [None] * self.nAgents
        for i in range(self.nAgents // 2):
            goal_list[i] = 0
        i = i + 1
        for j in range(len(goal_list) - i):
            goal_list[i + j] = 1

        return goal_list

    def runWithExpert(self):
        comms = None
        for t in range(0, self.timesteps):
            for i in range(0, len(self.agentsList)):
                self.agentsList[i].state, self.agentsList[i].vels = self.agentsList[i].observe(self.agentsList,
                                                                                               self.actualL, i)
                self.agentsList[i].color = self.goalList[i]
                self.targets[t, i] = self.goalList[i]
            # save state

            self.states[t] = save_state(self.agentsList, self.stateSize)
            self.errors[t] = 0.0

        return self.states, self.targets, self.errors, comms

    def controllerRun(self, controller):
        return RunTask2(self.actualL, controller=controller, dt=0.1)