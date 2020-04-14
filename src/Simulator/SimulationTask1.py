import numpy as np

from Simulation import Simulation
from SimulatorRun import save_state
from PID import PID
from RunTask1 import RunTask1


class SimulationTask1(Simulation):
    def initializeTargets(self):
        goal_list = [None] * self.nAgents

        agent_width = self.agentsList[0].width
        free_space = self.actualL - (self.nAgents * agent_width * 2)
        # assert free_space > 0
        space_dist = free_space / (self.nAgents + 1)
        # first agent
        goal_list[0] = space_dist + agent_width
        if self.nAgents > 2:
            for i in range(1, self.nAgents - 1):
                goal_list[i] = space_dist + agent_width + (2 * agent_width + space_dist) * i
        if self.nAgents > 1:
            goal_list[self.nAgents - 1] = self.actualL - (agent_width + space_dist)

        return goal_list

    def runWithExpert(self):
        controller = PID(-5, 0, 0)  # (-5, 0, 0)
        comms = None
        for t in range(0, self.timesteps):
            for i in range(0, len(self.agentsList)):
                self.agentsList[i].state, self.agentsList[i].vels = self.agentsList[i].observe(self.agentsList,
                                                                                               self.actualL, i)
                error = self.agentsList[i].getxy()[0] - self.goalList[i]
                v = controller.step(error, self.dt)
                v = np.clip(v, -self.masVel, +self.masVel)
                self.targets[t, i] = v
            # save state

            self.states[t] = save_state(self.agentsList, self.stateSize)
            state_error = self.states[t, :, 0]
            self.errors[t] = np.mean(abs(state_error - np.array(self.goalList)))

            # update agents
            for i, agent in enumerate(self.agentsList):
                old_position = agent.getxy()[0]
                agent.step(self.targets[t, i], self.dt)

                # check collisions
                if not agent.check_collisions(self.agentsList[:i] + self.agentsList[i + 1:], self.actualL):
                    agent.setx(old_position)
                    self.targets[t, i] = 0.0

        return self.states, self.targets, self.errors, comms

    def controllerRun(self, controller):
        return RunTask1(self.actualL, controller=controller, dt=0.1)
