import numpy as np

from Simulation import Simulation
from PID import PID


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

            self.states[t] = self.save_state(self.agentsList, self.stateSize)
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

    def runStep(self, control, goalList, agents_list, state, mas_vel)  -> float:
        e = np.mean(abs(np.array(state) - np.array(goalList)))
        # update

        for i,v in enumerate(control):
            v = np.clip(v, -mas_vel, +mas_vel)
            agents_list[i].step(v,self.dt)
            control[i]=v

            #check collisions
            if not agents_list[i].check_collisions(agents_list[:i]+agents_list[i+1:],self.actualL):
                agents_list[i].setx(state[i])
                agents_list[i].velocity= 0.0
                control[i]=0.0
        return e
