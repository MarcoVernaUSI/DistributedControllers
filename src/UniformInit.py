from SimulationInit import SimulationInit


class UniformInit(SimulationInit):

    def calcolatePositions(self):

        # create initial positions
        distance_between_agents= (self.L - ((self.agentWidth*2)*self.N))/ (self.N+1)

        initial_positions = []
        j = 1
        for i in range(self.N):
            x = i+1
            initial_positions.append(x*distance_between_agents + j*self.agentWidth)
            j = j+2

        return initial_positions
