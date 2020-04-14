from SimulationInit import SimulationInit


class UniformInit(SimulationInit):

    def calcolatePositions(self, N: int, L: float, agentWidth: float, bufferDistance: float):

        # create initial positions
        distance_between_agents= (L - ((agentWidth*2)*N))/ (N+1)

        initial_positions = []
        j = 1
        for i in range(N):
            x = i+1
            initial_positions.append(x*distance_between_agents + j*agentWidth)
            j = j+2

        return initial_positions
