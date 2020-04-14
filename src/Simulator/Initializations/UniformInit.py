from SimulationInit import SimulationInit


class UniformInit(SimulationInit):

    def calcolatePositions(self, n: int, l: float, agentwidth: float, bufferdistance: float):

        # create initial positions
        distance_between_agents= (l - ((agentwidth * 2) * n)) / (n + 1)

        initial_positions = []
        j = 1
        for i in range(n):
            x = i+1
            initial_positions.append(x * distance_between_agents + j * agentwidth)
            j = j+2

        return initial_positions
