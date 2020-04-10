import numpy as np
import random as rand

from SimulationInit import SimulationInit


class RandomInit(SimulationInit):

    def calcolatePositions(self):

        # create initial positions
        offset = int(round((self.agentWidth + self.bufferDistance) * 100))
        range_ = int(round(self.L * 100)) - 2 * offset  # lo trasformo in centimetri

        initial_positions = np.array(self.sample_with_minimum_distance(range_, self.N, offset * 2))
        initial_positions = initial_positions + offset  # sommo a tutti l'offset
        # divido per cento per tornare a metri
        return initial_positions / 100

    # Function to generate the random samples
    def ranks(self, sample):
        indices = sorted(range(len(sample)), key=lambda i: sample[i])
        return sorted(indices, key=lambda i: indices[i])

    def sample_with_minimum_distance(self, n=100, k=5, d=13):
        sample = rand.sample(range(n - (k - 1) * (d - 1)), k)
        return [s + (d - 1) * r for s, r in zip(sample, self.ranks(sample))]

