import numpy as np
import sys
import time
import math


def read_file(filename):
    with open(filename) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        n, m, r = list(map(int, content[0].split(' ')))
        vector = np.array(list(map(int, list(content[1]))), dtype=bool)
        matrix = vector.reshape(n, m)
        return n, m, r, matrix


class BoolFactorization():
    def __init__(self):
        self.n, self.m, self.r, self.matrix = read_file(sys.argv[1])
        self.A, self.B = None, None
        self.initialize_matrices()
        self.number_of_cells_A = self.n * self.r
        self.number_of_cells = self.number_of_cells_A + self.r * self.m
        self.matrix_samples = self.random_matrix_samples()
        self.prob_samples = self.random_prob_samples()
        self.start_time = time.time()
        self.simulated_annealing()

    def initialize_matrices(self):
        self.A = np.zeros((self.n, self.r), dtype=bool)
        self.B = np.zeros((self.r, self.m), dtype=bool)

    def temperature(self):
        """
        Drops linearly in 10 seconds from 10 to 0.
        """
        return 10 - (time.time() - self.start_time)

    def random_matrix_samples(self):
        return np.random.random_integers(0, self.number_of_cells - 1, 1000000)

    @staticmethod
    def random_prob_samples():
        return np.random.uniform(0, 1, 1000000)

    def sample2matrix_position(self, sample):
        """
        Returns index of row and index of column in first (True) or second (False) matrix.
        :param sample: number between 0 and self.number_of_cells - 1
        """
        if sample < self.number_of_cells_A:
            return True, sample // self.r, sample % self.r
        else:
            sample -= self.number_of_cells_A
            return False, sample // self.m, sample % self.m

    def cost(self, my, real):
        return np.sum(np.logical_xor(my, real))

    def print_matrices(self):
        a = ''.join(list(map(lambda x: '1' if x else '0', self.A.reshape(self.n * self.r))))
        b = ''.join(list(map(lambda x: '1' if x else '0', self.B.reshape(self.r * self.m))))
        print(a)
        print(b)
        print()

    def simulated_annealing(self):
        iteration = 0
        while True:
            # get temperature according to time
            temp = self.temperature()
            if temp <= 0:
                # print('Temp below 0!')
                # print('Iterations: ', iteration)
                # print('Final error: ', round(self.cost(self.matrix, self.A.dot(self.B)) / (self.n * self.m), 4))
                break

            # randomly select cell in one matrix and change it
            first, row, column = self.sample2matrix_position(self.matrix_samples[iteration])
            if first:
                previous_cost = self.cost(self.A[row, :].dot(self.B), self.matrix[row, :])
                self.A[row, column] = not self.A[row, column]
                cost = self.cost(self.A[row, :].dot(self.B), self.matrix[row, :])
            else:
                previous_cost = self.cost(self.A.dot(self.B[:, column]), self.matrix[:, column])
                self.B[row, column] = not self.B[row, column]
                cost = self.cost(self.A.dot(self.B[:, column]), self.matrix[:, column])

            # calculate delta cost
            delta = previous_cost - cost

            # if cost is better than change the state, else change it with probability proportional to temperature
            if delta > 0:
                if temp < 2:
                    self.print_matrices()
            else:
                # random between 0 and 1
                r = self.prob_samples[iteration]
                # print('delta: ', delta, ' temp: ', temp)

                boundary = 0.1 * math.exp(delta / temp)
                # print('boundary: ', boundary)

                if r > boundary:      # change the state back
                    if first:
                        self.A[row, column] = not self.A[row, column]
                    else:
                        self.B[row, column] = not self.B[row, column]

            # print('iteration: ', iteration, ' cost: ', cost)
            # print()

            iteration += 1



if __name__ == '__main__':
    bf = BoolFactorization()

