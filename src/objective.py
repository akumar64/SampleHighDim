import numpy as np
import toolbox

class Objective():

    def __init__(self, n_dim, n_results):
        """
        In charge of finding the next point with the greatest distance to the existing points
        :param n_dim:
        """

        self.n_dim = n_dim
        self.n_results = n_results

        self.sample_points = np.empty([n_results, n_dim], dtype="float32")
        self.num_points = 0

    def add_point(self, x):
        self.sample_points[self.num_points] = x
        self.num_points += 1


    def calc_average_distance(self, x):
        """
        calculate average distance to already found points

        :param x: new point
        :return: average distance
        """
        sum_distance = 0
        for ind in range(self.num_points):
            sum_distance += toolbox.calc_distance(x, self.sample_points[ind, :])
        return sum_distance/self.num_points
