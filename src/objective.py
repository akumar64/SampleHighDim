import numpy as np
import toolbox

class Objective:

    def __init__(self, n_dim, n_results):
        """
        In charge of finding the next point with the greatest distance to the existing points
        :param n_dim:
        """
        self.n_dim = n_dim
        self.n_results = n_results

        # self.max_distance = toolbox.calc_norm(np.ones(n_dim, dtype="float32"))
        self.prev_min_distance = toolbox.calc_norm(np.ones(n_dim, dtype="float32"))
        self.max_objective = 0  # toolbox.calc_norm(np.ones(n_dim, dtype="float32"))
        self.min_objective = 0

        self.sample_mean = np.zeros([1, n_dim], dtype="float32")
        self.sample_sqr_mean = np.zeros([n_dim, n_dim], dtype="float32")

        self.sample_points = np.empty([n_results, n_dim], dtype="float32")
        self.num_points = 0

        self.total_obj_distance = 0
        self.total_count_fail = {False: 0, True: 0}

    def add_point(self, x):
        self.set_max_distance(x)

        self.sample_mean = (self.num_points*self.sample_mean+x)/(self.num_points+1)
        self.sample_sqr_mean = (self.num_points*self.sample_sqr_mean+x.reshape([self.n_dim, 1])*x)/(self.num_points+1)

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

    def calc_min_distance(self, x):
        """
        calculate distance to closest existing point
        :param x: new point
        :return: distance to closest existing point
        """
        min_distance = toolbox.calc_norm(np.ones(self.n_dim, dtype="float32"))
        for ind in range(self.num_points):
            dist = toolbox.calc_distance(x, self.sample_points[ind, :])
            if dist < min_distance:
                min_distance = dist
        return min_distance

    def set_max_distance(self, x):
        """
        Setting maximum distance between the all the points (before adding the new x)
        :param x: New x
        """
        if abs(self.num_points-1) < 1e-3:
            self.max_distance = toolbox.calc_distance(x, self.sample_points[0, :])
        elif self.num_points > 1.5:
            for ind in range(self.num_points):
                new_distance = toolbox.calc_distance(x, self.sample_points[ind, :])
                if new_distance > self.max_distance:
                    self.max_distance = new_distance

    def calc_covariance(self, x):
        """
        Calculates Fro norm of new covariance  E[XX^T] - E[X]E[X]^T
        :param x: New x
        :return: covariance
        """
        new_mean = (self.num_points*self.sample_mean+x)/(self.num_points+1)
        sqr_x = x.reshape([self.n_dim, 1])*x
        new_sqr_mean = (self.num_points*self.sample_sqr_mean+sqr_x)/(self.num_points+1)

        covariance = new_sqr_mean-np.transpose(new_mean)*new_mean
        return toolbox.calc_norm(covariance)

    def set_prev_min_distance(self, prev_min_distance):
        self.prev_min_distance = prev_min_distance

    def set_max_objective(self, objective_val):
        if objective_val > self.max_objective:
            self.max_objective = objective_val

    def set_min_objective(self, method):
        if method is 'covariance':
            self.min_objective = self.calc_covariance(self.sample_mean)
        else:
            self.min_objective = 0

    def add_count_fail(self, count_fail):
        self.total_count_fail[True] += count_fail[True]
        self.total_count_fail[False] += count_fail[False]
