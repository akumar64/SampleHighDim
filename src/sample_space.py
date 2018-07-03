import sys
import numpy as np
import random
import heapq
import math
import toolbox
from constraints import Constraints
from objective import Objective
import main as parameters


class ConstraintOptimization:
    """
    Main class that samples the high dimensional space

    Optimizes the distance between the points sample,
    While satisfying the constraints
    """

    def __init__(self, infilename, n_results):
        self.n_results = n_results

        self.constraints = Constraints(infilename)
        self.n_dim = self.constraints.get_ndim()
        self.example = self.constraints.example

    def sample_points(self):
        """
        Main function, calls find_new_vector method for the required number of times
        """
        objective = Objective(self.n_dim, self.n_results)
        objective.add_point(np.array(self.example))

        addI = 0
        while addI < self.n_results-1:
            if addI % 10 == 0:
                print addI
            new_vector, count_fail, best_val = self.find_new_vector(objective)
            if new_vector is not None:
                addI += 1
                objective.add_point(new_vector)

                # adds to total values for final reporting
                # sets min and max objective values for subsequent normalization
                objective.total_obj_distance += best_val
                objective.add_count_fail(count_fail)
                objective.set_max_objective(best_val)
                objective.set_min_objective(parameters.OBJECTIVE_METHOD)
            else:
                print "New vector not found...Consider parameter change if problem exists"

        print
        print "Total Count Fail ", objective.total_count_fail
        print "Total Objective Distance ", objective.total_obj_distance
        return objective.sample_points

    def find_new_vector(self, objective):
        """
        Uses branch and bound algorithm to find next vector that is
        far from previous vectors but also obeys all constraints.

        Uses priority queue/stack to search greedily for areas that minimizes
        the constraint error and maximizes the distance
        :param objective: object that contains the previous points
        :return: new vector
        """
        best_branch = Branch(self.n_dim)
        best_branch.set_score_explore(objective, self.constraints)

        # starting of with a stack, then a heap
        branch_list = [best_branch]
        heap = []
        prev_vector = None

        # keeping track of no. of branches visited using stack (False) and heap (True)
        best_obj_distance = count_fail = 0
        total_count_fail = {False: 0, True: 0}

        # keeping track of branches that most likely do not lead to the right solution
        # err_ancestor_info helps deciding whether branches should be made an error ancestor (Details in set_as_err_ancestor_info)
        inefficient_ancestors = set()
        err_ancestor_info = {'len': 0, 'prev': 0, 'num': 0}

        # search ends till the new vector is not that far off from the previously found vector
        change_dist_th = math.sqrt(parameters.CHANGE_TH**2*self.n_dim)
        change_dist = sys.float_info.max
        while change_dist > change_dist_th and len(branch_list) > 0:
            if best_obj_distance > 0:
                best_branch = heapq.heappop(branch_list)[1]
            else:
                best_branch = branch_list.pop()

            vector = best_branch.get_center()
            count_fail += 1

            # do not check if likely not going to find a solution
            do_check = best_branch.ancestor not in inefficient_ancestors

            if do_check:
                if best_branch.is_pass:
                    if best_branch.obj_distance > best_obj_distance:
                        if prev_vector is not None:
                            change_dist = toolbox.calc_distance(vector, prev_vector)
                        prev_vector = vector
                        total_count_fail[best_obj_distance > 0] += count_fail
                        count_fail = 0

                        # first solution found, switch to heap
                        if best_obj_distance == 0:
                            objective.set_max_objective(best_branch.obj_distance)
                            branch_list = heap
                        best_obj_distance = best_branch.obj_distance

                # if solution still not found, consider setting branch as a bad error ancestor
                if best_obj_distance == 0:
                    set_as_err_ancestor(best_branch, err_ancestor_info)

                child_branches, is_smallest = best_branch.split_longest()

                # two different scoring methods for new branches depending on whether solution is found
                for branch in child_branches:
                    if best_obj_distance > 0:
                        branch.set_score_diff(objective, self.constraints)
                        branch.set_ancestor_pass()
                    else:
                        branch.set_score_explore(objective, self.constraints)

                add_branches_to_list(branch_list, child_branches, best_obj_distance, is_smallest)

                # if branch is small and is not a viable solution, add to set
                if is_smallest and not count_fail == 0:
                    if best_branch.ancestor is not None:
                        add_inefficient_branches(inefficient_ancestors, best_branch.ancestor, err_ancestor_info)

        print "obj distance ", best_obj_distance
        print prev_vector
        total_count_fail[best_obj_distance > 0] += count_fail

        return prev_vector, total_count_fail, best_obj_distance


def set_as_err_ancestor(best_branch, err_ancestor_info):
    """
    Wrapper method to decide whether branch should be made an ancestor,
    which allows for quickly backtracking to an alternate path if solution still can't be found
    :param best_branch: branch to be checked
    :param err_ancestor_info:   len  - no. of bad branches in the set;
                                prev - previous no. of ancestors in the set before needing to backtrack
                                num  - current number of ancestors
    """
    is_added = best_branch.set_ancestor_err(err_ancestor_info['len'], err_ancestor_info['prev'])
    err_ancestor_info['num'] += is_added


def add_branches_to_list(branch_list, child_branches, best_obj_distance, is_empty):
    """
    Adding new branches to stack or heap, based on whether first solution is found
    :param branch_list: stack or heap of branches
    :param child_branches: new branches
    :param best_obj_distance:
    :param is_empty: if branches too small to add to the data structure
    """
    if not is_empty:
        if best_obj_distance > 0:
            for branch in child_branches:
                heapq.heappush(branch_list, (-branch.score, branch))
        else:
            # add new branches to stack in the order of lowest to highest, to deque the highest first
            order_bool = child_branches[0].score > child_branches[1].score
            branch_list.append(child_branches[0 ^ order_bool])
            branch_list.append(child_branches[1 ^ order_bool])


def add_inefficient_branches(inefficient_ancestors, ancestor, err_ancestor_info):
    """
    Adding bad branches to set of inefficient branches. Re-sets length of error ancestors. Re-sets previous value
    of length of inefficient error ancestors if all previously recorded ancestors are added to the set
    :param inefficient_ancestors: set of bad ancestors
    :param ancestor: bad ancestor to be added
    :param err_ancestor_info: book keeping for error ancestors
    """
    inefficient_ancestors.add(ancestor)
    err_ancestor_info['len'] = len(inefficient_ancestors)
    if abs(len(inefficient_ancestors) - err_ancestor_info['num']) < 0.1:
        err_ancestor_info['prev'] = len(inefficient_ancestors)


class Branch:
    """
    Represents a section of the hypercube as its domain and its potential in the constraint optimization

    To allow for approximation/pruning/bounding/backgracking, storing the oldest parent that is of the same isPass,
    or that doesn't try to go to legal space through the variable ancestor, which is set in the set_score method
    """
    def __init__(self, n_dim, old_branch=None):
        self.n_dim = n_dim
        self.ancestor = None

        if old_branch is None:
            self.min = np.zeros(n_dim, dtype="float32")
            self.max = np.ones(n_dim, dtype="float32")

            self.parent = None
        else:
            self.ancestor = old_branch.ancestor
            if self.ancestor is None:
                self.ancestor = old_branch

            self.min = np.copy(old_branch.min)
            self.max = np.copy(old_branch.max)

            self.parent = old_branch

    def set_err_obj_distances(self, objective, constraints):
        """
        Sets objective value and error distance and whether branch passed all constraints
        :param objective:
        :param constraints:
        """
        center = self.get_center()

        is_pass = constraints.apply(center)
        err_distance, dx = constraints.calc_error_distance(center)

        obj_distance = objective.min_objective
        if parameters.OBJECTIVE_METHOD is 'avg_distance':
            obj_distance = objective.calc_average_distance(center)
        elif parameters.OBJECTIVE_METHOD is 'min_distance':
            obj_distance = objective.calc_min_distance(center)
        elif parameters.OBJECTIVE_METHOD is 'covariance':
            obj_distance = objective.calc_covariance(center)

        self.center = center
        self.is_pass = is_pass
        self.obj_distance = obj_distance
        self.err_distance = err_distance
        self.dx = dx

    def set_score_diff(self, objective, constraints):
        """
        Sets score as difference between objective and error distance
        """
        self.set_err_obj_distances(objective, constraints)

        # score = obj_distance-COST_RATIO*err_distance    # try without normalization
        # score = self.obj_distance/objective.prev_min_distance - COST_RATIO * self.err_distance / movable_distance
        movable_distance = toolbox.calc_distance(self.max, self.center)
        normalized_objective = (self.obj_distance-objective.min_objective)/(objective.max_objective-objective.min_objective)
        score = normalized_objective - parameters.COST_RATIO*self.err_distance/movable_distance   # with normalization

        self.score = score

    def set_score_explore(self, objective, constraints):
        """
        Sets score for branches that are not in legal space.
        :param objective:
        :param constraints:
        :return:
        """
        self.set_err_obj_distances(objective, constraints)

        # normalize vector by which branch has to move to legal space by the vector of its size
        norm_err_dist = toolbox.calc_norm(np.abs(self.dx)/(self.max-self.min))

        exp_den = math.sqrt(0.25**2*self.n_dim)
        score = self.obj_distance/math.exp(norm_err_dist/parameters.EXP_FACTOR/exp_den)
        self.score = score

    def set_score_land(self, objective, constraints):
        """
        Sets score for branch using an inverse type of function
        :param objective:
        :param constraints:
        :return:
        """
        self.set_err_obj_distances(objective, constraints)

        score = self.obj_distance/(parameters.RATIONAL_EPSILON+self.err_distance)
        self.score = score

        self.set_ancestor_pass()

    def set_ancestor_pass(self):
        """
        Set as ancestor if whether it's in legal space is different to its parent ancestor
        """
        if self.ancestor is not None:
            if self.is_pass is not self.ancestor.is_pass:
                self.ancestor = None

    def set_ancestor_err(self, num_inefficient_ancestors, prev_inefficient):
        """
        As we are trying to move to legal space, branch is set as ancestor if it is farther away than its parent
        BUT stop setting branches as ancestors if new inefficient ancestor is found - which means it is time to backtrack
        :param num_inefficient_ancestors: current number of inefficient ancestors
        :param prev_inefficient: number of inefficient ancestors before descending to find a solution
        :return:
        """
        if num_inefficient_ancestors-prev_inefficient < 0.1:
            if self.ancestor is not None:
                if not self.is_pass and self.err_distance > self.parent.err_distance:
                    self.ancestor = None
                    return True
        return False

    def split_dimension(self, split_dim):
        """
        Splits domain by given dimension and uses the center to do so
        :param split_dim: dimension at which domain is split
        :return: new branches
        """
        dim_middle = 0.5*(self.min[split_dim]+self.max[split_dim])

        branch1 = Branch(self.n_dim, self)
        branch2 = Branch(self.n_dim, self)

        branch1.max[split_dim] = dim_middle
        branch2.min[split_dim] = dim_middle

        return [branch1, branch2]

    def split_all(self):
        """
        Splits domain in all dimensions and uses the center to do so
        :return: new branches
        """
        branch_size = self.max-self.min

        if toolbox.calc_norm(branch_size) < math.sqrt(parameters.CHANGE_TH**2*self.n_dim):
            return [], True
        else:
            branches = [None]*(self.n_dim*2)

            for dim_i in range(self.n_dim):
                branches[dim_i*2:(dim_i+1)*2] = self.split_dimension(dim_i)

            return branches, False

    def split_longest(self):
        """
        Splits domain by the longest dimension, but picks among the longest dimensions randomly
        :return: new branches and True is reached small size threshold
        """
        branch_size = self.max-self.min

        # if score is approaching very low, way to end search quicker
        if self.score < 1e-100:
            return [], True

        if toolbox.calc_norm(branch_size) < math.sqrt(parameters.CHANGE_TH**2*self.n_dim):
            return [], True
        else:
            dim_max = np.max(branch_size)
            dims = np.arange(self.n_dim)
            longest_dims = dims[branch_size > 0.99*dim_max]
            longest_dim = random.choice(longest_dims)

            return self.split_dimension(longest_dim), False

    def get_center(self):
        return 0.5*(self.min+self.max)
