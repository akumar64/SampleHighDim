import numpy as np
import random

class Branch():
    """
    Contains a section of the hypercube as its domain
    """

    def __init__(self, n_dim, old_branch=None):
        self.n_dim = n_dim

        if old_branch is None:
            self.min = np.zeros([n_dim], dtype="float32")
            self.max = np.ones([n_dim], dtype="float32")
        else:
            self.min = old_branch.min
            self.max = old_branch.max

    def split(self):
        """
        Splits domain using a random dimension and uses the center to do so
        :return: new branches
        """

        split_dim = random.randint(0, self.n_dim-1)
        dim_middle = 0.5*(self.min[split_dim]+self.max[split_dim])

        branch1 = Branch(self.n_dim, self)
        branch2 = Branch(self.n_dim, self)

        branch1.max[split_dim] = dim_middle
        branch2.min[split_dim] = dim_middle

        return [branch1, branch2]

    def get_center(self):
        return 0.5*(self.min+self.max)
