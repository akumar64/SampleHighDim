from constraints import Constraints


class ConstraintOptimization():
    """
    Main class that samples the high dimenstional space

    Need to optimize the distance between the points sample,
    While satisfying the constraints
    """

    def __init__(self, infilename, outfilename, n_results):
        self.constraints = Constraints(infilename)
