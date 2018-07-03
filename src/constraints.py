import numpy as np
import toolbox


class Constraints:
    """Constraints loaded from a file."""

    def __init__(self, fname):
        """
        Construct a Constraint object from a constraints file
        ASSUME THAT FORM TAKES g(x) >= 0.0...NOT >= 0

        :param fname: Name of the file to read the Constraint from (string)
        """
        with open(fname, "r") as f:
            lines = f.readlines()
        # Parse the dimension from the first line
        self.n_dim = int(lines[0])
        # Parse the example from the second line
        self.example = [float(x) for x in lines[1].split(" ")[0:self.n_dim]]

        # Run through the rest of the lines and compile the constraints
        self.inequalities = []
        self.exprs = []

        for i in range(2, len(lines)):
            # support comments in the first line
            if lines[i][0] == "#":
                continue
            self.inequalities.append(compile(lines[i], "<string>", "eval"))
            self.exprs.append(compile(lines[i][:-7], "<string>", "eval"))
        return

    def get_example(self):
        """Get the example feasible vector"""
        return self.example

    def get_ndim(self):
        """Get the dimension of the space on which the constraints are defined"""

        return self.n_dim

    def apply(self, x):
        """
        Apply the constraints to a vector, returning True only if all are satisfied

        :param x: list or array on which to evaluate the constraints
        """
        for ineq in self.inequalities:
            if not eval(ineq):
                return False
        return True

    def evaluate(self, x):
        """
        Evaluate constraints and return vector of "error" from zero

        :param x:  array (vector) of point to check
        :return: vector of "error"
        """
        error_vec = np.empty(len(self.exprs), dtype="float32")
        ind = 0
        for expr in self.exprs:
            eval_val = eval(expr)
            error_vec[ind] = eval_val
            ind += 1

        return error_vec

    def calc_error_distance(self, x):
        """
        Linearizes the constraints and finds the delta x required to satisfy each constraint
        And finds the probable distance x has to move to satisfy all constraints

        :param x: array (vector) of point to check
        :return: probable distance x has to move
        """
        dx = np.zeros(np.size(x), dtype="float32")
        err_vec = self.evaluate(x)

        # heuristically combines the vector to satisfy each constraint into a final delta x vector
        for ind in range(len(self.exprs)):
            dx_ind = toolbox.calc_dx(self.exprs[ind], x, err_vec[ind])
            dx[abs(dx_ind) > abs(dx)] = dx_ind[abs(dx_ind) > abs(dx)]

        return toolbox.calc_norm(dx), dx
