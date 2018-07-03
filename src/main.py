import sys
import numpy as np
import sample_space as ss

CHANGE_TH = 0.00024  # 0.00024    # ending condition for change in ratio of distances of iterative next point
COST_RATIO = 0.5     # ratio of constraint error to objective distance for
EXP_FACTOR = 0.1  # 1.5
RATIONAL_EPSILON = 1e-5
OBJECTIVE_METHOD = 'covariance'  # 'min_distance' 'avg_distance'

if __name__ == '__main__':
    constraint_optimization = ss.ConstraintOptimization(sys.argv[1], int(sys.argv[3]))
    sample_points = constraint_optimization.sample_points()

    out_filename = sys.argv[2]
    np.savetxt(out_filename, sample_points, delimiter=' ')
