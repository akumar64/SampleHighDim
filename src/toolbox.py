from scipy.spatial import distance


def calc_distance(x, y):
    return distance.euclidean(x, y)
