#############################################
# IMPORTANT: DO NOT MODIFY THIS FILE
# The autograder uses the original version of this file
# Any change you make will NOT be used in the autograder
#############################################

import numpy as np

LABELED_FILE = "surveylabeled.dat"
LABELED_FILE_0 = "surveylabeled_0.dat"
LABELED_FILE_1 = "surveylabeled_1.dat"

UNLABELED_FILE = "surveyunlabeled.dat"

#===============================================================================
# General helper functions

def log_sum_exp(A, B):
    """
    Compute log(exp(A) + exp(B)) in a numerical stable way.
    """
    M = np.maximum(A, B)
    return M + np.log(np.exp(A-M) + np.exp(B-M))


def log_sum_exp_np(a, axis):
    """
    Compute log sum exp for a numpy array on a specified set of axis
    """
    M = np.amax(a, axis=axis, keepdims=True)
    return M + np.log(np.sum(np.exp(a - M), axis=axis, keepdims=True))


def get_random_psd(n):
    x = np.random.normal(0, 1, size=(n, n))
    return np.dot(x, x.transpose())


def get_random_params(seed=None):
    """
    Get some randomly initialized parameters
    If seed is None, do not set the random seed.
    """
    if seed is not None:
        rng = np.random.get_state()
        np.random.seed(seed)
    params = {'phi': np.random.uniform(0, 1),
              'lambda': np.random.uniform(0, 1),
              'pi': np.random.uniform(0, 1),
              'mu0': np.random.normal(0, 1, size=(2,)),
              'mu1': np.random.normal(0, 1, size=(2,)),
              'sigma0': get_random_psd(2),
              'sigma1': get_random_psd(2)}
    if seed is not None:  # Reset random state
        np.random.set_state(rng)
    return params


def colorprint(message, color="rand"):
    """Prints your message in pretty colors!

    So far, only the colors below are available.
    """
    if color == 'none': print(message); return
    if color == 'demo':
        for i in range(99):
            print('%i-'%i + '\033[%sm'%i + message + '\033[0m\t',)
    print('\033[%sm'%{
        'neutral' : 99,
        'flashing' : 5,
        'underline' : 4,
        'magenta_highlight' : 45,
        'red_highlight' : 41,
        'pink' : 35,
        'yellow' : 93,
        'teal' : 96,
        'rand' : np.random.randint(1,99),
        'green?' : 92,
        'red' : 91,
        'bold' : 1
    }.get(color, 1)  + message + '\033[0m')


def read_labeled_matrix(path = LABELED_FILE):
    """Read and parse the labeled dataset.
    The first two dimensions of X and Z are (N, M): Counts of precincts and voters.

    Output:
        X: A numpy array of size (N, M, 2), where X[i, j] is the 2-dimensional vector
            representing the voter's properties
        Z: A numpy array of size (N, M), where Z[i, j] = 0 or 1 indicating the party preference

    """
    Zij = {}
    Xij = {}
    M = 0.0
    N = 0.0
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            i, j, Z, X1, X2 = line.split()
            i, j = int(i), int(j)
            if i > N:
                N = i
            if j > M:
                M = j
            Zij[i - 1, j - 1] = float(Z)
            Xij[i - 1, j - 1] = np.matrix([float(X1), float(X2)])

    Z = np.zeros(shape=(N, M))
    X = np.zeros(shape=(N, M, 2))
    for i in range(N):
        for j in range(M):
            Z[i, j] = Zij[(i, j)]
            X[i, j] = Xij[(i, j)]
    return X, Z


def read_unlabeled_matrix(path = UNLABELED_FILE):
    """Read and parse the unlabeled dataset.

    Output:
        X: A numpy array of size (N, M, 2), where X[i, j] is the 2-dimensional vector
            representing the voter's properties, N,M are counts of precincts and voters respectively
    """
    Xij = {}
    M = 0.0
    N = 0.0
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            i, j, X1, X2 = line.split()
            i, j = int(i), int(j)
            if i > N: N = i
            if j > M: M = j

            Xij[i - 1, j - 1] = np.matrix([float(X1), float(X2)])
    X = np.zeros(shape=(N, M, 2))
    for i in range(N):
        for j in range(M):
            X[i, j] = Xij[(i, j)]
    return X


