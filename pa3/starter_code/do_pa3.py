###############################################################################
# Finishes PA 3
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018


## Edited by Zhangyuan Wang, 01/2019
## Edired by Akshat Jindal, 01/2020
###############################################################################

## Utility code for PA3
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools
from factor_graph import *
from factors import *
import code


def loadLDPC(name):
    """
    :param - name: the name of the file containing LDPC matrices

    return values:
    G: generator matrix
    H: parity check matrix

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    """
    A = sio.loadmat(name)
    G = A['G']
    H = A['H']
    return G, H

def loadImage(fname, iname):
    '''
    :param - fname: the file name containing the image
    :param - iname: the name of the image
    (We will provide the code using this function, so you don't need to worry too much about it)

    return: image data in matrix form
    '''
    img = sio.loadmat(fname)
    return img[iname]

def applyChannelNoise(y, epsilon):
    '''
    :param y - codeword with 2N entries
    :param epsilon - the probability that each bit is flipped to its complement

    return corrupt message yTilde
    yTilde_i is obtained by flipping y_i with probability epsilon

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    ###############################################################################
    # TODO: Your code here!

    raise NotImplementedError()
    ###############################################################################
    assert y.shape == yTilde.shape
    return yTilde

def encodeMessage(x, G):
    '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword y=Gx mod 2

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    return np.mod(np.dot(G, x), 2)

def constructFactorGraph(yTilde, H, epsilon):
    '''
    Args
    - yTilde: np.array, shape [2N, 1], observed codeword containing 0's and 1's
    - H: np.array, shape [N, 2N], parity check matrix
    - epsilon: float, probability that each bit is flipped to its complement

    Returns: FactorGraph

    You should consider two kinds of factors:
    - M unary factors
    - N each parity check factors

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    N = H.shape[0]
    M = H.shape[1]
    G = FactorGraph(numVar=M, numFactor=N+M)
    G.var = list(range(M))

    # code.interact(local=locals())

    ##############################################################
    # To do: your code starts here
    # Add unary factors

    # Add parity factors
    # You may find the function itertools.product useful
    # (https://docs.python.org/3/library/itertools.html#itertools.product)

    raise NotImplementedError()
    ##############################################################
    return G

def do_part_a():
    yTilde = np.array([1, 1, 1, 1, 1, 1]).reshape(6, 1)
    print("yTilde.shape", yTilde.shape)
    H = np.array([
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 1]])
    epsilon = 0.05
    G = constructFactorGraph(yTilde, H, epsilon)
    ##############################################################
    # To do: your code starts here
    # Design two invalid codewords ytest1, ytest2 and one valid codewords
    # ytest3. Report their weights respectively. Keep the shape of each ytest as (6,) instead of (6,1)

    ##############################################################
    print(G.evaluateWeight(ytest1),
          G.evaluateWeight(ytest2),
          G.evaluateWeight(ytest3))

def sanity_check_noise():
    '''
    Sanity check applyChannelNoise to make sure bits are flipped at
    a reasonable proportion.
    '''
    N = 256
    epsilon = 0.05
    err_percent = 0
    num_trials = 1000
    x = np.zeros((N, 1), dtype='int32')
    for _ in range(num_trials):
        x_noise = applyChannelNoise(x, epsilon)
        err_percent += np.sum(x_noise)/N
    err_percent /= num_trials
    assert abs(err_percent-epsilon) < 0.005

def do_part_b(fixed=False, npy_file=None):
    '''
    We provide you an all-zero initialization of message x. If fixed=True and
    `npy_file` is not given, you should apply noise on y to get yTilde.
    Otherwise, load in the npy_file to get yTilde. Then do loopy BP to obtain
    the marginal probabilities of the unobserved y_i's.

    Args
    - fixed: bool, False if using random noise, True if loading from given npy file
    - npy_file: str, path to npy file, must be specified when fixed=True
    '''
    G, H = loadLDPC('ldpc36-128.mat')

    print((H.shape))
    epsilon = 0.05
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)
    if not fixed:
        yTilde = applyChannelNoise(y, epsilon)
        print("Applying random noise at eps={}".format(epsilon))
    else:
        assert npy_file is not None
        yTilde = np.load(npy_file)
        print("Loading yTilde from {}".format(npy_file))

    ##########################################################################################
    # To do: your code starts here


    ##############################################################
    #You dont need to return anything for this function
    pass
    
def do_part_cd(numTrials, error, iterations=50):
    '''
    param - numTrials: how many trials we repreat the experiments
    param - error: the transmission error probability
    param - iterations: number of Loopy BP iterations we run for each trial
    '''
    G, H = loadLDPC('ldpc36-128.mat')
    ##############################################################
    # To do: your code starts here



    ##############################################################
    #You dont need to return anything for this function
    pass

def do_part_ef(error):
    '''
    param - error: the transmission error probability
    '''
    G, H = loadLDPC('ldpc36-1600.mat')
    img = loadImage('images.mat', 'cs242')
    ##############################################################
    # To do: your code starts here
    # You should flattern img first and treat it as the message x in the previous parts.



    ################################################################
    #You dont need to return anything for this function
    pass

if __name__ == '__main__':
    print('Doing part (a): Should see 0.0, 0.0, >0.0')
    do_part_a()
    print("Doing sanity check applyChannelNoise")
    # sanity_check_noise()
    print('Doing part (b) fixed')
    # do_part_b(fixed=True, npy_file='part_b_test_1.npy')    # This should perfectly recover original code
    # do_part_b(fixed=True, npy_file='part_b_test_2.npy')    # This may not recover at perfect probability
    print('Doing part (b) random')
    # do_part_b(fixed=False)
    print('Doing part (c)')
    #do_part_cd(10, 0.06)
    print('Doing part (d)')
    # do_part_cd(10, 0.08)
    # do_part_cd(10, 0.10)
    print('Doing part (e)')
    # do_part_ef(0.06)
    print('Doing part (f)')
    # do_part_ef(0.10)
