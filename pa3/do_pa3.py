###############################################################################
# Finishes PA 3
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018


## Edited by Zhangyuan Wang, 01/2019
## Edired by Akshat Jindal, 01/2020
###############################################################################

## Utility code for PA3
from re import X
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools as it
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

    yTilde=np.array(y)
    i=np.random.uniform()
    for i in range(yTilde.size):
        if np.random.uniform() <= epsilon:
            yTilde[i] = 1 if yTilde[i] == 0 else 0
    # code.interact(local=locals())

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

def checkParity(y, H):
    '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword x=Hy mod 2

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    return np.mod(np.dot(H, y), 2)

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
    # G.domain = [[0,1] for _ in range(numVar)]
    # G.messagesVarToFactor = {}
    # G.messagesFactorToVar = {}

    numVar=M
    numUnaryFactor=M
    numParityFactor=N
    numFactor=numUnaryFactor + numParityFactor

    assert yTilde.size == numVar

    G.factors = [[] for _ in range(numFactor)]

    # add unary factors + edges
    for m in range(numVar):
        bitId=m
        unaryFactorId=m

        # var (bit) -> factor (unary)
        G.varToFactor[bitId].append(unaryFactorId)

        # factor (unary) -> var (bin)
        G.factorToVar[unaryFactorId].append(bitId)

        # unary factor
        bit=yTilde[m]
        val=np.array(( epsilon, epsilon ))
        val[bit]=1-epsilon

        G.factors[unaryFactorId] = Factor(
            scope=[ bitId ], card=[ 2 ], val=val, name="%d (unary), bit=%d" % (unaryFactorId, bit))

    # add parity factors + edges
    for m in range(numVar):
        bitId=m

        for n in range(numParityFactor):
            parityFactorId=n + numUnaryFactor

            if H[n][m] == 0:
                continue

            # var (bit) -> factor (parity)
            G.varToFactor[bitId].append(parityFactorId)

            # factor (parity) -> var (bit)
            G.factorToVar[parityFactorId].append(bitId)

    for n in range(numParityFactor):
        parityFactorId=n + numUnaryFactor

        scope=np.where(H[n] == 1)[0]
        dimensions=scope.size
        card=[2] * dimensions        
        
        binary=(0., 1.)

        # list of binary of length dimensions; one binary pair per parity factor edge
        vals=list(it.repeat(binary, dimensions))

        # permutations of bit value for parity factor
        cartesian_product=np.array(list(it.product(*vals)))

        # count of set bits per permutation
        set_bits_per_cartesian_product=np.array(cartesian_product.sum(1))

        # even number of set bits => 1, otherwise 0
        parity_bits= np.array((set_bits_per_cartesian_product + 1) % 2)
        assert parity_bits[0] == 1
        
        val=parity_bits.reshape(card)
        assert card == list(val.shape)

        # parity factor
        G.factors[parityFactorId] = Factor(
            scope=scope, card=card, val=val, name="%d (parity)" % parityFactorId)


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
    ytest1=np.array([1, 0, 1, 0, 1, 0])
    ytest2=np.array([1, 1, 1, 1, 1, 1])
    ytest3=np.array([0, 0, 0, 0, 0, 0])
    assert ytest1.shape == (6,)
    assert ytest2.shape == (6,)
    assert ytest3.shape == (6,)
    # code.interact(local=locals())

    print(G.evaluateWeight([1, 0, 1, 0, 1, 0]))
    print(G.evaluateWeight([1, 1, 1, 1, 1, 1]))
    print(G.evaluateWeight([0, 0, 0, 0, 0, 0]))
    print(G.evaluateWeight([1, 0, 1, 1, 1, 1]))

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

    parity=checkParity(y, H)
    # print(parity)

    if not fixed:
        yTilde = applyChannelNoise(y, epsilon)
        print("Applying random noise at eps={}".format(epsilon))
        print("{} out of {} bits flipped".format(sum([yTilde[i]!=y[i][0] for i in range(len(y))]),len(y)))

    else:
        assert npy_file is not None
        yTilde = np.load(npy_file)
        print("Loading yTilde from {}".format(npy_file))

    graph = constructFactorGraph(yTilde, H, epsilon)
    graph.runParallelLoopyBP(20)
    yRecovered=graph.getMessage()

    print("miss {} out of {} bits".format(sum([yRecovered[i]!=y[i][0] for i in range(len(y))]),len(y)))


    parity=checkParity(yRecovered.reshape(y.shape), H)
    print(parity.T)
    
def do_part_cd(numTrials, epsilon, iterations=50):
    '''
    param - numTrials: how many trials we repreat the experiments
    param - epsilon: the transmission epsilon probability
    param - iterations: number of Loopy BP iterations we run for each trial
    '''
    G, H = loadLDPC('ldpc36-128.mat')
    ##############################################################
    # To do: your code starts here

    print((H.shape))
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)

    for i in range(numTrials):
        print("--", i)
        yTilde = applyChannelNoise(y, epsilon)
        print("Applying random noise at eps={}".format(epsilon))
        print("Flipped bits = {}".format(np.sum([yTilde==1])))
        # print(yTilde.T)

        graph = constructFactorGraph(yTilde, H, epsilon)
        graph.runParallelLoopyBP(iterations)

        # code.interact(local=locals())

    ##############################################################
    #You dont need to return anything for this function
    pass

def do_part_ef(epsilon):
    '''
    param - error: the transmission error probability
    '''
    G, H = loadLDPC('ldpc36-1600.mat')
    img = loadImage('images.mat', 'cs242')
    # x = np.tile(img, (2,2))

    N = G.shape[1]
    x=np.array(img).reshape(N,1)
    y = encodeMessage(x, G)
    yTilde=applyChannelNoise(y, epsilon)

    graph=constructFactorGraph(yTilde, H, epsilon)
    # code.interact(local=locals())

    ##############################################################
    # To do: your code starts here
    # You should flattern img first and treat it as the message x in the previous parts.

    # code.interact(local=locals())
    snapshot=np.array((0,1,2,3,5,10,20,30))
    plt.figure()
    graph.runParallelLoopyBP(0)

    plot=0
    for i in range(31):
        print(i)
        if i in snapshot:
            yRecovered=yTilde if i==0 else graph.getMessage()
            plt.subplot(2, 4, plot + 1)
            plot=plot+1
            plt.imshow(
                yRecovered[:1600].reshape(40,40)
            )
            plt.title("Sample: " + str(i))
        graph.runParallelLoopyBPi(1)
    plt.tight_layout()
    plt.savefig("a4", bbox_inches="tight")
    plt.show()
    plt.close()


    ################################################################
    #You dont need to return anything for this function
    pass

if __name__ == '__main__':
    print('Doing part (a): Should see 0.0, 0.0, >0.0')
    # do_part_a()
    # do_part_aa()
    print("Doing sanity check applyChannelNoise")
    # sanity_check_noise()
    print('Doing part (b) fixed')
    # do_part_b(fixed=True, npy_file='part_b_test_1.npy')    # This should perfectly recover original code
    # do_part_b(fixed=True, npy_file='part_b_test_2.npy')    # This may not recover at perfect probability
    print('Doing part (b) random')
    # do_part_b(fixed=False)
    print('Doing part (c)')
    iterations=50
    trials=10
    # do_part_cd(trials, 0.06, iterations)
    print('Doing part (d)')
    # do_part_cd(trials, 0.08, iterations)
    # do_part_cd(trials, 0.10, iterations)
    print('Doing part (e)')
    # do_part_ef(0.06)
    print('Doing part (f)')
    do_part_ef(0.10)
    # code.interact(local=locals())
