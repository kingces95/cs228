# Gibbs sampling algorithm to denoise an image
# Author : Gunaa AV, Isaac Caswell
# Edits : Bo Wang, Kratarth Goel, Aditya Grover, Stephanie Wang
# Date : 2/10/2019

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import code
import random
import threading

def markov_blanket(i, j, Y, X):
    '''Gets the values of the Markov blanket of Y_ij.

    Args
    - i: int, row index of Y
    - j: int, col index of Y
    - Y: np.array, shape [H, W], note that read_txt_file() pads the image with
            0's to help you take care of corner cases
    - X: np.array, shape [H, W]

    Returns
    - blanket: list, values of the Markov blanket of Y_ij

    Example: if i = j = 1, the function should return
        [Y[0,1], Y[1,0], Y[1,2], Y[2,1], X[1,1]]

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    blanket = [ Y[i-1, j], Y[i, j-1], Y[i+1, j], Y[i, j+1], X[i, j] ]

    return blanket


def sampling_prob(markov_blanket):
    '''Computes P(Y=1 | MarkovBlanket(Y)).

    Args
    - markov_blanket: list, values of a variable's Markov blanket, e.g. [1,1,-1,1,-1]
        Because beta = eta in this case, the order doesn't matter. See part (a)
        sampling_prob([1,1,-1,1,-1])

    Returns
    - prob: float, the probability of the variable being 1 given its Markov blanket

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    ########
    # TODO: Your code here!
    m=np.array(markov_blanket)
    m_sum=m.sum()

    exp_m_sum=np.exp(m_sum)
    exp_m_neg_sum=np.exp(-m_sum)

    prob=exp_m_sum/(exp_m_sum + exp_m_neg_sum)
    # code.interact(local=locals())

    ########
    return prob


def sample(i, j, Y, X, dumb_sample=False):
    '''Samples a value for Y_ij. It should be sampled by:
    - if dumb_sample=True: the consensus of Markov blanket
    - if dumb_sample=False: the probability conditioned on all other variables

    Args
    - i: int, row index of Y
    - j: int, col index of Y
    - Y: np.array, shape [H, W], note that read_txt_file() pads the image with
            0's to help you take care of corner cases
    - X: np.array, shape [H, W]

    Returns: -1 or +1

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    blanket = markov_blanket(i, j, Y, X)

    assert X.shape == Y.shape
    assert i >= 0
    assert j >= 0
    assert i < X.shape[0]
    assert j < X.shape[1]

    if not dumb_sample:
        prob = sampling_prob(blanket)
        return np.random.choice([+1, -1], p=[prob, 1 - prob])
    else:
        return 1 if sum(blanket) > 0 else -1


def compute_energy(Y, X):
    '''Computes the energy E(Y, X) of the current assignment for the image.

    Args
    - Y: np.array, shape [H, W], note that read_txt_file() pads the image with
            0's to help you take care of corner cases
    - X: np.array, shape [H, W]

    Returns: float

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.

    This function can be efficiently implemented in one line with numpy parallel operations.
    You can give it a try to speed up your own experiments. This is not required.
    '''
    energy = 0.0
    ########
    # TODO: Your code here!

    up=np.roll(Y,1,0)
    right=np.roll(Y,1,1)

    energy=-(np.sum(Y*X) + np.sum(Y*up) + np.sum(Y*right))

    # code.interact(local=locals())
    #raise NotImplementedError()
    #########

    return energy


def get_posterior_by_sampling(
    filename, 
    max_burns, 
    max_samples,
    initialization='same', 
    logfile=None,
    dumb_sample=False):
    '''Performs Gibbs sampling and computes the energy of each  assignment for
    the image specified in filename.
    - If dumb_sample=False: runs max_burns iterations of burn in and then
        max_samples iterations for collecting samples
    - If dumb_sample=True: runs max_samples iterations and returns final image

    Args
    - filename: str, file name of image in text format, ends in '.txt'
    - max_burns: int, number of iterations of burn in
    - max_samples: int, number of iterations of collecting samples
    - initialization: str, one of ['same', 'neg', 'rand']
    - logfile: str, file name for storing the energy log (to be used for
        plotting later), see plot_energy()
    - dumb_sample: bool, True to use the trivial reconstruction in part (e)

    Returns
    - posterior: np.array, shape [H, W], type float64, value of each entry is
        the probability of it being 1 (estimated by the Gibbs sampler)
    - Y: np.array, shape [H, W], type np.int32,
        the final image (for dumb_sample=True, in part (e))
    - frequencyZ: dict, keys: count of the number of 1's in the Z region,
        values: frequency of such count

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    print ('Reading file:', filename)
    X = read_txt_file(filename)
    ph,pw = X.shape # padded hight/width
    
    S = np.zeros((ph,pw))
    frequencyZ = { }

    if initialization == "pass":
        return X, None, None

    if initialization == "same":
        Y = np.array(X)
    elif initialization == "neg":
        Y = np.full(X.shape, -1)
    elif initialization == "rand":    
        # fill Y with random pixel values -1 or 1
        rand_pixels = lambda x: random.choices((-1, 1), k=x)    
        Y = np.array(rand_pixels(ph * pw)).reshape((ph, pw))
    else:
        assert False, "Bad initialization value"
    
    Y[0]=0      # zero first row
    Y[-1]=0     # zero last row
    Y[:,0]=0    # zero first column
    Y[:,-1]=0   # zero last column

    if logfile != None:
        log = open(logfile, 'w')

    for t in range(max_burns + max_samples):
        if t % 10 == 0:
            print("%s -> %s, %d"  % (filename, logfile, t))

        # code.interact(local=locals())

        for pi in range(ph-2): # pixel i (pi) does not range over padding
            for pj in range(pw-2): # pixel j (pj) does not range over padding
                i = pi+1 # adjust for padding
                j = pj+1 # adjust for padding
                Y[i,j]=sample(i, j, Y, X, dumb_sample)

        if logfile != None:
            e = compute_energy(Y, X)
            log.write("%d\t%f\t%s\n" % (t, e, "B" if t < max_burns else "S"))

        if t < max_burns:
            continue

        S[Y > 0] += 1

        zcount=np.sum(Y[125+1:143+1, 162-1:174-1] > 0) # remove non-z + padding
        frequencyZ[zcount] = frequencyZ.get(zcount, 0) + 1

    # code.interact(local=locals())

    if logfile != None:
        log.close()

    posterior = np.zeros(S.shape, np.float64)
    posterior = S / max_samples

    return posterior, Y, frequencyZ

def denoise_image(
    filename, 
    max_burns, 
    max_samples, 
    initialization='rand',
    logfile=None, 
    dumb_sample=False):
    '''Performs Gibbs sampling on the image.

    Args:
    - filename: str, file name of image in text format, ends in '.txt'
    - max_burns: int, number of iterations of burn in
    - max_samples: int, number of iterations of collecting samples
    - initialization: str, one of ['same', 'neg', 'rand']
    - logfile: str, file name for storing the energy log (to be used for
        plotting later), see plot_energy()
    - dumb_sample: bool, True to use the trivial reconstruction in part (e)

    Returns
    - denoised: np.array, shape [H, W], type float64,
        denoised image scaled to [0, 1], the zero padding is also removed
    - frequencyZ: dict, keys: count of the number of 1's in the Z region,
        values: frequency of such count

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    posterior, Y, frequencyZ = get_posterior_by_sampling(
        filename, max_burns, max_samples, initialization, logfile=logfile,
        dumb_sample=dumb_sample)

    if dumb_sample:
        denoised = 0.5 * (Y + 1.0)  # change Y scale from [-1, +1] to [0, 1]
    else:
        denoised = np.zeros(posterior.shape, dtype=np.float64)
        denoised[posterior > 0.5] = 1
    return denoised[1:-1, 1:-1], frequencyZ


# ===========================================
# Helper functions for plotting etc
# ===========================================
def plot_energy(filename):
    '''Plots the energy as a function of the iteration number.

    Args
    - filename: str, path to energy log file, each row has three terms
        separated by a '\t'
        - iteration: iteration number
        - energy: the energy at this iteration
        - 'S' or 'B': indicates whether it's burning in or a sample

    e.g.
        1   -202086.0   B
        2   -210446.0   S
        ...
    '''
    x = np.genfromtxt(filename, dtype=None, encoding='utf8')
    its, energies, phases = zip(*x)
    its = np.asarray(its)
    energies = np.asarray(energies)
    phases = np.asarray(phases)

    burn_mask = (phases == 'B')
    samp_mask = (phases == 'S')
    assert np.sum(burn_mask) + np.sum(samp_mask) == len(x), 'Found bad phase'

    its_burn, energies_burn = its[burn_mask], energies[burn_mask]
    its_samp, energies_samp = its[samp_mask], energies[samp_mask]

    p1, = plt.plot(its_burn, energies_burn, 'r')
    p2, = plt.plot(its_samp, energies_samp, 'b')
    plt.title(filename)
    plt.xlabel('iteration number')
    plt.ylabel('energy')
    plt.legend([p1, p2], ['burn in', 'sampling'])
    plt.savefig('%s.png' % filename)
    plt.close()

def read_txt_file(filename):
    '''Reads in image from .txt file and adds a padding of 0's.

    Args
    - filename: str, image filename, ends in '.txt'

    Returns
    - Y: np.array, shape [H, W], type int32, padded with a border of 0's to
        take care of edge cases in computing the Markov blanket
    '''
    f = open(filename, 'r')
    lines = f.readlines()
    height = int(lines[0].split()[1].split('=')[1])
    width = int(lines[0].split()[2].split('=')[1])
    Y = np.zeros([height+2, width+2], dtype=np.int32)
    for line in lines[2:]:
        i, j, val = [int(entry) for entry in line.split()]
        Y[i+1, j+1] = val
    return Y

def convert_to_png(denoised_image, title):
    '''Saves an array as a PNG file with the given title.

    Args
    - denoised_image: np.array, shape [H, W]
    - title: str, title and filename for figure
    '''
    plt.imshow(denoised_image, cmap='gray_r')
    plt.title(title)
    plt.savefig(title + '.png')
    plt.close()

def get_error(img_a, img_b):
    '''Computes the fraction of all pixels that differ between two images.

    Args
    - img_a: np.array, shape [H, W]
    - img_b: np.array, shape [H, W]

    Returns: float
    '''
    assert img_a.shape == img_b.shape, "%s vs %s" % (img_a.shape, img_b.shape)
    N = img_a.shape[0] * img_a.shape[1]  # number of pixels in an image
    return np.sum(np.abs(img_a - img_b) > 1e-5) / float(N)


#==================================
# doing part (c), (d), (e), (f)
#==================================

def perform_part_c():
    '''
    Run denoise_image() with different initializations and plot out the energy
    functions.
    '''

    MAX_BURNS=100
    MAX_SAMPLES=1000
    image_txt='./data/small.txt'
    # image_txt='./data/noisy_20.txt'

    get_posterior_by_sampling(image_txt, MAX_BURNS, MAX_SAMPLES, initialization='rand', logfile='logs/log_rand')
    plot_energy('log_rand')

    get_posterior_by_sampling(image_txt, MAX_BURNS, MAX_SAMPLES, initialization='neg', logfile='logs/log_neg')
    plot_energy('log_neg')

    get_posterior_by_sampling(image_txt, MAX_BURNS, MAX_SAMPLES, initialization='same', logfile='logs/log_same')
    plot_energy('log_same')

def perform_part_d():
    '''
    Run denoise_image() with different noise levels of 10% and 20%, and report
    the errors between denoised images and original image. Strip the 0-padding
    before computing the errors. Also, don't forget that denoise_image() strips
    the zero padding and scales them into [0, 1].
    '''
    MAX_BURNS=100
    MAX_SAMPLES=1000
    #MAX_SAMPLES=100

    # small, _ = denoise_image('./data/small.txt', 0, 0, initialization='pass')
    # convert_to_png(small, 'small')
    
    # denoised_small, _ = denoise_image('./data/small.txt', MAX_BURNS, MAX_SAMPLES, initialization='same')
    # convert_to_png(denoised_small, 'denoised_small')
    
    # print("small vs denoised_small = %f" % get_error(small, denoised_small))
    # code.interact(local=locals())
    # return

    orig, _ = denoise_image('./data/orig.txt', 0, 0, initialization='pass')
    print("orig vs orig = %f" % get_error(orig, orig))
    convert_to_png(orig, 'orig')

    denoised_10, _ = denoise_image('./data/noisy_10.txt', MAX_BURNS, MAX_SAMPLES, initialization='same')
    print("orig vs denoised_10 = %f" % get_error(orig, denoised_10))
    convert_to_png(denoised_10, 'denoised_10')

    denoised_20, _ = denoise_image('./data/noisy_20.txt', MAX_BURNS, MAX_SAMPLES, initialization='same')
    print("orig vs denoised_20 = %f" % get_error(orig, denoised_20))
    convert_to_png(denoised_20, 'denoised_20')

    # code.interact(local=locals())
    pass

def perform_part_e():
    '''
    Run denoise_image() using dumb sampling with different noise levels of 10%
    and 20%.
    '''
    MAX_BURNS=100
    MAX_SAMPLES=0
    #MAX_SAMPLES=100

    # small, _ = denoise_image('./data/small.txt', 0, 0, initialization='pass')
    # convert_to_png(small, 'small')
    
    # dumb_denoised_small, _ = denoise_image('./data/small.txt', MAX_BURNS, MAX_SAMPLES, initialization='same', dumb_sample=True)
    # convert_to_png(dumb_denoised_small, 'dumb_denoised_small')
    
    # print("small vs dumb_denoised_small = %f" % get_error(small, dumb_denoised_small))
    # code.interact(local=locals())
    # return

    orig, _ = denoise_image('./data/orig.txt', 0, 0, initialization='pass')
    print("orig vs orig = %f" % get_error(orig, orig))
    convert_to_png(orig, 'orig')

    # dumb_denoised_10, _ = denoise_image('./data/noisy_10.txt', MAX_BURNS, MAX_SAMPLES, initialization='same', dumb_sample=True)
    # print("orig vs dumb_denoised_10 = %f" % get_error(orig, dumb_denoised_10))
    # convert_to_png(dumb_denoised_10, 'dumb_denoised_10')

    # dumb_denoised_20, _ = denoise_image('./data/noisy_20.txt', MAX_BURNS, MAX_SAMPLES, initialization='same', dumb_sample=True)
    # print("orig vs dumb_denoised_20 = %f" % get_error(orig, dumb_denoised_20))
    # convert_to_png(dumb_denoised_20, 'dumb_denoised_20')

    ran_dumb_denoised_10, _ = denoise_image('./data/noisy_10.txt', MAX_BURNS, MAX_SAMPLES, initialization='rand', dumb_sample=True)
    print("orig vs ran_dumb_denoised_10 = %f" % get_error(orig, ran_dumb_denoised_10))
    convert_to_png(ran_dumb_denoised_10, 'ran_dumb_denoised_10')

    ran_dumb_denoised_20, _ = denoise_image('./data/noisy_20.txt', MAX_BURNS, MAX_SAMPLES, initialization='rand', dumb_sample=True)
    print("orig vs ran_dumb_denoised_20 = %f" % get_error(orig, ran_dumb_denoised_20))
    convert_to_png(ran_dumb_denoised_20, 'ran_dumb_denoised_20')

    # code.interact(local=locals())
    pass

def perform_part_f():
    '''
    Run Z square analysis
    '''
    MAX_BURNS = 100
    MAX_SAMPLES = 1000
    width = 1.0

    # _, f10 = denoise_image('./data/noisy_10.txt', MAX_BURNS, MAX_SAMPLES, initialization='same')
    # plt.clf()
    # plt.bar(list(f10.keys()), list(f10.values()), width, color='b')
    # plt.savefig('./z10')
    # plt.show()

    _, f20 = denoise_image('./data/noisy_20.txt', MAX_BURNS, MAX_SAMPLES, initialization='same')
    plt.clf()
    plt.bar(list(f20.keys()), list(f20.values()), width, color='b')
    plt.savefig('./z20')
    # plt.show()

    # code.interact(local=locals())

if __name__ == '__main__':
    # sampling_prob([1,1,-1,1,-1])
    # X = read_txt_file("./data/small.txt")
    # Y = X
    # compute_energy(X, Y)

    # perform_part_c()
    # perform_part_d()
    # perform_part_e()
    perform_part_f()
