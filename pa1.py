"""
CS 228: Probabilistic Graphical Models
Winter 2022
Programming Assignment 1: Bayesian Networks

Author: Aditya Grover
Updated by Jiaming Song and Chris Cundy

The following setup is recommended:
- python >= 3 (for pickle)
- scipy >= 1.2 (for logsumexp)
although code has been provided to handle import errors for earlier versions.

Note that the gradescope autograder (where your code will run)
has the following installed:

python=3.6.9
decorator==4.3.0
gradescope-utils==0.3.0
networkx==2.2
numpy==1.16.0
scipy==1.2.0
matplotlib==2.2.3
imageio=2.13.5

The autograder has 4 CPU cores, 6GB of RAM and times out after 40 minutes.
This should be sufficient for a straightforward implementation to finish in time. 

Since your plots will be checked against reference implementations, do not change
the plotting code for your submission. Be careful of importing packages
that may change the plotting behaviour. Submit only the `pa1.py` file to the 
autograder.

Note that the `main` function describes and instantiates global variables 
`disc_z1, disc_z2, bayes_net`, which you may use. 
"""
from cmath import exp
import sys
import pickle as pkl
import threading

import numpy as np

import matplotlib.pyplot as plt

from scipy.io import loadmat

try:
    # https://github.com/scipy/scipy/blob/v0.14.0/scipy/misc/common.py
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

from typing import Tuple


def plot_histogram(
    data, title="histogram", xlabel="value", ylabel="frequency", savefile="hist"
):
    """
    Plots a histogram.  DO NOT MODIFY
    """

    plt.figure()
    plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(savefile, bbox_inches="tight")
    plt.show()
    plt.close()

    return


def get_p_z1(z1_val: float) -> float:
    """
    Helper. Computes the prior probability for variable z1 to take value z1_val.
    P(Z1=z1_val)
    """

    return bayes_net["prior_z1"][z1_val]

def get_p_z2(z2_val: float) -> float:
    """
    Helper. Computes the prior probability for variable z2 to take value z2_val.
    P(Z2=z2_val)
    """

    return bayes_net["prior_z2"][z2_val]

def get_pixels_sampled_from_p_x_joint_z1_z2() -> np.ndarray:
    """
    Called by q4()

    This function should sample from the joint probability distribution specified by the model,
    and return the sampled values of all the pixel variables (x).
    Note that this function should return the sampled values of ONLY the pixel variables (x),
    discarding the z part.

    Returns:
        x: A size-(784,) array of pixel values

    TODO: replace pass with your code to implement the above function

    for i in Val(Z_1)
        for j in Val(Z_2)


    """
    # return np.random.randint(2, size=784)

    n_disc_z = 25

    prior_z1 = bayes_net['prior_z1']
    prior_z2 = bayes_net['prior_z2']
    cond_likelihood = bayes_net['cond_likelihood']

    z1_distribution = np.linspace(-3, 3, n_disc_z)
    for i in range(n_disc_z):
        z1_distribution[i]=prior_z1[z1_distribution[i]]

    z2_distribution = np.linspace(-3, 3, n_disc_z)
    for i in range(n_disc_z):
        z2_distribution[i]=prior_z2[z2_distribution[i]]

    z1_sample=np.random.choice(disc_z1, p=z1_distribution)
    z2_sample=np.random.choice(disc_z2, p=z2_distribution)
    cond_likelihood_z1_z2=cond_likelihood[z1_sample, z2_sample][0]
    
    sample=np.zeros(784)
    for i in range(len(sample)):
        sample[i]=np.random.choice([0, 1], p=[
            1 - cond_likelihood_z1_z2[i],
            cond_likelihood_z1_z2[i],
        ])

    return sample

def get_p_cond_z1_z2(z1_val: float, z2_val: float) -> np.ndarray:
    """
    Called by q5()

    Computes an array consisting of the conditional probabilies of each pixel taking on the value
    1, given that z1 assumes value z1_val and z2 assumes value z2_val

    Args:
        z1_val: The value taken on by z1
        z2_val: The value taken on by z2

    Returns:
        p: A size-(784,) array of conditional probabilities

    TODO: replace pass with your code to implement the above function
    """
    cond_likelihood = bayes_net['cond_likelihood']
    cond_likelihood_z1_z2=cond_likelihood[z1_val, z2_val][0]
    
    sample=np.zeros(784)
    for i in range(len(sample)):
        sample[i]=cond_likelihood_z1_z2[i]

    return sample

def get_log_marginal_likelihood(data: np.ndarray) -> np.ndarray:
    """
    Called by q6()

    Computes log marginal likelihood for a dataset.

    Using the identity p(x) = \sum_{z_1}\sum_{z_2}p(z_1,z_2,x),
    computes the marginal likelihood for each image in the input

    Args:
        data: A size-(n, 784) array representing n input images
    Returns:
        log_marginal_likelihoods: A size-(n,) array containing the 
            log-likelihood for each input image

    TODO: replace pass with your code to implement the above function.
    Remember that the `logsumexp` function can be used to increase numerical
    accuracy.
    """
    pass

def p_x0_gz0():
    z1_i = 0
    sum_p = 0
    sum_p_x = 0
    for z2_i in range(n_disc_z):
        sum_p += z1_distribution[z1_i] * z2_distribution[z2_i] * cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0][0]
        sum_p += z1_distribution[z1_i] * z2_distribution[z2_i] * (1 - cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0][0])
        sum_p_x += z1_distribution[z1_i] * z2_distribution[z2_i] * cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0][0]

    return sum_p_x / sum_p

def p_X_gz0():
    z1_i = 0
    sum_p = np.zeros(x_cardinality)
    sum_p_x = np.zeros(x_cardinality)

    for z2_i in range(n_disc_z):
        sum_p += z1_distribution[z1_i] * z2_distribution[z2_i] * cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0]
        sum_p += z1_distribution[z1_i] * z2_distribution[z2_i] * (1 - cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0])
        sum_p_x += z1_distribution[z1_i] * z2_distribution[z2_i] * cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0]

    return sum_p_x / sum_p

def p_X_gZ1() -> np.ndarray:

    cond_likelihood_z1 = np.zeros((n_disc_z, x_cardinality))
    for z1_i in range(n_disc_z):

        sum_p = np.zeros(x_cardinality)
        sum_p_x = np.zeros(x_cardinality)
        for z2_i in range(n_disc_z):
            sum_p += z1_distribution[z1_i] * z2_distribution[z2_i] * cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0]
            sum_p += z1_distribution[z1_i] * z2_distribution[z2_i] * (1 - cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0])
            sum_p_x += z1_distribution[z1_i] * z2_distribution[z2_i] * cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0]

        cond_likelihood_z1[z1_i]=sum_p_x / sum_p

    return cond_likelihood_z1

def p_X_gZ2() -> np.ndarray:

    cond_likelihood_z2 = np.zeros((n_disc_z, x_cardinality))
    for z2_i in range(n_disc_z):

        sum_p = np.zeros(x_cardinality)
        sum_p_x = np.zeros(x_cardinality)
        for z1_i in range(n_disc_z):
            sum_p += z1_distribution[z1_i] * z2_distribution[z2_i] * cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0]
            sum_p += z1_distribution[z1_i] * z2_distribution[z2_i] * (1 - cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0])
            sum_p_x += z1_distribution[z1_i] * z2_distribution[z2_i] * cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0]

        cond_likelihood_z2[z2_i]=sum_p_x / sum_p

    return cond_likelihood_z2

def get_conditional_expectation_old(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    global n_disc_z, x_cardinality
    x_cardinality = 784
    n_disc_z = 25

    global cond_likelihood
    cond_likelihood = bayes_net['cond_likelihood']

    global z1_distribution
    z1_distribution = np.zeros(n_disc_z)
    for i in range(n_disc_z):
        z1_distribution[i] = bayes_net['prior_z1'][disc_z1[i]]

    global z2_distribution
    z2_distribution = np.zeros(n_disc_z)
    for i in range(n_disc_z):
        z2_distribution[i]= bayes_net['prior_z2'][disc_z2[i]]

    return get_conditional_expectation_body(data)

def get_conditional_expectation_body(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    n = len(data)

    cond_likelihood_z1 = np.zeros((n_disc_z, x_cardinality))
    for z1_i in range(n_disc_z):
        for z2_i in range(n_disc_z):           
            cond_likelihood_z1[z1_i] += \
                z2_distribution[z2_i] * cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0]

    cond_likelihood_z2 = np.zeros((n_disc_z, x_cardinality))
    for z2_i in range(n_disc_z):
        for z1_i in range(n_disc_z):
            cond_likelihood_z2[z2_i] += \
                z1_distribution[z1_i] * cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0]

    print(np.min(cond_likelihood_z1))
    print(np.min(cond_likelihood_z2))
    # should match P(Z1|x_1)
    # print(cond_likelihood_z1[0][0])
    # print(p_x0_gz0())
    # print(cond_likelihood_z1[0][0] - p_x0_gz0())

    # print(cond_likelihood_z1[0] - p_X_gz0())

    # print(p_x0_gz0())
    # print(p_X_gz0()[0])
    # print(p_X_gZ1()[0][0])

    # cond_likelihood_z1=p_X_gZ1()
    # cond_likelihood_z2=p_X_gZ2()

    cond_e_z_1=np.zeros(n)
    cond_e_z_2=np.zeros(n)

    # global ak
    # ak=1.2

    for i in range(n):
        # if labels[i] != 8:
        #     continue

        image=data[i]
        log_pz1_px_z1=np.zeros(n_disc_z)
        log_pz2_px_z2=np.zeros(n_disc_z)
        for zi in range(n_disc_z):
            pz1 = z1_distribution[zi]
            pz2 = z2_distribution[zi]
       
            log_pz1_px_z1[zi]=np.log(pz1) + np.sum(np.log(pixel_to_p(cond_likelihood_z1[zi], image)))
            log_pz2_px_z2[zi]=np.log(pz2) + np.sum(np.log(pixel_to_p(cond_likelihood_z2[zi], image)))

        # print(np.min(pz1_px_z1), np.max(pz1_px_z1))

        pz1_px_z1_sum=logsumexp(log_pz1_px_z1)
        pz2_px_z2_sum=logsumexp(log_pz2_px_z2)

        pz1_image=np.zeros(n_disc_z)
        pz2_image=np.zeros(n_disc_z)
        for zi in range(n_disc_z):
            pz1_image[zi]=log_pz1_px_z1[zi] - pz1_px_z1_sum
            pz2_image[zi]=log_pz2_px_z2[zi] - pz2_px_z2_sum

        cond_e_z_1[i]=np.sum(disc_z1 * np.exp(pz1_image))
        cond_e_z_2[i]=np.sum(disc_z2 * np.exp(pz2_image))

    # print(bayes_net['prior_z1'])
    # print(len(bayes_net['prior_z1']))

    return cond_e_z_1, cond_e_z_2

def pixel_to_p(p_of_pixel, image):
    return (image) * (p_of_pixel) + (1 - image) * (1 - p_of_pixel)

# def pz_px_z(image, p_z, cond_likelihood_z, image: np.ndarray):
#     image = cond_likelihood_z    
#     pass

def q4():
    """
    Plots the pixel variables sampled from the joint distribution as 28 x 28 images.
    TODO: no need to modify the code here, but implement get_pixels_sampled_from_p_x_joint_z1_z2()
    """

    # DO NOT MODIFY
    # ----------
    plt.figure()
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(
            get_pixels_sampled_from_p_x_joint_z1_z2().reshape(28, 28), cmap="gray"
        )
        plt.title("Sample: " + str(i + 1))
    plt.tight_layout()
    plt.savefig("a4", bbox_inches="tight")
    plt.show()
    plt.close()
    # ----------

    return


def q5():
    """
    Plots the expected images for each latent configuration on a 2D grid.
    TODO: no need to modify the code here, but implement get_p_cond_z1_z2()
    """

    # DO NOT MODIFY
    # ----------
    canvas = np.empty((28 * len(disc_z2), 28 * len(disc_z1)))
    for i, z1_val in enumerate(disc_z1):
        for j, z2_val in enumerate(disc_z2):
            canvas[
                (len(disc_z2) - j - 1) * 28 : (len(disc_z2) - j) * 28,
                i * 28 : (i + 1) * 28,
            ] = get_p_cond_z1_z2(z1_val, z2_val).reshape(28, 28)

    plt.figure()
    plt.imshow(canvas, cmap="gray")
    plt.xlabel("Z_1")
    plt.ylabel("Z_2")
    plt.tight_layout()
    plt.savefig("a5", bbox_inches="tight")
    plt.show()
    plt.close()
    # ----------

    return

def get_conditional_expectation(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    global n_disc_z, x_cardinality
    x_cardinality = 784
    n_disc_z = 25

    global cond_likelihood
    cond_likelihood = bayes_net['cond_likelihood']

    global z1_distribution
    z1_distribution = np.zeros(n_disc_z)
    for i in range(n_disc_z):
        z1_distribution[i] = bayes_net['prior_z1'][disc_z1[i]]

    global z2_distribution
    z2_distribution = np.zeros(n_disc_z)
    for i in range(n_disc_z):
        z2_distribution[i]= bayes_net['prior_z2'][disc_z2[i]]

    n=len(data)
    cond_e_z_1=np.zeros(n)
    cond_e_z_2=np.zeros(n)

    for i in range(n):
        datum=data[i]
        if i % 100 == 0:
            print ("%d / %d" % (i, n))
            
        denominator=get_log_marginal_likelihood_datum(datum)
        denominator=np.exp(denominator)

        for z1_i in range(n_disc_z):
            cond_e_z_1[i] += disc_z1[z1_i] * np.exp(get_log_marginal_likelihood_datum_z1(datum, z1_i))
        cond_e_z_1[i] /= denominator

        for z2_i in range(n_disc_z):
            cond_e_z_2[i] += disc_z2[z2_i] * np.exp(get_log_marginal_likelihood_datum_z2(datum, z2_i))
        cond_e_z_2[i] /= denominator

    return cond_e_z_1, cond_e_z_2
       

def get_mx_x_Z2_Z1_of_log_zzp(data):
    n_disc_z = 25

    def f_log_zzp(d3, d2, d1):
        print(d2)
        p_z2=z2_distribution[d2]
        p_z1=z1_distribution[d1]
        p_pixel = pixel_to_p(cond_likelihood[disc_z1[d1], disc_z2[d2]][0], data[d3])
        return np.log(p_z1) + np.log(p_z2) + np.sum(np.log(p_pixel))

    return np.fromfunction(f_log_zzp, (len(data), n_disc_z, n_disc_z))

def get_log_marginal_likelihood(data):
    progress = 100

    n = len(data) 
    marginal_log_likelihood=np.zeros(n)
    for i in range(n):
        if i % progress == 0:
            print('val',i,'/',n)
        marginal_log_likelihood[i]=get_log_marginal_likelihood_datum(data[i])

    return marginal_log_likelihood

def get_log_marginal_likelihood_datum_z2(datum, z2_i):

    log_all_Z1_Z2_x = np.zeros(n_disc_z)

    i = 0
    for z1_i in range(n_disc_z):
        p_pixel = pixel_to_p(cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0], datum)
        
        p_z1=z1_distribution[z1_i]
        assert p_z1 > 0 and p_z1 < 1

        p_z2=z2_distribution[z2_i]
        assert p_z2 > 0 and p_z2 < 1

        # joint for one z1/z2
        log_all_Z1_Z2_x[i] = np.log(p_z1) + np.log(p_z2) + np.sum(np.log(p_pixel))
        i = i + 1

    a_max = np.max(log_all_Z1_Z2_x)
    log_all_Z1_Z2_x_trick = log_all_Z1_Z2_x - a_max
    expsum = np.sum(np.exp(log_all_Z1_Z2_x_trick))
    log_prob_x = a_max + np.log(expsum)

    return log_prob_x 

def get_log_marginal_likelihood_datum_z1(datum, z1_i):

    log_all_Z1_Z2_x = np.zeros(n_disc_z)

    i = 0
    for z2_i in range(n_disc_z):
        p_pixel = pixel_to_p(cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0], datum)
        
        p_z1=z1_distribution[z1_i]
        assert p_z1 > 0 and p_z1 < 1

        p_z2=z2_distribution[z2_i]
        assert p_z2 > 0 and p_z2 < 1

        # joint for one z1/z2
        log_all_Z1_Z2_x[i] = np.log(p_z1) + np.log(p_z2) + np.sum(np.log(p_pixel))
        i = i + 1

    a_max = np.max(log_all_Z1_Z2_x)
    log_all_Z1_Z2_x_trick = log_all_Z1_Z2_x - a_max
    expsum = np.sum(np.exp(log_all_Z1_Z2_x_trick))
    log_prob_x = a_max + np.log(expsum)

    return log_prob_x 

def get_log_marginal_likelihood_datum(datum):

    # joint for all z1/z2
    log_all_Z1_Z2_x = np.zeros(n_disc_z * n_disc_z)

    i = 0
    for z1_i in range(n_disc_z):
        for z2_i in range(n_disc_z):
            p_pixel = pixel_to_p(cond_likelihood[disc_z1[z1_i], disc_z2[z2_i]][0], datum)
            # assert  min(p_pixel) >= 0 and max(p_pixel) <= 1
            # print(np.max(p_pixel))
            
            p_z1=z1_distribution[z1_i]
            assert p_z1 > 0 and p_z1 < 1

            p_z2=z2_distribution[z2_i]
            assert p_z2 > 0 and p_z2 < 1

            # joint for one z1/z2
            log_all_Z1_Z2_x[i] = np.log(p_z1) + np.log(p_z2) + np.sum(np.log(p_pixel))
            i = i + 1

    assert i == n_disc_z * n_disc_z

    a_max = np.max(log_all_Z1_Z2_x)
    # assert a_max < -30, \
    #     "a_max < -30, %d" % a_max

    log_all_Z1_Z2_x_trick = log_all_Z1_Z2_x - a_max
    # assert max(log_all_Z1_Z2_x_trick) <= 0, \
    #     "max(log_all_Z1_Z2_x_trick) < 0, %f" % log_all_Z1_Z2_x_trick

    expsum = np.sum(np.exp(log_all_Z1_Z2_x_trick))
    # assert expsum >= 1 and expsum < 4, \
    #     "expsum > 1 and expsum < 4, %f, a_max=%f" % (expsum, a_max)

    log_prob_x = a_max + np.log(expsum)
    # assert a_max < -30, \
    #     "log_prob_x < -25, %d" % log_prob_x

    # print(np.max(log_all_Z1_Z2_x))
    # assert log_prob_x == logsumexp(log_all_Z1_Z2_x), \
    #     "log_prob_x == logsumexp(log_all_Z1_Z2_x)"

    # a_min = np.min(log_all_Z1_Z2_x)
    # if (log_prob_x > -75 and log_prob_x < -25):
    #     print('---')
    #     print('a_max',a_max, 'a_min',a_min, 'sum', expsum,'expsum', \
    #         'average', np.average(log_all_Z1_Z2_x), 'log_prob_x',log_prob_x)
    # else:
    #     print('a_max',a_max, 'a_min',a_min, 'sum', expsum,'expsum', \
    #         'average', np.average(log_all_Z1_Z2_x), 'log_prob_x',log_prob_x)

    return log_prob_x 

def q6():
    """
    The provided code loads the data and plots the histogram.
    """
    mat = loadmat("q6.mat")
    val_data = mat["val_x"]
    test_data = mat["test_x"]

    """
    TODO: Implement the `get_log_marginal_likelihood` function and use it to compute marginal likelihoods of the
    val data. Using the val statistics, populate two lists, `real_marginal_log_likelihood` and
    `corrupt_marginal_log_likelihood` which contains the log-likelihood of the "real" and
    "corrupted" samples in the test data.
    
    You might want to use the logsumexp function from scipy (imported from above).
    """
    
    # Your code here
    global n_disc_z, x_cardinality
    x_cardinality = 784
    n_disc_z = 25

    global cond_likelihood
    cond_likelihood = bayes_net['cond_likelihood']

    global z1_distribution
    z1_distribution = np.zeros(n_disc_z)
    for i in range(n_disc_z):
        z1_distribution[i] = bayes_net['prior_z1'][disc_z1[i]]

    global z2_distribution
    z2_distribution = np.zeros(n_disc_z)
    for i in range(n_disc_z):
        z2_distribution[i]= bayes_net['prior_z2'][disc_z2[i]]

    # print(len(val_data), len(test_data))

    real_n = len(val_data)
    real_n = 1000
    # print(real_n)

    val_marginal_log_likelihood=get_log_marginal_likelihood(val_data)
    # print(val_marginal_log_likelihood)

    average = np.average(val_marginal_log_likelihood)
    std = np.std(val_marginal_log_likelihood)
    std3 = std * 3
    lower_bound = average - std3
    upper_bound = average + std3
    # print('average', average, 'std', std, 'std3', std3, 'low', lower_bound, 'high', upper_bound)

    test_n = len(test_data)
    test_n = 1000
    # print(test_n)

    test_marginal_log_likelihood=get_log_marginal_likelihood(test_data)

    tmll=test_marginal_log_likelihood
    corrupt_marginal_log_likelihood = tmll[(tmll < lower_bound) | (tmll > upper_bound)]    
    real_marginal_log_likelihood    = tmll[(tmll >= lower_bound) & (tmll <= upper_bound)]

    # print(len(real_marginal_log_likelihood))
    # print(len(corrupt_marginal_log_likelihood))
    # print(corrupt_marginal_log_likelihood)

    # DO NOT MODIFY
    # ----------
    plot_histogram(
        real_marginal_log_likelihood,
        title="Histogram of marginal log-likelihood for real data",
        xlabel="marginal log-likelihood",
        savefile="a6_hist_real",
    )

    plot_histogram(
        corrupt_marginal_log_likelihood,
        title="Histogram of marginal log-likelihood for corrupted data",
        xlabel="marginal log-likelihood",
        savefile="a6_hist_corrupt",
    )
    # ----------

    return

def q7():
    """
    Loads the data and plots a color coded clustering of the conditional expectations.

    TODO: no need to modify code here, but implement the `get_conditional_expectation` function according to the problem statement.
    """

    mat = loadmat("q7.mat")
    data = mat["x"]
    labels = mat["y"]
    labels = np.reshape(labels, [-1])

    # DO NOT MODIFY
    # ----------
    mean_z1, mean_z2 = get_conditional_expectation(data)
    # return

    plt.figure()
    plt.scatter(mean_z1, mean_z2, c=labels)
    plt.xlabel("Z_1")
    plt.ylabel("Z_2")
    plt.colorbar()
    plt.grid()
    plt.savefig("a7", bbox_inches="tight")
    plt.show()
    plt.close()
    # ----------

    return

def load_model(model_file):
    """
    Loads a default Bayesian network with latent variables (in this case, a variational autoencoder)
    """

    with open("trained_mnist_model", "rb") as infile:
        if sys.version_info.major > 2:
            try:
                cpts = pkl.load(open("trained_mnist_model", "rb"), encoding="latin1")
            except ImportError:
                data = open("trained_mnist_model").read().replace("\r\n", "\n")
                cpts = pkl.loads(data.encode("latin1"), encoding="latin1")
        else:
            cpts = pkl.load(infile)

    model = {}
    model["prior_z1"] = cpts[0]
    model["prior_z2"] = cpts[1]
    model["cond_likelihood"] = cpts[2]

    return model


def cumlative(data: np.ndarray, p):
    if p > 0 and p <= print(data[-3]): return -3

def main():
    global disc_z1, disc_z2
    n_disc_z = 25
    disc_z1 = np.linspace(-3, 3, n_disc_z)
    disc_z2 = np.linspace(-3, 3, n_disc_z)

    global bayes_net
    """
    You can access the conditional probabilities by e.g. bayes_net['cond_likelihood'][0.25, -2.5], which 
    returns the probability of the pixels conditioned on z1=0.25, z2=-2.5
    """
    bayes_net = load_model("trained_mnist_model")

    """
    TODO: Using the above Bayesian Network model, complete the following parts.
    """ 

    # get_pixels_sampled_from_p_x_joint_z1_z2()
    # get an array of random numbers
    # rng = default_rng()
    # vals = rng.random(10)
    # print (vals)

    # print (bayes_net['prior_z1'])

    # pz1 = bayes_net['prior_z1']
    # print_z(pz1)

    # for i in range(len(pz1)):
    #     print (pz1[i])

    # print (bayes_net['cond_likelihood'][0.25, -2.5][0])
    # Your code should save a figure named a4
    # q4()

    # Your code should save a figure named a5
    # q5()

    # Your code should save two figures starting with a6
    # q6()

    # Your code should save a figure named a7
    q7()

    # print(np.random.uniform(0, 1, size=784))

    return


if __name__ == "__main__":
    main()
