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
import sys
import pickle as pkl

import numpy as np

import matplotlib.pyplot as plt

from scipy.io import loadmat

try:
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


def get_conditional_expectation(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    global n_disc_z, x_cardinality

    n = len(data)
    x_cardinality = 784
    n_disc_z = 25

    cond_likelihood = bayes_net['cond_likelihood']

    z1_distribution = np.zeros(n_disc_z)
    for i in range(n_disc_z):
        z1_distribution[i] = bayes_net['prior_z1'][disc_z1[i]]

    z2_distribution = np.zeros(n_disc_z)
    for i in range(n_disc_z):
        z2_distribution[i]= bayes_net['prior_z2'][disc_z2[i]]

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

    cond_e_z_1=np.zeros(n)
    cond_e_z_2=np.zeros(n)

    for i in range(n):
    #for i in range(1):
        image=data[i]
        pz1_px_z1=np.zeros(n_disc_z)
        pz2_px_z2=np.zeros(n_disc_z)
        for zi in range(n_disc_z):
            pz1 = z1_distribution[zi]
            pz2 = z2_distribution[zi]
       
            pz1_px_z1[zi]=pz1 * np.prod(pixel_to_p(cond_likelihood_z1[zi], image))
            pz2_px_z2[zi]=pz2 * np.prod(pixel_to_p(cond_likelihood_z2[zi], image))          

        pz1_px_z1_sum=np.sum(pz1_px_z1)
        pz2_px_z2_sum=np.sum(pz2_px_z2)

        cond_e_z_1[i]=np.sum(disc_z1 * (pz1_px_z1 / pz1_px_z1_sum))
        cond_e_z_2[i]=np.sum(disc_z2 * (pz2_px_z2 / pz2_px_z2_sum))

    return cond_e_z_1, cond_e_z_2

def pixel_to_p(p_of_pixel, image):
    return image * p_of_pixel + (1 - image) * (1 - p_of_pixel)

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
    # print(labels)
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
