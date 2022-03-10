"""
CS 228: Probabilistic Graphical Models
Instructor: Stefano Ermon
Starter Code for Part A
"""

from utils import *
import numpy as np
import code
import math
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

def estimate_params(X, Z):
    """Perform MLE estimation of model 1 parameters.

    Input:
        X: A numpy array of size (N, M, 2), where X[i, j] is the 2-dimensional vector
            representing the voter's properties
        Z: A numpy array of size (N, M), where Z[i, j] = 0 or 1 indicating the party preference

    Output: A dictionary with the following keys/values
        pi: (float), estimate of party proportions
        mu0: size (2,) numpy array econding the estimate of the mean of class 0
        mu1: size (2,) numpy array econding the estimate of the mean of class 1
        sigma0: size (2,2) numpy array econding the estimate of the covariance of class 0
        sigma1: size (2,2) numpy array econding the estimate of the covariance of class 1

    This function will be autograded.

    Note: your algorithm should work for any value of N and M
    """
    
    z = Z.flatten()
    z_sum = np.sum(z)
    z_length = len(z)
    pi = z_sum / z_length

    x = X.reshape(100,2) # (5,20,2) -> (100,2)
    zx = np.column_stack((z,x)) # -> (bool party, float social, float economic)
    
    zx_party0 = zx[zx[:,0] == 0] # -> (bool party = 0, float social, float economic)
    x_party0 = zx_party0[:,[1,2]] # -> (float social, float economic) for party 0
    sigma0 = np.cov(x_party0, rowvar=False, bias=True)
    
    x_party0_social = x_party0[:,0] # -> (float social) for party = 0
    x_party0_economic = x_party0[:,1] # -> (float economic) for party = 0
    mu0 = np.array(
        (np.average(x_party0_social), 
        np.average(x_party0_economic))
    )

    zx_party1 = zx[zx[:,0] == 1] # -> (bool party = 1, float social, float economic)
    x_party1 = zx_party1[:,[1,2]] # -> (float social, float economic) for party 1
    sigma1 = np.cov(x_party1, rowvar=False, bias=True)
    
    x_party1_social = x_party1[:,0] # -> (float social) for party = 1
    x_party1_economic = x_party1[:,1] # -> (float economic) for party = 1
    mu1 = np.array(
        (np.average(x_party1_social), 
        np.average(x_party1_economic))
    )

    # code.interact(local=locals())
    return {'pi': pi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}

def em_update(X, params):
    """ Perform one EM update based on unlabeled data X
    Input:
        X: A numpy array of size (N, M, 2), where X[i, j] is the 2-dimensional vector
            representing the voter's properties
        params: A dictionary, the previous parameters, see the description in estimate_params
    Output: The updated parameter. The output format is identical to estimate_params

    This function will be autograded.

    Note: You will most likely need to use the function estimate_z_prob_given_x
    """
    # TODO: Your code here!

    Z = estimate_z_prob_given_x(X, params) # hallucinate values
    # code.interact(local=locals())

    z = Z.flatten().reshape(1000,1)
    z_not = 1 - z

    z_sum = np.sum(z)
    z_not_sum = np.sum(z_not)

    pi = z_sum / len(z)

    x = X.reshape(1000,2) # (5,20,2) -> (100,2)
    # code.interact(local=locals())
    
    sigma0 = np.cov(x, rowvar=False, bias=True, aweights=z_not.reshape(1000,))
    
    x_party0 = x * z_not # -> (float social, float economic) * P(z = 0)
    x_party0_social = x_party0[:,0] # -> (float social) * P(z = 0)
    x_party0_economic = x_party0[:,1] # -> (float economic) * P(z = 0)
    mu0 = np.array(
        (np.sum(x_party0_social) / z_not_sum, 
        np.sum(x_party0_economic) / z_not_sum)
    )

    sigma1 = np.cov(x, rowvar=False, bias=True, aweights=z.reshape(1000,))
    
    x_party1 = x * z # -> (float social, float economic) * P(z = 1)
    x_party1_social = x_party1[:,0] # -> (float social) * P(z = 1)
    x_party1_economic = x_party1[:,1] # -> (float economic) * P(z = 1)
    mu1 = np.array(
        (np.sum(x_party1_social) / z_sum, 
        np.sum(x_party1_economic) / z_sum)
    )

    # code.interact(local=locals())
    return {'pi': pi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}

def estimate_z_prob_given_x(X, params):
    """ Estimate p(z_{ij} = 1|x_{ij}, theta)
    Input:
        X: Identical to the function em_update
        params: Identical to the function em_update
    Output: A 2D numpy array z_prob with the same size as X.shape[0:2],
            z_prob[i, j] should store the value of p(z_{ij} = 1|x_{ij}, theta)
            Note: it should be a normalized probability

    This function will be autograded.
    """

    pi0 = 1 - params['pi']
    mu0 = params['mu0']
    cov0 = params['sigma0']
    mvn0 = multivariate_normal(mean=mu0, cov=cov0)

    pi1 = params['pi']
    mu1 = params['mu1']
    cov1 = params['sigma1']
    mvn1 = multivariate_normal(mean=mu1, cov=cov1)

    z_prob = np.zeros(shape=X.shape[0:2])

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xij = X[i,j]
            numerator = pi1 * mvn1.pdf(xij)
            denominator = pi0 * mvn0.pdf(xij) + pi1 * mvn1.pdf(xij)
            z_prob[i, j] = numerator / denominator
            
    # code.interact(local=locals())

    return z_prob

def compute_log_likelihood(X, params):
    """ Estimate the log-likelihood of the entire data log p(X|theta)
    Input:
        X: Identical to the function em_update
        params: Identical to the function em_update
    Output A real number representing the log likelihood

    This function will be autograded.
    """
    # code.interact(local=locals())
    x = X.reshape(1000,2) # (50,20,2) -> (1000,2)

    pi0 = 1 - params['pi']
    mu0 = params['mu0']
    cov0 = params['sigma0']
    mvn0 = multivariate_normal(mean=mu0, cov=cov0)

    pi1 = params['pi']
    mu1 = params['mu1']
    cov1 = params['sigma1']
    mvn1 = multivariate_normal(mean=mu1, cov=cov1)

    loglikelihood = 0.0
    for i in range(x.shape[0]):
        loglikelihood += np.log(pi0 * mvn0.pdf(x[i]) + pi1 * mvn1.pdf(x[i]))

    # code.interact(local=locals())
    return loglikelihood

if __name__ == '__main__':
    #===============================================================================
    # This runs the functions that you have defined to produce the answers to the
    # assignment problems
    #===============================================================================

    # Read data
    X_labeled, Z_labeled = read_labeled_matrix(LABELED_FILE)
    X_unlabeled = read_unlabeled_matrix(UNLABELED_FILE)

    # pt a.i
    params = estimate_params(X_labeled, Z_labeled)

    colorprint("MLE estimates for PA part a.i:", "teal")
    colorprint("\tpi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
        %(params['pi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1']), "red")

    # pt a.ii

    params = estimate_params(X_labeled, Z_labeled)  # Initialize 1
    loglikelihoods = []
    while True:
        loglikelihoods.append(compute_log_likelihood(X_unlabeled, params))
        if len(loglikelihoods) > 2 and loglikelihoods[-1] - loglikelihoods[-2] < 0.01:
            break
        params = em_update(X_unlabeled, params)

    colorprint("MLE estimates for PA part a.ii.1:", "teal")
    colorprint("\tpi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
        %(params['pi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1']), "red")

    print(loglikelihoods)
    plt.plot(loglikelihoods)
    plt.show()

    # ----
    X_labeled_0, Z_labeled_0 = read_labeled_matrix(LABELED_FILE_0)
    params = estimate_params(X_labeled_0, Z_labeled_0)  # Initialize 2
    loglikelihoods = []
    while True:
        loglikelihoods.append(compute_log_likelihood(X_unlabeled, params))
        if len(loglikelihoods) > 2 and loglikelihoods[-1] - loglikelihoods[-2] < 0.01:
            break
        params = em_update(X_unlabeled, params)

    colorprint("MLE estimates for PA part a.ii.2:", "teal")
    colorprint("\tpi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
        %(params['pi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1']), "red")

    print(loglikelihoods)
    plt.plot(loglikelihoods)
    plt.show()

    # ----
    X_labeled_1, Z_labeled_1 = read_labeled_matrix(LABELED_FILE_1)
    params = estimate_params(X_labeled_1, Z_labeled_1)  # Initialize 2
    loglikelihoods = []
    while True:
        loglikelihoods.append(compute_log_likelihood(X_unlabeled, params))
        if len(loglikelihoods) > 2 and loglikelihoods[-1] - loglikelihoods[-2] < 0.01:
            break
        params = em_update(X_unlabeled, params)

    colorprint("MLE estimates for PA part a.ii.3:", "teal")
    colorprint("\tpi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
        %(params['pi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1']), "red")

    print(loglikelihoods)
    plt.plot(loglikelihoods)
    plt.show()
