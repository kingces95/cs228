"""
CS 228: Probabilistic Graphical Models
Winter 2019 (instructor: Stefano Ermon)
Starter Code for Part B
"""

from utils import *
import numpy as np
import math
import code
import sys
from scipy.stats import multivariate_normal

try:
    # https://github.com/scipy/scipy/blob/v0.14.0/scipy/misc/common.py
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

def estimate_phi_lambda(Z):
    """Perform MLE estimation of phi and lambda as described in B(i)
    Assumes that Y variables have been estimated using heuristic proposed in the question.
    Input:
        Z: A numpy array of size (N, M), where Z[i, j] = 0 or 1 indicating the party preference
    Output:
        MLE_phi: a real number, estimate of phi
        MLE_lambda: a real number, estimate of lambda
    Return as a dictionary {'phi': MLE_phi, 'lambda': MLE_lambda}

    This function will be autograded.
    """
    # TODO: Your code here!

    precincts = Z.shape[0]
    respondants_per_precinct = Z.shape[1]

    precinct_party = (np.sum(Z, axis=1) # party 1 voters by precinct
        / respondants_per_precinct # % party 1 voters by precinct
        > 0.5 # precinct with majority party 1 voters
    )

    MLE_phi = np.sum(
        precinct_party
    ) / precincts # MLE precinct has majority party 1 voters

    Y = np.zeros(Z.shape)
    for i in range(len(precinct_party)):
        Y[i,] = precinct_party[i]

    z = Z.flatten()
    y = Y.flatten()

    zy = np.column_stack((z, y)) # (voter party, precinct party)

    P_zy_match = len(zy[(zy[:,0] == zy[:,1])]) / len(zy)
    # P_z1_given_y1 = len(zy_11) / len(y_1)

    MLE_lambda = P_zy_match
    # code.interact(local=locals())

    # MLE_phi = .6
    # MLE_lambda = .93
    return {'phi': MLE_phi, 'lambda': MLE_lambda}

def compute_yz_marginal(X, params):
    """Evaluate log p(y_i=1|X) and log p(z_{ij}=1|X)

    Input:
        X: A numpy array of size (N, M, 2), where X[i, j] is the 2-dimensional vector
            representing the voter's properties
        params: A dictionary with the current parameters theta, elements include:
            phi: (float), P(precinct party) = 1
            lambda: (float), P(like-minded neighboors)
            mu0: size (2,) numpy array econding the estimate of the mean of class 0
            mu1: size (2,) numpy array econding the estimate of the mean of class 1
            sigma0: size (2,2) numpy array econding the estimate of the covariance of class 0
            sigma1: size (2,2) numpy array econding the estimate of the covariance of class 1
    Output:
        log_y_prob: An numpy array of size X.shape[0:1]; y_prob[i] store the value of log p(y_i=1|X, theta)
        log_z_prob: An numpy array of size X.shape[0:2]; z_prob[i, j] store the value of log p(z_{ij}=1|X, theta)

    You should use the log-sum-exp trick to avoid numerical overflow issues (Helper functions in utils.py)
    This function will be autograded.
    """
    log_y_prob, log_z_prob, _ = compute_yz_marginal_ex(X, params)

    return log_y_prob, log_z_prob
    
def compute_yz_marginal_ex(X, params):
    """Evaluate log p(y_i=1|X) and log p(z_{ij}=1|X)

    Input:
        X: A numpy array of size (N, M, 2), where X[i, j] is the 2-dimensional vector
            representing the voter's properties
        params: A dictionary with the current parameters theta, elements include:
            phi: (float), P(precinct party) = 1
            lambda: (float), P(like-minded neighboors)
            mu0: size (2,) numpy array econding the estimate of the mean of class 0
            mu1: size (2,) numpy array econding the estimate of the mean of class 1
            sigma0: size (2,2) numpy array econding the estimate of the covariance of class 0
            sigma1: size (2,2) numpy array econding the estimate of the covariance of class 1
    Output:
        log_y_prob: An numpy array of size X.shape[0:1]; y_prob[i] store the value of log p(y_i=1|X, theta)
        log_z_prob: An numpy array of size X.shape[0:2]; z_prob[i, j] store the value of log p(z_{ij}=1|X, theta)
        log_z_eq_y_prob: An numpy array of size X.shape[0:2]; z_prob[i, j] store the value of log p(z_{ij}=1|X, theta)

    You should use the log-sum-exp trick to avoid numerical overflow issues (Helper functions in utils.py)
    This function will be autograded.
    """
    N = X.shape[0] # 50; # of precincts; 0 <= i < N
    M = X.shape[1] # 20; # of people per precinct; 0 <= j < M

    log_y_prob = np.zeros((N))    
    log_z_prob = np.zeros((N,M))

    log_yz_join = compute_yz_joint(X, params)
    
    yz_join_table = log_yz_join.reshape(N * M, 4)

    y1z_join_table = logsumexp(yz_join_table[:,2:], axis=1)
    log_y_prob=logsumexp(y1z_join_table.reshape((N, M, 1)), axis=1).flatten() - np.log(20)

    yz1_join_table = logsumexp(yz_join_table[:,(1,3)], axis=1)
    log_z_prob=yz1_join_table.reshape((N, M))

    y_eq_z_join_table = logsumexp(yz_join_table[:,(0,3)], axis=1)
    log_z_eq_y_prob=y_eq_z_join_table.reshape((N, M))

    # code.interact(local=locals())

    return log_y_prob, log_z_prob, log_z_eq_y_prob

def compute_yz_joint(X, params):
    """ Compute the joint probability of log p(y_i, z_{ij}|X, params)
    Input:
        X: As usual
        params: A dictionary containing the old parameters, refer to compute compute_yz_marginal
    Output:
        log_yz_prob: A array of shape (X.shape[0], X.shape[1], 2, 2);
            u in {0,1}, v in {0,1}
            yz_prob[i, j, u, v] should store the value of log p(y_i=u, z_{ij}=v|X, params)
            Don't forget to normalize your (conditional) probability

    Note: To avoid numerical overflow, you should use log_sum_exp trick (Helper functions in utils.py)

    This function will be autograded.
    """

    N = X.shape[0] # 50; # of precincts; 0 <= i < N
    M = X.shape[1] # 20; # of people per precinct; 0 <= j < M

    log_yz_prob = compute_log_yz(X, params)

    # normalize
    cond_prob = log_yz_prob.reshape(M * N,4)
    cond_prob_sum = logsumexp(cond_prob,axis=1)
    cond_prob_norm = cond_prob - cond_prob_sum[:, np.newaxis]
    log_yz_normalized_prob=cond_prob_norm.reshape(log_yz_prob.shape)

    np.savetxt("log_yz.csv", log_yz_normalized_prob.reshape(M * N, 4), delimiter="\t")
    # code.interact(local=locals())

    return log_yz_normalized_prob

def compute_log_yz(X, params):
    """ Compute the joint probability of log p(y_i, z_{ij}|X, params)
    Input:
        X: As usual
        params: A dictionary containing the old parameters, refer to compute compute_yz_marginal
    Output:
        log_yz_prob: A array of shape (X.shape[0], X.shape[1], 2, 2);
            u in {0,1}, v in {0,1}
            yz_prob[i, j, u, v] should store the value of log p(y_i=u, z_{ij}=v|X, params)
            Don't forget to normalize your (conditional) probability

    Note: To avoid numerical overflow, you should use log_sum_exp trick (Helper functions in utils.py)

    This function will be autograded.
    """

    N = X.shape[0] # 50; # of precincts; 0 <= i < N
    M = X.shape[1] # 20; # of people per precinct; 0 <= j < M

    log_yz_prob = np.zeros((N, M, 2, 2))

    mu0 = params['mu0']
    cov0 = params['sigma0']
    mvn0 = multivariate_normal(mean=mu0, cov=cov0)

    mu1 = params['mu1']
    cov1 = params['sigma1']
    mvn1 = multivariate_normal(mean=mu1, cov=cov1)

    phi = params['phi'] # likelihood precinct is party 1
    lmbda = params['lambda'] # likelihood a person lives in likeminded precinct

    # for i in range(N):
    for i in range(50):
        if (i % 10 == 0):
            print("i=%d" % (i))

        for j in range(M):
            xij = X[i,j]
            mvn0_xij = mvn0.pdf(xij)
            mvn1_xij = mvn1.pdf(xij)
       
            for yi in range(2):
                y_prob=phi if yi == 1 else 1 - phi

                for zij in range(2):
                    ll = np.log(y_prob)
                    
                    z_given_y_prob = lmbda if yi == zij else 1 - lmbda
                    ll += np.log(z_given_y_prob) 
                    
                    x_given_z_prob = mvn1_xij if zij == 1 else mvn0_xij
                    ll += np.log(x_given_z_prob)

                    for k in range(M):
                        if (j == k):
                            continue

                        xik = X[i,k]
                        mvn0_xik = mvn0.pdf(xik)
                        mvn1_xik = mvn1.pdf(xik)
                        
                        sum_xik_zik_given_y_prob = np.zeros((2))

                        for zik in range(2):
                            zik_given_yi_prob = lmbda if yi == zik else 1 - lmbda
                            xik_given_zik_prob = mvn1_xik if zik == 1 else mvn0_xik

                            # zik_given_yi_prob * xik_given_zik_prob
                            sum_xik_zik_given_y_prob[zik] = \
                                np.log(zik_given_yi_prob) + np.log(xik_given_zik_prob)
                            
                        # code.interact(local=locals())
                        ll += logsumexp(sum_xik_zik_given_y_prob)

                    # print("(i,j,yi,zij)=(%d,%d,%d,%d) P(y)=%f * P(z_given_y)=%f * P(x_given_z)=%f = %f (%f)" % \
                    #     (i,j,yi,zij, y_prob, z_given_y_prob, x_given_z_prob, \
                    #         y_prob * z_given_y_prob * x_given_z_prob, np.exp(ll)))

                    log_yz_prob[i,j,yi,zij] = ll

            # code.interact(local=locals())

    return log_yz_prob

def em_step(X, params):
    """ Make one EM update according to question B(iii)
    Input:
        X: As usual
        params: A dictionary containing the old parameters, refer to compute compute_yz_marginal
    Output:
        new_params: A dictionary containing the new parameters

    This function will be autograded.
    """
    log_y_prob, log_z_prob, log_z_eq_y_prob = compute_yz_marginal_ex(X, params)

    y_count = np.exp(logsumexp(log_y_prob))
    phi = y_count / len(log_y_prob)

    N = X.shape[0] # 50; # of precincts; 0 <= i < N
    M = X.shape[1] # 20; # of people per precinct; 0 <= j < M

    log_z_eq_y_prob = log_z_eq_y_prob.flatten()
    lmbda = np.exp(logsumexp(log_z_eq_y_prob)) / len(log_z_eq_y_prob)

    z_prob = np.exp(log_z_prob)
    z = z_prob.reshape(1000,1)
    z[z > 1] = 1 # hack; unknown
    z_not = 1 - z

    z_sum = np.sum(z)
    z_not_sum = np.sum(z_not)

    x = X.reshape(1000,2) # (5,20,2) -> (100,2)
    
    print("min(z_not)=%d" % (np.min(z_not.reshape(1000,))))

    sigma0 = 0
    try:
        assert np.any(z_not.reshape(1000,) < 0) == False, "Negative aweight!"
        sigma0 = np.cov(x, rowvar=False, bias=True, aweights=z_not.reshape(1000,))
    except Exception as e:
        print("Sigma0 calculation threw an exception:")
        print(e)
    
    x_party0 = x * z_not # -> (float social, float economic) * P(z = 0)
    x_party0_social = x_party0[:,0] # -> (float social) * P(z = 0)
    x_party0_economic = x_party0[:,1] # -> (float economic) * P(z = 0)
    mu0 = np.array(
        (np.sum(x_party0_social) / z_not_sum, 
        np.sum(x_party0_economic) / z_not_sum)
    )

    print("min(z)=%d" % (np.min(z.reshape(1000,))))
    sigma1 = np.cov(x, rowvar=False, bias=True, aweights=z.reshape(1000,))
    
    x_party1 = x * z # -> (float social, float economic) * P(z = 1)
    x_party1_social = x_party1[:,0] # -> (float social) * P(z = 1)
    x_party1_economic = x_party1[:,1] # -> (float economic) * P(z = 1)
    mu1 = np.array(
        (np.sum(x_party1_social) / z_sum, 
        np.sum(x_party1_economic) / z_sum)
    )

    result = {
        'phi': phi, \
        'lambda': lmbda, \
        'mu0': mu0, 'sigma0': sigma0, \
        'mu1': mu1, 'sigma1': sigma1  \
    }

    print(result)
    return result

def compute_log_likelihood(X, params):
    """ Compute the log likelihood log p(X) under current parameters.
    To compute this you can first call the function compute_yz_joint

    Input:
        X: As usual
        params: As in the description for compute_yz_joint
    Output: A real number representing log p(X)

    This function will be autograded
    """

    log_yz_prob = compute_log_yz(X, params)
    ll = logsumexp(log_yz_prob.flatten())
    print("ll=%f" % ll)

    return ll


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # Read data
    X_labeled, Z_labeled = read_labeled_matrix()
    X_unlabeled = read_unlabeled_matrix()

    # Question B(i)
    from part_a import estimate_params
    params = estimate_params(X_labeled, Z_labeled)
    params.update(estimate_phi_lambda(Z_labeled))

    colorprint("MLE estimates for PA part b.i:", "teal")
    colorprint("\tMLE phi: %s\n\tMLE lambda: %s\n"%(params['phi'], params['lambda']), 'red')

    # Question B(ii)
    # log_y_prob, log_z_prob = compute_yz_marginal(X_unlabeled, params)   # Get the log probability of y and z conditioned on x
    # colorprint("Your predicted party preference:", "teal")
    # colorprint(str(np.exp(log_y_prob)), 'red')

    # plt.scatter(X_unlabeled[:, :, 0].flatten(), X_unlabeled[:, :, 1].flatten(),
    #             c=np.array(['red', 'blue'])[(log_z_prob > np.log(0.5)).astype(np.int).flatten()], marker='+')
    # plt.scatter(params['mu0'][0], params['mu0'][1], c='black', marker='d')
    # plt.scatter(params['mu1'][0], params['mu1'][1], c='black', marker='d')
    # plt.show()

    # Question B(iii)
    param_init = [params.copy(), get_random_params(), get_random_params()]
    param_final = []
    loglikelihood_final = []
    for params in param_init:
        loglikelihoods = []
        for i in range(10):
            loglikelihoods.append(compute_log_likelihood(X_unlabeled, params))
            params = em_step(X_unlabeled, params)
            sys.exit()


        param_final.append(params.copy())
        loglikelihood_final.append(loglikelihoods[-1])
        plt.plot(loglikelihoods)
    plt.legend(['MLE initialization', 'Random initialization', 'Random initialization'])
    # plt.show()
    params = param_final[np.argmax(loglikelihood_final)]
    colorprint("MLE estimates for PA part b.iii:", "teal")
    colorprint("\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s\n\tphi: %s\n\tlambda: %s\n"
            % (params['mu0'], params['mu1'], params['sigma0'], params['sigma1'], params['phi'], params['lambda']), "red")

    # Question B(iv)
    log_y_prob, log_z_prob = compute_yz_marginal(X_unlabeled, params)
    colorprint("Your predicted party preference:", "teal")
    colorprint(str(np.exp(log_y_prob)), 'red')
    plt.scatter(X_unlabeled[:, :, 0].flatten(), X_unlabeled[:, :, 1].flatten(),
                c=np.array(['red', 'blue'])[(log_z_prob > np.log(0.5)).astype(np.int).flatten()], marker='+')
    plt.scatter(params['mu0'][0], params['mu0'][1], c='black', marker='d')
    plt.scatter(params['mu1'][0], params['mu1'][1], c='black', marker='d')
    # plt.show()
