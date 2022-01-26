import os
import sys
import numpy as np
from collections import Counter
import random
import code

try:
    from scipy.misc import logsumexp
except:
    from scipy.special import logsumexp

# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry
# helpers to learn and traverse the tree over attributes
from tree import get_mst, get_tree_root, get_tree_edges, renormalize

# pseudocounts for uniform dirichlet prior (i.e. Laplace smoothing)
alpha = 0.1


# --------------------------------------------------------------------------
# Naive bayes CPT and classifier
# --------------------------------------------------------------------------


class NBCPT(object):
    '''
    NB Conditional Probability Table (CPT) for a child attribute.  Each child
    has only the class variable as a parent.
    NBCPT is just a helper class to implement NBClassifier. You do not
    have to implement and call NBCPT, if you think you have a better solution
    for NBClassifier without NBCPT.
    This class won't be graded in the autograder.
    '''

    # A_i: the bill identifier
    def __init__(self, x: np.ndarray):
        '''
        TODO: create any persistent instance variables you need that hold the
        state of the learned parameters for this CPT

        Params:
          - x: records of members votes on a single bill
                n x 2 matrix where rows = n members, columns = records (C, A_n)
        '''

        # count rows whose first column is 0; count the republicans
        republicans=x[x[:,0] == 0].shape[0]

        # count rows whose first column is 0 and second column 1
        # count the republicans who voted for the bill
        republican_votes=x[(x[:,0] == 0) & (x[:,1] == 1)].shape[0]

        self.aye_given_republican = (republican_votes + alpha)/(republicans + alpha * 2)


        # count rows whose first column is 1; count the republicans
        democrats=x[x[:,0] == 1].shape[0]

        # count rows whose first column is 1 and second column 1
        # count the democrats who voted for the bill
        democrat_votes=x[(x[:,0] == 1) & (x[:,1] == 1)].shape[0]

        self.aye_given_democrat = (democrat_votes + alpha)/(democrats + alpha * 2)

    def get_cond_prob(self, vote, party):
        '''
        TODO: return the conditional probability vote given the party

        Params:
         - vote: 0 is nay, 1 is aye
         - party: 0 is republican, 1 is democrat
        Returns:
         - p: a scalar, the conditional probability P(vote|party)
        '''

        is_republican = party == 0

        if (vote == 1):
            if (is_republican):
                return self.aye_given_republican
            else:
                return self.aye_given_democrat
        else:
            if (is_republican):
                return 1 - self.aye_given_republican
            else:
                return 1 - self.aye_given_democrat

class NBClassifier(object):
    '''
    NB classifier class specification.
    '''

    def __init__(self, A_train, C_train):
        '''
        TODO: create any persistent instance variables you need that hold the
        state of the trained classifier and populate them with a call to self._train
        Suggestions for the attributes in the classifier:
            - self.logP_republican: log-probability a congressperson is a republican
            - self.logP_democrat: log-probability a congressperson is a democrat
            - self.conditional_probability_table: conditional probability tables
        '''
        self.logP_republican = np.log(
            # count of rows whose first column is 0
            C_train[C_train == 0].size / C_train.size
        )

        self.logP_democrat = np.log(
            # count of rows whose first column is 1
            C_train[C_train == 1].size / C_train.size
        )

        assert np.exp(self.logP_democrat) + np.exp(self.logP_republican) == 1.

        self.conditional_probability_table=[]

        bill_count=A_train.shape[1]
        for bill_id in range(bill_count):            
            self.conditional_probability_table.append(
                NBCPT(
                    # rows and rows of (party, vote on bill i)
                    np.array((C_train, A_train[:,bill_id])).T
                )
            )

        # code.interact(local=locals())

    def classify(self, entry):
        '''
        TODO: return the log probabilites for class == 0 or class == 1 as a
        tuple for the given entry

        Params:
          - entry: full assignment of variables
            e.g. entry = np.array([0,1,1,...]) means variable A_0 = 0, A_1 = 1, A_2 = 1, etc.
        Returns:
         - c_pred: the predicted label, one of {0, 1}
         - logP_c_pred: the log of the conditional probability of the label |c_pred|
        '''

        logP_republican=self.logP_republican
        logP_democrat=self.logP_democrat
        for i in range(entry.size):
            logP_republican += np.log(self.conditional_probability_table[i].get_cond_prob(entry[i], 0))
            logP_democrat += np.log(self.conditional_probability_table[i].get_cond_prob(entry[i], 1))

        if (logP_republican > logP_democrat):
            return (0, logP_republican)
        else:
            return (1, logP_democrat)

# --------------------------------------------------------------------------
# TANB CPT and classifier
# --------------------------------------------------------------------------
class TANBCPT(object):
    '''
    TANB CPT for a child attribute.  Each child can have one other attribute
    parent (or none in the case of the root), and the class variable as a
    parent.
    TANBCPT is just a helper class to implement TANBClassifier. You do not
    have to implement and call TANBCPT, if you think you have a better solution
    for TANBClassifier without TANBCPT.
    This class won't be graded in the autograder.
    '''

    def __init__(self, classifier, x: np.ndarray):
        '''
        TODO: create any persistent instance variables you need that hold the
        state of the learned parameters for this CPT

        Params:
         - A_i: the index of the child variable
         - A_p: the index of its parent variable (in the Chow-Liu algorithm,
           the learned structure will have up to a single parent for each child)

        '''
        pass

    def learn(self, A, C):
        '''
        TODO: populate any instance variables specified in __init__ we need to learn
        the parameters for this CPT

        Params:
         - A: a (n,k) numpy array where each row is a sample of assignments
         - C: a (n,)  numpy array where the elements correspond to the class
           labels of the rows in A
        Returns:
         - None

        '''
        pass

    def get_cond_prob(self, entry, c):
        '''
        TODO: return the conditional probability P(A_i | Pa(A_i)) for the values
        specified in the example entry and class label c
        Note: in the expression above, the class C is also a parent of A_i!

        Params;
            - entry: full assignment of variables
              e.g. entry = np.array([0,1,1,...]) means variable A_0 = 0, A_1 = 1, A_2 = 1, etc.
            - c: the class
        Returns:
         - p: a scalar, the conditional probability P(A_i | Pa(A_i))

        '''
        pass


class TANBClassifier(NBClassifier):
    '''
    TANB classifier class specification
    '''

    def __init__(self, A_train, C_train):
        '''
        TODO: create any persistent instance variables you need that hold the
        state of the trained classifier and populate them with a call to self._train

        Params:
          - A_train: a (n,k) numpy array where each row is a sample of assignments
          - C_train: a (n,)  numpy array where the elements correspond to
            the class labels of the rows in A

        '''
        mst=get_mst(A_train, C_train)
        root=get_tree_root(A_train, C_train)
        edges=get_tree_edges(mst, root)

        code.interact(local=locals())

        raise NotImplementedError()

    def _train(self, A_train, C_train):
        '''
        TODO: train your TANB classifier with the specified data and class labels
        hint: learn the parameters for the required CPTs
        hint: you will want to look through and call functions imported from tree.py:
            - get_mst(): build the mst from input data
            - get_tree_root(): get the root of a given mst
            - get_tree_edges(): iterate over all edges in the rooted tree.
              each edge (a,b) => a -> b

        Params:
          - A_train: a (n,k) numpy array where each row is a sample of assignments
          - C_train: a (n,)  numpy array where the elements correspond to
            the class labels of the rows in A
        Returns:
         - None

        '''
        raise NotImplementedError()

    def classify(self, entry):
        '''
        TODO: return the log probabilites for class == 0 and class == 1 as a
        tuple for the given entry

        Params:
         - entry: full assignment of variables
            e.g. entry = np.array([0,1,1,...]) means variable A_0 = 0, A_1 = 1, A_2 = 1, etc.
        Returns:
         - c_pred: the predicted label in {0, 1}
         - logP_c_pred: the log conditional probability of predicting the label |c_pred|

        NOTE: this class inherits from NBClassifier, and optionally, it is possible to
        write this method in NBClassifier, such that this implementation can
        be removed.

        '''
        raise NotImplementedError()


# =========================================================================


# load all data
A_base, C_base = load_vote_data()


def evaluate(classifier_cls, train_subset=False):
    '''
    =======* DO NOT MODIFY this function *=======

    evaluate the classifier specified by classifier_cls using 10-fold cross
    validation
    Params:
     - classifier_cls: either NBClassifier or TANBClassifier
     - train_subset: train the classifier on a smaller subset of the training
      data
    Returns:
     - accuracy as a proportion
     - total number of predicted samples

    '''
    global A_base, C_base

    A, C = A_base, C_base

    # score classifier on specified attributes, A, against provided labels, C
    def get_classification_results(classifier, A, C):
        results = []
        pp = []
        for entry, c in zip(A, C):
            c_pred, _ = classifier.classify(entry)
            results.append((c_pred == c))
            pp.append(_)
        return results

    # partition train and test set for 10 rounds
    M, N = A.shape
    tot_correct = 0
    tot_test = 0
    step = M // 10
    for holdout_round, i in enumerate(range(0, M, step)):
        A_train = np.vstack([A[0:i, :], A[i+step:, :]])
        C_train = np.hstack([C[0:i], C[i+step:]])
        A_test = A[i:i+step, :]
        C_test = C[i:i+step]
        if train_subset:
            A_train = A_train[:16, :]
            C_train = C_train[:16]

        # train the classifiers
        classifier = classifier_cls(A_train, C_train)

        train_results = get_classification_results(
            classifier, A_train, C_train)
        test_results = get_classification_results(classifier, A_test, C_test)
        tot_correct += sum(test_results)
        tot_test += len(test_results)

    return 1.*tot_correct/tot_test, tot_test


def evaluate_incomplete_entry(classifier_cls):
    '''
    TODO: Fill out the function to compute marginal probabilities.

    Params:
     - classifier_cls: either NBClassifier or TANBClassifier
     - train_subset: train the classifier on a smaller subset of the training
      data
    Returns:
     - P_c_pred: P(C = 1 | A_observed) as a scalar.
     - PA_12_eq_1: P(A_12 = 1 | A_observed) as a scalar.

     Hint: Since the index starts from 0 in the code,
     A_12 should correspond to "entry[11]" in the implementation.

    '''
    global A_base, C_base

    # train a TANB classifier on the full dataset
    classifier = classifier_cls(A_base, C_base)

    # load incomplete entry 1
    entry = load_incomplete_entry()

    c_pred, logP_c_pred = classifier.classify(entry)
    P_c_pred = np.exp(logP_c_pred)
    print('  P(C={}|A_observed) = {:2.4f}'.format(c_pred, P_c_pred))

    # TODO: write code to compute this!
    PA_12_eq_1 = None

    return P_c_pred, PA_12_eq_1


def main():
    '''
    (optional) TODO: modify or add calls to evaluate your implemented classifiers.
    '''

    # Part (a)
    print('Naive Bayes')
    accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
    print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
        accuracy, num_examples))

    # Part (b)
    # print('TANB Classifier')
    # accuracy, num_examples = evaluate(TANBClassifier, train_subset=False)
    # print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    #     accuracy, num_examples))

    # # Part (c)
    # print('Naive Bayes Classifier on missing data')
    # evaluate_incomplete_entry(NBClassifier)

    # print('TANB Classifier on missing data')
    # evaluate_incomplete_entry(TANBClassifier)

    # # Part (d)
    # print('Naive Bayes')
    # accuracy, num_examples = evaluate(NBClassifier, train_subset=True)
    # print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    #     accuracy, num_examples))

    # print('TANB Classifier')
    # accuracy, num_examples = evaluate(TANBClassifier, train_subset=True)
    # print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    #     accuracy, num_examples))


if __name__ == '__main__':
    main()
