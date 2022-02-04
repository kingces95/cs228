###############################################################################
# factor graph data structure implementation
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018
###############################################################################

from factors import *
import numpy as np

class FactorGraph:
    def __init__(self, numVar=0, numFactor=0):
        '''
        Args
        - numVar: int, number of variables in the factor graph
        - numFactor: int, number of factors in the factor graph

        Creates the following properties of the FactorGraph:
        - var: list of indices/names of variables
        - domain: list of lists, domain[i] is the domain of the i-th variable,
            all the domains are [0,1] for this programming assignment
        - varToFactor: list of lists, varToFactor[i] is a list of the indices
            of Factors that are connected to variable i
        - factorToVar: list of lists, factorToVar[i] is a list of the indices
            of variables that are connected to factor i
        - factors: list of Factors
        - messagesVarToFactor: dict, stores messages from variables to factors,
            keys are (src, dst), values are corresponding messages of type Factor
        - messagesFactorToVar: dict, stores messages from factors to variables,
            keys are (src, dst), values are corresponding messages of type Factor
        '''
        self.var = [None for _ in range(numVar)]
        self.domain = [[0,1] for _ in range(numVar)]
        self.varToFactor = [[] for _ in range(numVar)]
        self.factorToVar = [[] for _ in range(numFactor)]
        self.factors = []
        self.messagesVarToFactor = {}
        self.messagesFactorToVar = {}

    def evaluateWeight(self, assignment):
        '''
        param - assignment: the full assignment of all the variables
        return: the multiplication of all the factors' values for this assigments
        '''
        a = np.array(assignment, copy=False)
        output = 1.0
        for f in self.factors:
            output *= f.val.flat[assignment_to_indices([a[f.scope]], f.card)]
        return output[0]

    def getInMessage(self, src, dst, type="varToFactor"):
        '''
        If a message from src to dst already exists, returns it. Otherwise,
        initializes a uniform (normalized) message.

        Args
        - src: int, index of the source factor/clique
        - dst: int, index of the destination factor/clique
        - type: str, one of ["varToFactor", "factorToVar"]
            - "varToFactor": if message is from a variable to a factor
            - "factorToVar": if message is from a factor to a variable

        Returns: Factor, represents a message from src to dst
        '''
        if type == "varToFactor":
            if (src, dst) not in self.messagesVarToFactor:
                inMsg = Factor()
                inMsg.scope = [src]
                inMsg.card = [len(self.domain[src])]
                inMsg.val = np.ones(inMsg.card) / inMsg.card[0]
                self.messagesVarToFactor[(src, dst)] = inMsg
            return self.messagesVarToFactor[(src, dst)]

        elif type == "factorToVar":
            if (src, dst) not in self.messagesFactorToVar:
                inMsg = Factor()
                inMsg.scope = [dst]
                inMsg.card = [len(self.domain[dst])]
                inMsg.val = np.ones(inMsg.card) / inMsg.card[0]
                self.messagesFactorToVar[(src, dst)] = inMsg
            return self.messagesFactorToVar[(src, dst)]

        else:
            raise ValueError("Unknown type: {}".format(type))

    def runParallelLoopyBP(self, iterations):
        '''
        Implements the Loopy BP algorithm. The only values you should update in
        this function are self.messagesVarToFactor and self.messagesFactorToVar.

        Args
        - iterations: the number of iterations you do Loopy BP

        Warning: Don't forget to normalize the messages (factors) before 
        	multiplying! You may find the Factor.normalize() method useful.

        THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
        '''
        #######################################################################
        # To do: your code here


        raise NotImplementedError()
        #######################################################################
        pass

    def estimateMarginalProbability(self, var):
        '''
        Estimate the marginal probabilities of a single variable after running
        loopy belief propogation. This method assumes runParallelLoopyBP has
        been run.

        Args
        - var: int, index of a single variable

        Returns: np.array, shape [2], the marginal probabilities that the
            variable takes the values 0 and 1

        example:
        >>> factor_graph.estimateMarginalProbability(0)
        >>> [0.2, 0.8]

        THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
        '''
        #######################################################################
        # To do: your code here

        #######################################################################
        pass

    def getMarginalMAP(self):
        '''
        Computes the marginal MAP assignments for the variables. You may
        utilize the method estimateMarginalProbability.

        Returns: np.array, shape [num_var], marginal MAP assignment for each
            variable

        example: (N=2, 2*N=4)
        >>> factor_graph.getMarginalMAP()
        >>> [0, 1, 0, 0]
        '''
        output = np.zeros(len(self.var))
        #######################################################################
        # To do: your code here

        #######################################################################
        return output
