###############################################################################
# factor graph data structure implementation
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018
###############################################################################

from factors import *
import numpy as np
import code
import traceback

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
            # if f.name == "6 (parity)":
            #     code.interact(local=locals())
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

        numVar=self.var.__len__()
        numFactors=self.factors.__len__()

        # initialize variable to factor messages to 50/50
        for varId in range(numVar):
            for factorId in self.varToFactor[varId]:
                self.messagesVarToFactor[(varId, factorId)]=Factor( \
                    scope=[varId], card=[2], val=np.array([0.5, 0.5]))

        # code.interact(local=locals())
        self.runParallelLoopyBPi(iterations)

    def runParallelLoopyBPi(self, iterations):
        numVar=self.var.__len__()
        numFactors=self.factors.__len__()

        for i in range(iterations):

            #if i % 5 == 0:
            #print(i)
            codeword=[]

            # for each factor, send a message to adjacent variables
            for factorId in range(numFactors):
                factor=self.factors[factorId]

                # for each variable adjacent to this factor
                for recipientVarId in self.factorToVar[factorId]:

                    # multiply factor with all inbound messages
                    product=factor

                    # for each variable adjacent to this factor (except for recipient)
                    for varId in self.factorToVar[factorId]:

                        if recipientVarId == varId:
                            continue

                        # accumulate messages to sent this factor
                        product=product.multiply(self.messagesVarToFactor[(varId, factorId)]).normalize()

                    # pivot on variable
                    marginalized_product=product.marginalize_all_but([recipientVarId]).normalize()

                    # send normalized message
                    self.messagesFactorToVar[(factorId, recipientVarId)]=marginalized_product.normalize()

            # code.interact(local=locals())

            # for each variable, send a message to adjacent factors
            for varId in range(numVar):

                # for each factor adjacent to this variable
                for recipientFactorId in self.varToFactor[varId]:

                    # multiply all inbound messages
                    product=None

                    # for each factor adjacent to this variable (except for recipient)
                    for factorId in self.varToFactor[varId]:

                        if factorId == recipientFactorId:
                            continue

                        # receive message from factor sent to this variable
                        inbound_message=self.messagesFactorToVar[(factorId, varId)].normalize()

                        # accumulate messages to this variable
                        product=inbound_message if product == None else product.multiply(inbound_message).normalize()

                    # send normalized message
                    self.messagesVarToFactor[(varId, recipientFactorId)]=product.normalize()
                    
                # codeword.append(self.estimateMarginalProbability(varId)[0])
            
            # yRecovered=self.getMessage()
            # print(np.sum([yRecovered==1]))
            # code.interact(local=locals())

            # print(codeword)

                

    def estimateMarginalProbability(self, varId):
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

        # multiply all inbound messages
        product=None

        # for each factor adjacent to this variable
        for factorId in self.varToFactor[varId]:

            # receive message from factor sent to this variable
            message=self.messagesFactorToVar[(factorId, varId)]

            # accumulate messages to this variable
            product=message if product == None else product.multiply(message).normalize()

        # code.interact(local=locals())
        product.normalize()
            
        return product.val

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
        assert False, "> getMarginalMAP"
        output = np.zeros(len(self.var))
        #######################################################################
        # To do: your code here

        #######################################################################
        return output

    def getMessage(self):
        numVar=self.var.__len__()

        x=[]
        for i in range(numVar):
            x.append(0 if self.estimateMarginalProbability(i)[0] > .5 else 1)
        x=np.array(x)
        return x

