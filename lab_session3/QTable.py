import copy
import numpy as np

class QTable:
    def __init__(self, nrStates, actions, learnRate, discountRate):
        # initialise qtable with zeroes
        #self.q_values = dict.fromkeys(range(nrStates), copy.deepcopy([0] * nrActions)) #don't use lists are not copies!
        self.nrStates = nrStates
        self.actions = actions
        init = [0.00]*len(actions)
        self.q_values = {}
        for i in range(nrStates):
            self.q_values[i] = copy.deepcopy(init)
        self.learnRate = learnRate
        self.discountRate = discountRate

    def __str__(self):
        string = ''
        for entry in self.q_values:
            if entry < 10:
                string += str(entry) + '  :  ' + str(self.q_values[entry]) + '\n'
            elif 10 <= entry < 100:
                string += str(entry) + ' :  ' + str(self.q_values[entry]) + '\n'
            else:
                string += str(entry) + ':  ' + str(self.q_values[entry]) + '\n'

        string += '---------------------------------------------------\n'
        return string

    def resetQvalues(self):
        init = [0.00] * self.nrActions
        self.q_values = {}
        for i in range(self.nrStates):
            self.q_values[i] = copy.deepcopy(init)

    def getValue(self, state, action):
        return self.q_values[state][action]

    def getAction(self, state):
        return self.actions[self.q_values[state][:].index(max(self.q_values[state][:]))]

    # using Bellman equation
    def updateValue(self, state, action, new_state, reward):
        # Bellman equation
        fromState = self.q_values[state][action]
        maxState = np.max(self.q_values[new_state][:])
        self.q_values[state][action] = fromState + (self.learnRate * (reward + (self.discountRate * maxState) - fromState))
