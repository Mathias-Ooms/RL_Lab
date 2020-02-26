"""
Module containing the agent classes to solve a Bandit problem.

Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
An example can be seen on the Bandit_Agent and Random_Agent classes.
"""
# -*- coding: utf-8 -*-
import numpy as np
from bandits.utils import softmax, my_random_choice

class Bandit_Agent(object):
    """
    Abstract Agent to solve a Bandit problem.

    Contains the methods learn() and act() for the base life cycle of an agent.
    The reset() method reinitializes the agent.
    The minimum requirement to instantiate a child class of Bandit_Agent
    is that it implements the act() method (see Random_Agent).
    """
    def __init__(self, k:int, **kwargs):
        """
        Simply stores the number of arms of the Bandit problem.
        The __init__() method handles hyperparameters.
        Parameters
        ----------
        k: positive int
            Number of arms of the Bandit problem.
        kwargs: dictionary
            Additional parameters, ignored.
        """
        self.k = k

    def reset(self):
        """
        Reinitializes the agent to 0 knowledge, good as new.

        No inputs or outputs.
        The reset() method handles variables.
        """
        pass

    def learn(self, a:int, r:float):
        """
        Learning method. The agent learns that action a yielded reward r.
        Parameters
        ----------
        a: positive int < k
            Action that yielded the received reward r.
        r: float
            Reward for having performed action a.
        """
        pass

    def act(self) -> int:
        """
        Agent's method to select a lever (or Bandit) to pull.
        Returns
        -------
        a : positive int < k
            The action the agent chose to perform.
        """
        raise NotImplementedError("Calling method act() in Abstract class Bandit_Agent")

class Random_Agent(Bandit_Agent):
    """
    This agent doesn't learn, just acts purely randomly.
    Good baseline to compare to other agents.
    """
    def act(self):
        """
        Random action selection.
        Returns
        -------
        a : positive int < k
            A randomly selected action.
        """
        return np.random.randint(self.k)


class EpsGreedy_SampleAverage(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # This class uses Sample Averages to estimate q; others are non stationary.
    def __init__(self, k:int, lr:float, alpha:float, eps:float, c:float, q0:float, **kwargs):
        self.k = k
        keys = [i for i in range(self.k)]
        self.q = { key: 0 for key in keys}
        self.n = {key: 1 for key in keys}
        self.eps = eps

    def reset(self):
        keys = [i for i in range(self.k)]
        self.q = {key: 0 for key in keys}
        self.n = {key: 1 for key in keys}

    def learn(self, a:int, r:float):
        self.q[a] = self.q[a] + (1.0/self.n[a])*(r-self.q[a])
        self.n[a] += 1

    def act(self) -> int:
        # random = np.random.randint(low=0, high=100)
        random = np.random.random_sample()
        if self.eps > random:
            return np.random.randint(low=0, high=7)
        else:
            return max(self.q, key=self.q.get)

class EpsGreedy(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # Non stationary agent with q estimating and eps-greedy action selection.
    def __init__(self, k: int, lr: float, alpha: float, eps: float, c: float, q0: float, **kwargs):
        self.k = k
        keys = [i for i in range(self.k)]
        self.q = {key: 0 for key in keys}
        self.eps = eps
        self.lr = lr

    def reset(self):
        keys = [i for i in range(self.k)]
        self.q = {key: 0 for key in keys}

    def learn(self, a: int, r: float):
        self.q[a] = self.q[a] + self.lr * (r - self.q[a])

    def act(self) -> int:
        # random = np.random.randint(low=0, high=100)
        random = np.random.random_sample()
        if self.eps > random:
            return np.random.randint(low=0, high=7)
        else:
            return max(self.q, key=self.q.get)


class OptimisticGreedy(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # Same as above but with optimistic starting values.
    def __init__(self, k: int, lr: float, alpha: float, eps: float, c: float, q0: float, **kwargs):
        self.k = k
        keys = [i for i in range(self.k)]
        self.q = {key: q0 for key in keys}
        self.eps = 0
        self.lr = lr

    def reset(self):
        keys = [i for i in range(self.k)]
        self.q = {key: 0 for key in keys}

    def learn(self, a: int, r: float):
        self.q[a] = self.q[a] + self.lr * (r - self.q[a])

    def act(self) -> int:
        # random = np.random.randint(low=0, high=100)
        random = np.random.random_sample()
        if self.eps > random:
            return np.random.randint(low=0, high=7)
        else:
            return max(self.q, key=self.q.get)


class UCB(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    def __init__(self, k:int, lr:float, alpha:float, eps:float, c:float, q0:float, **kwargs):
        self.k = k
        self.c = c
        keys = [i for i in range(self.k)]
        self.q = {key: 0 for key in keys}
        self.n = {key: 1 for key in keys}
        self.a = {key: 0 for key in keys}
        self.t = 1
        self.eps = eps

    def reset(self):
        keys = [i for i in range(self.k)]
        self.q = {key: 0 for key in keys}
        self.n = {key: 1 for key in keys}

    def learn(self, a:int, r:float):
        self.q[a] = self.q[a] + (1.0/self.n[a])*(r-self.q[a])
        self.t += 1
        self.n[a] += 1

    def act(self) -> int:
        for a in range(self.k):
            ln = np.log(self.t)
            self.a[a] = self.q[a] + (self.c * np.sqrt(ln / self.n[a]))
        return max(self.a, key=self.a.get)


class Gradient_Bandit(Bandit_Agent):
    # TODO: implement this class following the formalism above.
    # If you want this to run fast, use the my_random_choice function from
    # utils instead of np.random.choice to sample from the softmax
    # You can also find the softmax function in utils.
    def __init__(self, k:int, lr:float, alpha:float, eps:float, c:float, q0:float, **kwargs):
        self.k = k
        self.c = c
        self.alpha = alpha
        keys = [i for i in range(self.k)]
        self.q = {key: 0 for key in keys}
        self.n = {key: 1 for key in keys}
        self.a = {key: 0 for key in keys}
        self.h = {key: 0 for key in keys}
        self.t = 1
        self.eps = eps
        self.sum = 0

    def reset(self):
        keys = [i for i in range(self.k)]
        self.q = {key: 0 for key in keys}
        self.n = {key: 1 for key in keys}

    def learn(self, a:int, r:float):
        self.q[a] = self.q[a] + (1.0/self.n[a])*(r-self.q[a])
        ln = np.log(self.t)
        self.sum += r
        AVG = self.sum/self.t
        self.a[a] = self.q[a] + (self.c * np.sqrt(ln / self.n[a]))
        self.h[a] = self.h[a] - self.alpha*(r-AVG)*softmax(a)
        self.t += 1
        self.n[a] += 1

    def act(self) -> int:
        for a in range(self.k):
            ln = np.log(self.t)
            self.a[a] = self.q[a] + (self.c * np.sqrt(ln / self.n[a]))
        return max(self.a, key=self.a.get)

