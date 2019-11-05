import random

import numpy as np
from gym import spaces
from gym.utils import seeding


class evn_pso():
    def __init__(self, w, c1, c2, lower, upper, p_dim, p_count, function):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.lower = lower
        self.upper = upper
        self.p_dim = p_dim
        self.p_count = p_count
        self.function = function

        self.init_swarms()

        """format [ d1 , d2 ] """
        # self.low = np.array([self.lower, self.speed.flatten().min()])
        # self.high = np.array([self.upper,self.speed.flatten().max()])
        self.action_space = spaces.Discrete(6)
        # self.observation_space = spaces.Box(self.low, self.high)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        i = self.np_random.randint(low=0, high=self.p_count)

        self.state = np.array(
            [
                np.sum(self.best_population[i] - self.population[i]),
                np.sum(self.gloab_population - self.population[i])
            ]
        )
        return np.array(self.state)

    def init_swarms(self):

        self.population = np.array([np.random.randint(
            self.lower, self.upper, self.p_dim) for _ in range(self.p_count)], dtype=float)
        self.fitnesses = np.zeros(self.p_count, dtype=float)
        self.best_population = np.zeros(self.population.shape, dtype=float)
        self.best_fitnesses = np.full(
            self.p_count, np.finfo(np.float32).max, dtype=float)
        # use spso 2011 suggest
        self.speed = np.array([np.random.randint(self.lower, self.upper, self.p_dim) -
                               self.population[_] for _ in range(self.p_count)], dtype=float)
        # self.speed = np.zeros(self.population.shape)

        """ record gloab value"""
        self.gloab_population = np.zeros(self.p_dim)
        self.gloab_fitness = np.finfo(np.float32).max

    def step(self, action):
        # d1, d2 = self.state

        if action == 0:
            self.c1 += 0.01
        elif action == 1:
            self.c1 -= 0.01
        elif action == 2:
            self.c2 += 0.01
        elif action == 3:
            self.c2 -= 0.01
        elif action == 4:
            self.w += 0.01
        elif action == 5:
            self.w -= 0.01

        for i in range(self.p_count):
            """ self.function() according to function """
            self.fitnesses[i] = self.function(
                self.p_dim, self.population[i])
            """compare with self best """
            if(self.fitnesses[i] < self.best_fitnesses[i]):
                self.best_fitnesses[i] = self.fitnesses[i]
                self.best_population[i] = self.population[i]

        """ updata gloab best value"""
        if(np.min(self.best_fitnesses) < self.gloab_fitness):
            self.gloab_fitness = np.min(self.best_fitnesses)
            self.gloab_population = self.best_population[np.argmin(
                self.best_fitnesses)]
            """set done"""
            done = True
            reward = 1
        else:
            done = False
            reward = 0

        """update each searm speed"""

        for i in range(self.p_count):
            self.speed[i] = self.w * self.speed[i] + self.c1 * random.uniform(0, 1) * (
                self.best_population[i] - self.population[i]) + self.c2 * random.uniform(0, 1) * (self.gloab_population - self.population[i])

        """update populations"""
        # i think this can improve by anther way to imply
        for i in range(self.p_count):
            self.population[i] = self.population[i] + self.speed[i]

        """check lower and upper bound,if over set the upper num,same as lower"""

        self.population = np.clip(self.population, self.lower,
                                  self.upper)
        print "{}".format(np.min(self.gloab_fitness))
        
        """ for return """
        reward = -1
        i = np.argmin(self.best_fitnesses)
        d1 = np.sum(self.best_population[i] - self.population[i])
        d2 = np.sum(self.gloab_population - self.population[i])
        self.state = (d1, d2)

        return np.array(self.state), reward, done, {}
