from __future__ import division
import random
import math
import numpy as np


class Particle:
    def __init__(self, position, velocity):
        self.position = []          # particle position
        self.velocity = []          # particle velocity
        self.best_position = []          # best position individual
        self.best_fitness = -1          # best error individual
        self.fitnesses = -1               # error individual

        for i in range(len(position)):
            self.velocity.append(velocity[i])
            self.position.append(position[i])


class pso():
    def __init__(self, dimension, min_bound, max_bound, w, c1, c2, n_points, roundtime, iterations, func):
        self.dimension = dimension

        self.iteration = iterations
        self.roundtime = roundtime

        self.min_bound = min_bound
        self.max_bound = max_bound

        self.w = 1.0 / (2.0 * np.log(2.0))
        self.c1 = 0.5 + np.log(2.0)
        self.c2 = 0.5 + np.log(2.0)

        self.n_point = n_points

        self.func = func

        self.gloab_fitness = -1                   # best error for group
        self.gloab_position = []                   # best position for group

        self.swarm = self.init_swarm()

    def init_swarm(self):
        swarm = []
        for i in range(0, self.n_point):
            position = np.random.randint(
                self.min_bound, self.max_bound, self.dimension)
            velocity = np.random.uniform(
                self.min_bound, self.max_bound, self.dimension)
            swarm.append(Particle(position, velocity))
        return swarm

    def move_swarm(self):

        for _ in range(self.iteration):
            
            for i in range(self.n_point):
                # self.swarm[j].evaluate(self.func)
                p = self.swarm[i]

                p.fitnesses = self.func(p.position)

                if p.fitnesses < p.best_fitness or p.best_fitness == -1:
                    p.best_position = p.position
                    p.best_fitness = p.fitnesses

                if p.fitnesses < self.gloab_fitness or self.gloab_fitness == -1:
                    self.gloab_position = list(p.position)
                    self.gloab_fitness = float(p.fitnesses)

            for i in range(self.n_point):
                p = self.swarm[i]

                # self.swarm[j].update_velocity(self.gloab_position)
                for j in range(self.dimension):
                    r1 = random.random()
                    r2 = random.random()

                    vel_cognitive = self.c1*r1 * \
                        (p.best_position[j] -
                         p.position[j])
                    vel_social = self.c2*r2 * \
                        (self.gloab_position[j]-p.position[j])
                    p.velocity[j] = self.w * \
                        p.velocity[j]+vel_cognitive+vel_social

                # p.update_position(self.min_bound, self.max_bound)
                for j in range(self.dimension):
                    p.position[j] = p.position[j] + \
                        p.velocity[j]
                    # adjust maximum position if necessary
                    if p.position[j] > self.max_bound:
                        p.position[j] = self.max_bound
                    # adjust minimum position if neseccary
                    if p.position[j] < self.min_bound:
                        p.position[j] = self.min_bound

            print self.gloab_fitness
        # print self.gloab_position
