import math
import numpy as np

# from evn_pso import evn_pso
# from RL_brain import PolicyGradient

from pso_class import pso
from spso2011_class import spso2011
from m_spso2011_class import m_spso2011
from m2_spso2011_class import m2_spso2011

from wrong_pso import wrong_pso


def Func1(particle):
    val = 0.0
    for i in range(len(particle)):
        val = val + particle[i] * particle[i]
    return val


def Rosenbrock_Fun(particle):
    val = 0.0
    for i in range(len(particle)-1):
        val = val + math.pow((1 - particle[i]), 2) + 100 * math.pow(
            (particle[i+1] - math.pow(particle[i], 2)), 2)
    return val


if __name__ == "__main__":

    roundtime = 1
    iteration = 2500
    n_point = 40
    min_bound = -100
    max_bound = 100
    dimension = 10
    c1 = 0.5 + np.log(2.0)
    c2 = 0.5 + np.log(2.0)
    w = 1.0 / (2.0 * np.log(2.0))

    # pso1 = pso(dimension, min_bound, max_bound, w, c1,
    #            c2, n_point, roundtime, iteration, Func1)
    # pso1.move_swarm()

    spso2011 = spso2011(dimension, min_bound, max_bound, w, c1, n_point, roundtime, iteration, Func1)
    spso2011.swarm_move()

    m_spso2011 = m_spso2011(dimension, min_bound, max_bound,
                            w, c1, n_point, roundtime, iteration, Func1)
    m_spso2011.swarm_move()

    m2_spso2011 = m2_spso2011(
        dimension, min_bound, max_bound, w, c1, n_point, roundtime, iteration, Func1)
    m2_spso2011.swarm_move()

    # pso2 = wrong_pso(dimension, min_bound, max_bound, w, c1, c2, n_point, iteration, Func1)
    # pso2.move_swarm()

    # env = evn_pso(w=0.9, c1=2, c2=2, lower=-10, upper=10,
    #               p_dim=20, p_count=400, function=Rosenbrock_Fun)

    # RL = PolicyGradient(
    #     n_actions=env.action_space.n,
    #     n_features=2,
    #     learning_rate=0.02,
    #     reward_decay=0.995,
    #     # output_graph=True,
    # )

    # for i_episode in range(2):
    #     observation = env.reset()
    #     while True:
    #         action = RL.choose_action(observation)

    #         observation_, reward, done, info = env.step(action)

    #         RL.store_transition(observation, action, reward)

    #         if done:
    #             ep_rs_sum = sum(RL.ep_rs)
    #             print(ep_rs_sum)
    #             if 'running_reward' not in globals():
    #                 running_reward = ep_rs_sum
    #             else:
    #                 running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

    #             print("episode:", i_episode, "  reward:", int(running_reward))

    #             vt = RL.learn()  # train
    #             break

    #         observation = observation_
