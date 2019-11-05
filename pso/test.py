# import numpy as np
# from gym import spaces
# from gym.utils import seeding

# a = np.array([np.random.randint(0, 9, 5) for _ in range(5)])
# print a


# low = np.array([0, a.flatten().min()])
# high = np.array([2, a.flatten().max()])
# print low
# print high

# # action_space = spaces.Discrete(6)
# # print action_space
# observation_space = spaces.Box(low, high)
# print observation_space.low

import time
import random
import os
from multiprocessing import Pool


class MyThread(object):
    def __init__(self, func):

        self.func = func

    def long_time_task(self, i):
        print 'Run task %s (%s)...' % (i, os.getpid())
        time.sleep(random.random() * 3)
        print i
        return (i, os.getpid())

    def parse_thread(self):
        print 'Parent process %s.' % os.getpid()
        p = Pool()
        results = []
        for i in range(10):
            results.append(p.apply_async(
                long_time_task_wrapper, args=(self, i,)))
        for res in results:
            print res.get()
            print 'Waiting for all subprocesses done...'
            p.close()
            p.join()
            print 'All subprocesses done.'


def long_time_task_wrapper(cls_instance, i):
    return cls_instance.long_time_task(i)


def main():
    print "start"
    tt = MyThread(long_time_task_wrapper)
    tt.parse_thread()

if __name__=="__main__": 
    # main()
    a = -84
    b = 31.198128862519297
    c = a + b
    print c