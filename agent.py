import numpy as np

class Agent(object):
    def __init__(self, env):
        self._env = env

    def learn(self, max_episodes=1000):
        pass

    def query(self, observation):
        return self._env.action_space.sample()
