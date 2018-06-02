import numpy as np
import operator

class Agent(object): # Q-learning, not solved
    def __init__(self, env):
        self._env = env
        self.q = {} # {s:{a:v}}
        self.alpha = 0.6
        self.gamma = 0.9 # discount factor
        self.epsilon = 0.1
        self.max_steps = 1000

    def obs_to_state(self, observation):
        s = tuple((observation*10).astype(int))
        return s

    def greedy(self, observation, e = 0):
        state = self.obs_to_state(observation)
        if (not bool(self.q.get(state,{}))) or (e != 0 and np.random.random_sample(None) < e):
            return self._env.action_space.sample()
        _q = self.q[state]
        return max(_q.items(), key=operator.itemgetter(1))[0] 

    def get_q(self, observation, action):
        state = self.obs_to_state(observation)
        return self.q.get(state, {}).get(action, 0)
    
    def update_q(self, observation, action, value):
        state = self.obs_to_state(observation)
        if state not in self.q:
            self.q[state] = {}
        if action not in self.q[state]:
            self.q[state][action] = 0
        self.q[state][action] = self.q[state][action] + value
        
    def learn(self, max_episodes=1000):
        for i in range(max_episodes):
            obs = self._env.reset()
            for st in range(self.max_steps):
                a = self.greedy(obs, self.epsilon)
                _obs, r, done, _ = self._env.step(a)
                _q = self.get_q(obs, a)
                self.update_q(obs, a, self.alpha * (r + self.gamma * \
                        self.get_q(_obs, self.greedy(_obs))- _q))
                obs = _obs
                if done:
                    #print("Episode finished after {} timesteps".format(st+1))
                    break

    def query(self, observation):
        return self.greedy(observation)
