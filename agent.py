import numpy as np
import random
from collections import deque
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn

class Agent(object): # DQN, Solved with around 150 episodes
    def __init__(self, env):
        self._env = env
        self.q = {} # {s:{a:v}}
        self.gamma = 0.9 # discount factor
        self.epsilon = 0.9
        self.epsilon_decay = 0.002
        self.epsilon_min = 0.1
        self.batch_size = 32
        self.max_steps = 1000
        
        self.memory = deque() # s, a, r, s', done
        self.memory_limit = 15000
        self.model = self.create_model()
        self.trainer = gluon.Trainer(self.model.collect_params(),
                'sgd', {'learning_rate':0.01})
        self.loss = gluon.loss.L2Loss()

        self.train_loss = 0.

    def create_model(self):
        net = nn.Sequential()
        with net.name_scope():
            net.add(
                    nn.Dense(32, activation="relu"),
                    nn.Dense(16, activation="relu"),
                    nn.Dense(self._env.action_space.n)
                    )
        net.initialize(init=init.Xavier())
        return net
            
    def obs_to_state(self, observation):
        #s = tuple((observation*10).astype(int))
        state = nd.array(observation)
        state = nd.reshape(state, [1,self._env.observation_space.shape[0]])
        return state

    def greedy(self, observation):
        if np.random.random_sample(None) < self.epsilon:
            return self._env.action_space.sample()
        state = self.obs_to_state(observation)
        return int(nd.argmax(self.model(state), 1).asnumpy().item(0)) # orz
    
    def replay(self):
        # experience replay
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state_batch = nd.array([b[0] for b in batch]) 
        action_batch = nd.array([b[1] for b in batch])
        target_batch = nd.array([b[2] for b in batch])
        next_state_batch = nd.array([b[3] for b in batch])
        for i in range(self.batch_size): # s, a, r, _s, d
            if not batch[i][4]:
                target_batch[i] = target_batch[i] + self.gamma * \
                        np.max(self.model(nd.reshape(next_state_batch[i],[1,4])),1)
        with autograd.record():
            q_target_batch = self.model(state_batch)
            #print(q_target_batch.shape,"\n", target_batch.shape)
            output_batch = nd.pick(q_target_batch, action_batch, 1)
            loss = self.loss(output_batch,target_batch)
        loss.backward()
        self.train_loss += loss.mean().asscalar()
        self.trainer.step(self.batch_size)
        return

    def learn(self, max_episodes=1000):
        for i in range(max_episodes):
            self.train_loss = 0.
            obser = self._env.reset()
            for t in range(self.max_steps):
                action = self.greedy(obser)
                next_obser, reward, done, _ =  self._env.step(action)
                #next_state = self.obs_to_state(next_state)
                self.memory.append((obser, action, reward, next_obser, done))
                if len(self.memory) > self.memory_limit:
                    self.memory.popleft()
                obser = next_obser
                self.replay()
                if done:
                    self.epsilon -= self.epsilon_decay
                    if self.epsilon < self.epsilon_min:
                        self.epsilon = self.epsilon_min
                    print("Episode {} over, loss = {:.3f}, epi = {:.3f}, steps = {}"
                            .format(i, self.train_loss, self.epsilon, t))
                    break;

    def query(self, observation):
        return int(nd.argmax(self.model(self.obs_to_state(observation)), 1).asnumpy().item(0))
