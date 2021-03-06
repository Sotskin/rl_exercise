import numpy as np
import random
from collections import deque
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn

class Agent(object): # DQN (experience replay and delayed update)
    def __init__(self, env):
        self._env = env
        self.gamma = 0.9 # discount factor
        self.epsilon = 1.
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.update_c = 1000 # steps to update target model
        self.max_steps = 1000
        
        self.memory = deque() # s, a, r, s', done
        self.memory_limit = 20000
        self.param_file_name = "net.params"
        self.train_model = self.create_model()
        self.target_model = self.create_model(True)
        self.trainer = gluon.Trainer(self.train_model.collect_params(),
                'sgd', {'learning_rate':0.01})
                #'nag', {'learning_rate':0.005})
                #'adam', {'learning_rate':0.001}) # all adaptive methods give bad result
        self.loss = gluon.loss.L2Loss()

        self.train_loss = 0.

    def create_model(self, target = False):
        net = nn.Sequential()
        with net.name_scope():
            net.add(
                    nn.Dense(32, activation="relu", in_units = 4),
                    nn.Dense(16, activation="relu", in_units = 32),
                    nn.Dense(self._env.action_space.n, in_units = 16)
                    )
        if target:
            self.train_model.save_params(self.param_file_name)
            net.load_params(self.param_file_name)
        else:
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
        return int(nd.argmax(self.train_model(state), 1).asscalar())
    
    def replay(self):
        # experience replay
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state_batch = nd.array([b[0] for b in batch]) 
        action_batch = nd.array([b[1] for b in batch])
        reward_batch = nd.array([b[2] for b in batch])
        next_state_batch = nd.array([b[3] for b in batch])
        target_batch = reward_batch + self.gamma * \
                np.max(self.target_model(next_state_batch),1)

        for i in range(self.batch_size): # s, a, r, _s, d
            if batch[i][4]:
                target_batch[i] = reward_batch[i]
                #target_batch[i] = target_batch[i] + self.gamma * \
                #        np.max(self.model(nd.reshape(next_state_batch[i],[1,4])),1)
        with autograd.record():
            q_target_batch = self.train_model(state_batch)
            #print(q_target_batch.shape,"\n", target_batch.shape)
            output_batch = nd.pick(q_target_batch, action_batch, 1)
            loss = self.loss(output_batch,target_batch)
        loss.backward()
        self.train_loss += loss.mean().asscalar()
        self.trainer.step(self.batch_size)
        return

    def learn(self, max_episodes=1000):
        verb_s = 1
        c = 0
        stop_count = 0
        stop_limit = 20
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
                if c % self.update_c == 0:
                    print("update target function at episode {}".format(i))
                    self.train_model.save_params(self.param_file_name)
                    self.target_model.load_params(self.param_file_name)
                c += 1

                if done:
                    stop_count = stop_count + 1 if t == 199 else 0
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay
                    if i % verb_s == 0:
                        print("Episode {} over, loss = {:.3f}, epi = {:.3f}, steps = {}"
                            .format(i+1, self.train_loss, self.epsilon, t))
                    break
            if stop_count == stop_limit:
                print("{} consecutive max reward, train over".format(stop_limit))
                break

        # update again at the end
        print("update target function at the end of last episode")
        self.train_model.save_params(self.param_file_name)
        self.target_model.load_params(self.param_file_name)

    def query(self, observation):
        return int(nd.argmax(self.target_model(self.obs_to_state(observation)), 1).asscalar())
