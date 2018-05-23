# Exercise on Reinforcement Learning
In this exercise you will be implementing a [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
to solve the [CartPole](https://gym.openai.com/envs/CartPole-v0/) problem.
This exercise will test you on how quickly you can pick up new concepts through self-study.

For an thorough introduction to reinforcement learning you can see the 
[UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html),
taught by David Silver (a name you will likely see in other RL materials). Accompanying this
course is the book [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html).
Additionally, as the exercise here will deal deep learning, another resource that may be useful
is the [Deep RL course](http://rll.berkeley.edu/deeprlcourse/) at Berkely.

# Description
The problem you will be solving is to get a cart to balance a pole, without it
tipping over. See [this link](https://gym.openai.com/envs/CartPole-v0/) for
a depiction of the problem, and information on the environment you will be using.

The algorithm we wish you to use to solve this problem is [Deep Q-Learning](http://www.davidqiu.com:8888/research/nature14236.pdf) (note,
you will find additional resources below). You will be solving this problem using an [OpenAI Gym](https://gym.openai.com/) environment,
which is already setup.

The machine learning framework you will be working with to implement the neural networks that you will likely
need is [MXNet](https://mxnet.incubator.apache.org/). This framework makes writing neural networks a lot
more simple, especially using the new high-level subpackage called [Gluon](https://mxnet.incubator.apache.org/gluon/index.html).

You can find a list of the only dependencies you will need to install for Python in the file named "dependencies" above.

### In summary
- Implement an RL agent in the file `agent.py`, which learns via Deep Q-Learning
- This agent should implement the methods already defined in it
- Be sure to have sufficient print statements to show learning progress
- Test your implementation using `main.py` and you should, at least, reach 100% accuracy within 1000 episodes
- (Optional) If you finish this, see additional resources below to try to improve your agent and/or use it in harder environments


# Resources
Here is an accumulation of the resources listed above, as well as some additional.

### Main resources
- [Introduction to RL course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)
- [Introduction to RL book](http://incompleteideas.net/book/the-book-2nd.html)
- [Deep RL course](http://rll.berkeley.edu/deeprlcourse/)
- [OpenAI Gym](https://gym.openai.com/)
- [CartPole Gym Environment](https://gym.openai.com/envs/CartPole-v0/)
- [Deep Q-Learning Nature](http://www.davidqiu.com:8888/research/nature14236.pdf)
- [Deep Q-Learning for Atari](https://arxiv.org/abs/1312.5602) (trying an Atari game if you solve CartPole would be an idea)

### Optional
If you successfully implemented the Deep Q-Learning agent, and solved the CartPole environment, consider some
of the following resources. If you have time you may wish to extend on this exercise by implementing one of the methods below, and
trying it on another environment as well (feel free to turn that code in too!). Looking at the work cited in the papers below
is a good way to get additional resources on topics you are not familiar with.

### Additional Resources (extensions to Deep Q-Learning and other algorithms)
- [Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [Dueling Networks](https://arxiv.org/abs/1511.06581)
- [Prioritized Replay](https://arxiv.org/abs/1511.05952)
- [Asynchronous Advantage Actor-Critic (A3C)][https://arxiv.org/abs/1602.01783] (this is a policy gradient method)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) (also a policy gradient method)
