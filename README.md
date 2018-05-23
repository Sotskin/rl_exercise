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
- Test your implementation using `main.py` and you should, at least, reach 100% accuracy within 100 episodes
