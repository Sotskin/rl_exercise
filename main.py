import gym
import enum
import time
import argparse
import numpy as np

from agent import Agent

class Colors(enum.Enum):
    green = '\033[32m'
    red = '\033[91m'
    end = '\033[0m'

    @staticmethod
    def encode(color, message):
        return color.value + message + Colors.end.value

def evaluate(agent, env, min_reward, n_samples, viz=False):
    rewards = np.zeros(n_samples)

    for i in range(n_samples):
        total_reward = 0
        done = False
        observation = env.reset()

        while not done:
            if viz:
                env.render()

            action = agent.query(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward

        rewards[i] = total_reward

    print('Agent summary:')
    print('\tRewards > {}:\t{}'.format(
        min_reward, np.sum(rewards >= min_reward) / n_samples * 100.))
    print('\tLowest reward:\t{}'.format(np.min(rewards)))
    print('\tHighest reward:\t{}'.format(np.max(rewards)))
    print('\tMedian reward:\t{}'.format(np.median(rewards)))
    print('\tMean reward:\t{}'.format(np.mean(rewards)))
    print('\tStatus:\t\t{}'.format(Colors.encode(Colors.green, 'PASSED')
                                   if np.mean(rewards) > min_reward
                                   else Colors.encode(Colors.red, 'FAILED')))

def format_time(seconds):
    return '{}:{}.{} '.format(int(seconds // 60),
                              int(seconds % 60), int((seconds * 100) % 100))

def main(args):
    env = gym.make('CartPole-v0')
    agent = Agent(env)

    pre_learn = time.time()
    agent.learn(max_episodes=1000)
    post_learn = time.time()

    evaluate(agent, env, args.reward_for_solved, args.solved_samples)
    post_eval = time.time()

    print('Agent training time on {} episodes: {}'.format(
        1000, format_time(post_learn - pre_learn)))
    print('Agent evaluation time on {} episodes: {}'.format(
        args.solved_samples, format_time(post_eval - post_learn)))

    if args.visualize:
        evaluate(agent, env, np.inf, 10, viz=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward-for-solved', type=int, default=195,
                        help="how much reward to consider episode solved")
    parser.add_argument('--solved-samples', type=int, default=100,
                        help='how many samples to take to determine if solved')
    parser.add_argument('--visualize', action='store_true',
                        help='whether to visualize the agent graphically')
    main(parser.parse_args())
