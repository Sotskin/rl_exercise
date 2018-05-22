import gym
import argparse
import numpy as np

from agent import Agent

def evaluate(agent, env, min_reward, n_samples):
    rewards = np.zeros(n_samples)

    for i in range(n_samples):
        total_reward = 0
        done = False
        observation = env.reset()

        while (not done) and total_reward < min_reward:
            action = agent.query(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward

        rewards[i] = total_reward

    print('Agent passed {} percent. Low score {}, median score {}'.format(
        np.sum(rewards >= min_reward) / n_samples, np.min(rewards), 
        np.median(rewards)))


def main(args):
    env = gym.make('CartPole-v0')
    agent = Agent(env)

    agent.learn(max_episodes=3000)
    evaluate(agent, env, args.reward_for_solved, args.solved_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward-for-solved', type=int, default=195,
                        help="how much reward to consider episode solved")
    parser.add_argument('--solved-samples', type=int, default=100,
                        help='how many samples to take to determine if solved')
    main(parser.parse_args())
