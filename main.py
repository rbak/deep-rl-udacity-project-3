from comet_ml import Experiment

from agent import Agent
import argparse
from collections import deque
from env import Environment
import numpy as np
import torch

hyper_params = {
    'memory_size': 20000,      # replay buffer size
    'batch_size': 64,          # sample batch size
    't_random': 3,             # random steps at start of trajectory
    't_max': 1000,             # trajectory length
    'num_epochs': 10,          # number of updates
    'c_vf': 0.5,               # coefficent for vf loss (c1)
    'c_entropy': 0.001,        # starting value for coefficent for entropy (c2)
    'epsilon': 0.2,            # starting value for clipping parameter
    'gae_param': 0.95,         # gae param (λ)
    'discount': .99,           # discount (γ)
    'curation_percentile': 0,  # percent of trajectory data to drop
    'gradient_clip': 5,        # gradient clip
}


def main(args):
    if args.examine:
        with Environment(no_graphics=True) as env:
            examine(env)
    if args.random:
        with Environment(no_graphics=False) as env:
            for i in range(5):
                random(env)
    if args.train:
        with Environment(no_graphics=True) as env:
            experiment = _setup_experiment(disabled=(not args.log))
            if experiment:
                with experiment.train():
                    train(env, experiment)
            else:
                train(env, experiment)
    if args.test:
        with Environment(no_graphics=False) as env:
            test(env)


def _setup_experiment(disabled=False):
    try:
        experiment = Experiment(project_name="udacity-deeprl-project-3", log_code=False,
                                log_env_details=False, disabled=disabled)
    except Exception as e:  # Even when disabled, the comet library will throw an exception if the key is not set.
        print(e)
        experiment = None
    return experiment


def examine(env):
    env_info = env.reset()
    print('Number of agents:', len(env_info.agents))
    action_size = env.action_space_size
    print('Number of actions:', action_size)
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


def random(env):
    env_info = env.reset(train_mode=False)
    num_agents = len(env_info.agents)
    action_size = env.action_space_size
    rewards = np.zeros(num_agents)
    while True:
        actions = np.random.randn(num_agents, action_size)
        actions = np.clip(actions, -1, 1)
        env_info = env.step(actions)
        dones = env_info.local_done
        rewards += env_info.rewards
        if np.any(dones):
            break
        print('\rReward averaged across agents: {:.2f}'.format(np.mean(rewards)), end="")
    print('\rReward averaged across agents: {:.2f}'.format(np.mean(rewards)))


def test(env):
    env.reset(train_mode=False)
    states = env.info.vector_observations
    agent = Agent(env, hyper_params)
    agent.actor_local.load_state_dict(torch.load('results/actor_checkpoint.pth'))
    agent.critic_local.load_state_dict(torch.load('results/critic_checkpoint.pth'))
    rewards_total = 0
    while True:
        actions = [agent.act(state) for state in states]
        env.step(actions)
        rewards = env.info.rewards
        rewards_total += np.array(rewards)
        dones = env.info.local_done
        if np.any(dones):
            break
    print('\rRewards: ', np.mean(rewards_total))


def train(env, experiment=None, max_episodes=1000, max_t=300):
    agent = Agent(env, hyper_params)
    rewards_window = [deque(maxlen=100) for n in range(env.num_agents)]
    for i_episode in range(1, max_episodes + 1):
        env.reset(train_mode=True)
        states = env.info.vector_observations
        agent.reset()
        rewards_total = 0
        for t in range(max_t):
            actions = agent.act(states)
            env.step(actions)
            next_states = env.info.vector_observations
            rewards = env.info.rewards
            dones = env.info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            rewards_total += np.array(rewards)
            if np.any(dones):
                break
        # Track rewards
        for i, r in enumerate(rewards_total):
            rewards_window[i].append(r)
        if experiment:
            experiment.log_metric('reward', np.mean(rewards_window), step=i_episode)
        print('\rEpisode {}\tAverage Reward: {:.2f}'.format(i_episode, np.mean(rewards_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Reward: {:.2f}'.format(i_episode, np.mean(rewards_window)))
            torch.save(agent.actor_local.state_dict(), 'results/actor_checkpoint.pth')
            torch.save(agent.critic_local.state_dict(), 'results/critic_checkpoint.pth')
        mean_window_rewards = np.mean(rewards_window, axis=1)
        if (mean_window_rewards >= 30.0).all():
            print('\nEnvironment solved in {:d} episodes.  Average agent rewards: '
                  .format(i_episode), mean_window_rewards)
            torch.save(agent.actor_local.state_dict(), 'results/actor_checkpoint.pth')
            torch.save(agent.critic_local.state_dict(), 'results/critic_checkpoint.pth')
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepRL - Continuous control project')
    parser.add_argument('--examine',
                        action="store_true",
                        dest="examine",
                        help='Print environment information')
    parser.add_argument('--random',
                        action="store_true",
                        dest="random",
                        help='Start a random agent')
    parser.add_argument('--train',
                        action="store_true",
                        dest="train",
                        help='Train a new network')
    parser.add_argument('--test',
                        action="store_true",
                        dest="test",
                        help='Load an existing network and test it')
    parser.add_argument('--log',
                        action="store_true",
                        dest="log",
                        help='Log results to comet.ml')
    parser.add_argument('--single',
                        action="store_true",
                        dest="single",
                        help='Run the single agent version of the environment')
    parser.add_argument('--crawler',
                        action="store_true",
                        dest="crawler",
                        help='Run the crawler environment')
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.error('No arguments provided.')
    main(args)
