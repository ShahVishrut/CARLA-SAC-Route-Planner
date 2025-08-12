
from replay_buffer import ReplayBuffer
from model import SAC

from config import action_map, env_params
from utils import *
from environment import SimEnv
import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter

def run():
    try:
        parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
        parser.add_argument('--env-name', default="HalfCheetah-v2",
                            help='Mujoco Gym environment (default: HalfCheetah-v2)')
        parser.add_argument('--policy', default="Gaussian",
                            help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
        parser.add_argument('--eval', type=bool, default=True,
                            help='Evaluates a policy a policy every 10 episode (default: True)')
        parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                            help='discount factor for reward (default: 0.99)')
        parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                            help='target smoothing coefficient(τ) (default: 0.005)')
        parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                            help='learning rate (default: 0.0003)')
        parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                            help='Temperature parameter α determines the relative importance of the entropy\
                                    term against the reward (default: 0.2)')
        parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                            help='Automaically adjust α (default: False)')
        parser.add_argument('--seed', type=int, default=123456, metavar='N',
                            help='random seed (default: 123456)')
        parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                            help='batch size (default: 256)')
        parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                            help='maximum number of steps (default: 1000000)')
        parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                            help='hidden size (default: 256)')
        parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                            help='model updates per simulator step (default: 1)')
        parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                            help='Steps sampling random actions (default: 10000)')
        parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                            help='Value target update per no. of updates per step (default: 1)')
        parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                            help='size of replay buffer (default: 10000000)')
        parser.add_argument('--cuda', action="store_true",
                            help='run on CUDA (default: False)')
        args = parser.parse_args()

        buffer_size = 1e4
        batch_size = 32
        state_dim = (128, 128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        episodes = 10000

        ep = 0

        action_space = gym.spaces.Box(low=-0.75, high=0.75, shape=(1,), dtype=np.float32)

        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = SAC(8196, state_dim, action_space, args)

        #model.load('weights/model_ep_1400')
        #ep = 1400

        env = SimEnv(visuals=True, **env_params)
        env.load(model, 5, replay_buffer)
        ep = 6

        while ep < episodes:
            env.create_actors()
            env.generate_episode(model, replay_buffer, ep, evaluate=False)
            env.reset()
            ep += 1
    finally:
        env.quit()

if __name__ == "__main__":
    run()
