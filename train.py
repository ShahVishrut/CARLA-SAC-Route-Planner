
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

def run():
    try:
        buffer_size = 1e4
        batch_size = 32
        state_dim = (128, 128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        episodes = 10000

        ep = 0

        action_space = gym.spaces.Box(low=-0.75, high=0.75, shape=(1,), dtype=np.float32)

        replay_buffer = ReplayBuffer(state_dim, batch_size, buffer_size, device)
        model = SAC(8196, state_dim, action_space)


        env = SimEnv(visuals=True, **env_params)

        while ep < episodes:
            env.create_actors()
            rw, ct = env.generate_episode(model, replay_buffer, ep, evaluate=False)
            env.reset()
            with open("training_log.csv", "a") as f:
                f.write(f"{ep},{rw},{ct}\n")
            ep += 1
    finally:
        env.quit()

if __name__ == "__main__":
    run()
