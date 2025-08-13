import cv2
import torch
import numpy as np
import pickle

class ReplayBuffer(object):
    def __init__(self, state_dim, batch_size, buffer_size, device) -> None:
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size,) + state_dim)
        self.state2 = np.zeros((self.max_size, 4))
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.next_state2 = np.array(self.state2)
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state[0]
        self.state2[self.ptr] = np.array(state[1])
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state[0]
        self.next_state2[self.ptr] = np.array(next_state[1])
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]).unsqueeze(1).to(self.device),
            torch.FloatTensor(self.state2[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).unsqueeze(1).to(self.device),
            torch.FloatTensor(self.next_state2[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                'state': self.state,
                'state2': self.state2,
                'action': self.action,
                'next_state': self.next_state,
                'next_state2': self.next_state2,
                'reward': self.reward,
                'done': self.done,
                'ptr': self.ptr,
                'crt_size': self.crt_size
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.state = data['state']
        self.state2 = data['state2']
        self.action = data['action']
        self.next_state = data['next_state']
        self.next_state2 = data['next_state2']
        self.reward = data['reward']
        self.done = data['done']
        self.ptr = data['ptr']
        self.crt_size = data['crt_size']
