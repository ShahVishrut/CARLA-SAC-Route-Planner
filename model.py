import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from model_utils import soft_update, hard_update

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ConvNet(nn.Module):
    def __init__(self, in_channels) -> None:
        super(ConvNet, self).__init__()
        # conv2d(in_channels, out_channels, kernel_size, stride)
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
    
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = x.reshape(x.size(0), -1)
        # x = torch.cat((x,y), dim=-1)
        # x = F.relu(self.fc1_bn(self.fc1(x)))
        # x = F.relu(self.fc2_bn(self.fc2(x)))
        # x = self.fc3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class SAC(object):
    def __init__(self, num_inputs, state_dim, action_space):

        self.iterations = 0
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2

        self.state_shape = (-1,) + state_dim

        self.policy_type = "Gaussian"
        self.target_update_interval = 1
        self.automatic_entropy_tuning = False

        self.hidden_size = 256
        self.lr = 0.0003

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.convnet = ConvNet(in_channels=1).to(self.device)
        self.convnet_target = ConvNet(in_channels=1).to(self.device)
        hard_update(self.convnet_target, self.convnet)


        self.feature_dim = 8192 + 4

        self.critic = QNetwork(self.feature_dim, action_space.shape[0], self.hidden_size).to(device=self.device)
        self.critic_optim = Adam(list(self.critic.parameters()) + list(self.convnet.parameters()), lr=self.lr)

        self.critic_target = QNetwork(self.feature_dim, action_space.shape[0], self.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

        self.policy = GaussianPolicy(self.feature_dim, action_space.shape[0], self.hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)


    def select_action(self, state, evaluate=False):
        image, vector = state
        self.convnet.eval()
        self.policy.eval()
        image = torch.FloatTensor(image).reshape(self.state_shape).unsqueeze(0).to(self.device)
        vector = torch.FloatTensor(vector).unsqueeze(0).to(self.device)

        conv_features = self.convnet(image)
        combined_state = torch.cat([conv_features, vector], dim=1)

        if evaluate is False:
            action, _, _ = self.policy.sample(combined_state)
        else:
            _, _, action = self.policy.sample(combined_state)
        return action.detach().cpu().numpy()[0]

    def train(self, memory):
        # Sample a batch from memory
        self.iterations += 1
        self.convnet.train()
        self.policy.train()
        self.critic.train()

        image_batch, vector_batch, action_batch, next_image_batch, next_vector_batch, reward_batch, done_batch = memory.sample()

        conv_features = self.convnet(image_batch)
        combined_state = torch.cat([conv_features, vector_batch], dim=1)

        with torch.no_grad():
            next_conv_features = self.convnet_target(next_image_batch)
            combined_next_state = torch.cat([next_conv_features, next_vector_batch], dim=1)


        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(combined_next_state)
            qf1_next_target, qf2_next_target = self.critic_target(combined_next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1-done_batch) * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(combined_state, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        with torch.no_grad():
            conv_features_detached = self.convnet(image_batch)
            state_batch = torch.cat([conv_features_detached.detach(), vector_batch], dim=1)

        pi, log_pi, _ = self.policy.sample(state_batch)


        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.iterations % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.convnet_target, self.convnet, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def save(self, filename):
        print('Saving models to {}'.format(filename))
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'convnet_state_dict': self.convnet.state_dict(),
            'convnet_target_state_dict': self.convnet_target.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),
            'iterations': self.iterations,
            'alpha': self.alpha,
            'log_alpha': self.log_alpha.detach().cpu() if self.automatic_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optim.state_dict() if self.automatic_entropy_tuning else None,
        }, filename)

    def load(self, filename):
        print('Loading models from {}'.format(filename))
        checkpoint = torch.load(filename, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.convnet.load_state_dict(checkpoint['convnet_state_dict'])
        self.convnet_target.load_state_dict(checkpoint['convnet_target_state_dict'])

        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

        self.iterations = checkpoint.get('iterations', 0)
        self.alpha = checkpoint.get('alpha', self.alpha)

        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha'].to(self.device).requires_grad_()

            if 'alpha_optimizer_state_dict' in checkpoint and checkpoint['alpha_optimizer_state_dict'] is not None:
                self.alpha_optim.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
