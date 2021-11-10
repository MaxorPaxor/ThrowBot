import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy
import matplotlib.pyplot as plt


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2, bn=False, lr_critic=3e-04):
        super(Critic, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.linear_input = nn.Linear(input_size, hidden_size)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)])
        self.linear_output = nn.Linear(hidden_size, output_size)

        # Custom Initialization
        scale_factor = 0.003
        nn.init.uniform_(self.linear_output.weight.data, -scale_factor, scale_factor)
        nn.init.uniform_(self.linear_output.bias.data, -scale_factor, scale_factor)

        # self.bn_test = nn.BatchNorm1d(input_size)

        self.bn = bn
        if self.bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_hidden_layers)])

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic, weight_decay=1e-2)  # weight_decay=1e-2

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        # x = self.bn_test(x)

        x = self.linear_input(x)
        if self.bn:
            x = self.bn[0](x)
        # x = torch.tanh(x)
        x = F.relu(x)

        for i, l in enumerate(self.linear_hidden):
            x = l(x)
            if self.bn:
                x = self.bn[i+1](x)
            # x = torch.tanh(x)
            x = F.relu(x)

        x = self.linear_output(x)

        return x

    def save(self, file_name='critic.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)

        torch.save({'state_dict': self.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, file_name)

    def load(self):
        checkpoint = torch.load("./model/critic_best.pth", map_location=torch.device('cpu'))

        try:
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        except KeyError:
            self.load_state_dict(checkpoint)


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2, bn=False, lr_actor=1e-04):
        super(Actor, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.linear_input = nn.Linear(input_size, hidden_size)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers - 1)])
        self.linear_output = nn.Linear(hidden_size, output_size)

        # Xavier initialization
        # scale_factor = 0.01
        # gain = torch.nn.init.calculate_gain('tanh', param=None)
        # fo = scale_factor * \
        #      gain * np.sqrt(1.0 / (self.linear_output.weight.size()[1] + self.linear_output.weight.size()[0]))
        # nn.init.uniform_(self.linear_output.weight.data, -fo, fo)
        # nn.init.uniform_(self.linear_output.bias.data, -fo, fo)

        # Custom Initialization
        scale_factor = 0.003
        nn.init.uniform_(self.linear_output.weight.data, -scale_factor, scale_factor)
        nn.init.uniform_(self.linear_output.bias.data, -scale_factor, scale_factor)

        # self.bn_test = nn.BatchNorm1d(input_size)

        self.bn = bn
        if self.bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_hidden_layers)])
            self.bn_output = nn.BatchNorm1d(output_size)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)

    def forward(self, state):
        # state = self.bn_test(state)
        x = self.linear_input(state)

        if self.bn:
            x = self.bn[0](x)
        # x = torch.tanh(x)
        x = F.relu(x)

        for i, l in enumerate(self.linear_hidden):
            x = l(x)
            if self.bn:
                x = self.bn[i+1](x)
            # x = torch.tanh(x)
            x = F.relu(x)

        x = self.linear_output(x)
        if self.bn:
            x = self.bn_output(x)
        x = torch.tanh(x)
        return x

    def save(self, file_name='actor.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)

        torch.save({'state_dict': self.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, file_name)

    def load(self):
        checkpoint = torch.load("./model/actor_best.pth", map_location=torch.device('cpu'))

        try:
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        except KeyError:
            self.load_state_dict(checkpoint)


class CriticRes(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bn=False, lr_critic=3e-04):
        super(CriticRes, self).__init__()
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.bn = bn
        if self.bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(3)])

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic, weight_decay=1e-2)  # weight_decay=1e-2

    def forward(self, state, action):
        # Input
        x = torch.cat([state, action], 1)
        x = self.linear0(x)
        if self.bn:
            x = self.bn[0](x)
        x = F.relu(x)

        # Residual Block 1
        residual = x
        x = self.linear1(x)
        if self.bn:
            x = self.bn[1](x)
        x = F.relu(x)
        x = self.linear2(x)
        x = x + residual
        if self.bn:
            x = self.bn[2](x)
        x = F.relu(x)

        # Output
        x = self.linear3(x)
        return x

    def save(self, file_name='critic.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save({'state_dict': self.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, file_name)

    def load(self):
        checkpoint = torch.load("./model/critic.pth", map_location=torch.device('cpu'))

        try:
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        except KeyError:
            self.load_state_dict(checkpoint)


class ActorRes(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bn=False, lr_actor=1e-04):
        super(ActorRes, self).__init__()
        self.linear0 = nn.Linear(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

        self.bn = bn
        if self.bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(3)])
            self.bn_output = nn.BatchNorm1d(output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)

    def forward(self, state):
        # Input
        x = state
        x = self.linear0(x)
        if self.bn:
            x = self.bn[0](x)
        x = F.relu(x)

        # Residual Block 1
        residual = x
        x = self.linear1(x)
        if self.bn:
            x = self.bn[1](x)
        x = F.relu(x)
        x = self.linear2(x)
        x = x + residual
        if self.bn:
            x = self.bn[2](x)
        x = F.relu(x)

        # Output
        x = self.linear3(x)
        if self.bn:
            x = self.bn_output(x)
        x = torch.tanh(x)
        return x

    def save(self, file_name='actor.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save({'state_dict': self.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, file_name)

    def load(self):
        checkpoint = torch.load("./model/actor.pth", map_location=torch.device('cpu'))

        try:
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        except KeyError:
            self.load_state_dict(checkpoint)


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.45, theta=0.1, dt=2e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

    def test_noise(self, interval=100):
        x_list = []
        normal_noise_list = []
        mu = 0
        x_prev = 0
        for i in range(interval):
            normal_noise = np.random.normal(size=1)[0]
            x = x_prev + self.theta * (mu - x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * normal_noise
            x_prev = x
            x_list.append(x)
            normal_noise_list.append(normal_noise)

        # Plot Noise behavior
        plt.plot(range(interval), x_list)
        plt.plot(range(interval), normal_noise_list)
        plt.legend(["Ornstein Noise", "Normal Noise"])
        plt.show(block=True)


def compare_models(model_1, model_2):
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            print('Models mismatch')
            return
    print('Models match perfectly! :)')


class DDPGTrainer:
    def __init__(self, actor, critic, gamma):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.actor = actor
        self.critic = critic
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.batch_num = 0

    def update_network_parameters(self, tau=0.001):  # tau=0.03 works best
        # Network params
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.actor_target.named_parameters()
        target_critic_params = self.critic_target.named_parameters()

        critic_params_dict = dict(critic_params)
        actor_params_dict = dict(actor_params)
        target_critic_params_dict = dict(target_critic_params)
        target_actor_params_dict = dict(target_actor_params)

        # Network buffers
        actor_buffers = self.actor.named_buffers()
        critic_buffers = self.critic.named_buffers()
        target_actor_buffers = self.actor_target.named_buffers()
        target_critic_buffers = self.critic_target.named_buffers()

        critic_buffers_dict = dict(critic_buffers)
        actor_buffers_dict = dict(actor_buffers)
        target_critic_buffers_dict = dict(target_critic_buffers)
        target_actor_buffers_dict = dict(target_actor_buffers)

        # import ipdb; ipdb.set_trace(context=10)

        # Update params
        for name in critic_params_dict:
            critic_params_dict[name] = tau * critic_params_dict[name].clone() + \
                                      (1 - tau) * target_critic_params_dict[name].clone()

        for name in actor_params_dict:
            actor_params_dict[name] = tau * actor_params_dict[name].clone() + \
                                     (1 - tau) * target_actor_params_dict[name].clone()

        # Update buffers
        for name in critic_buffers_dict:
            critic_params_dict[name] = tau * critic_buffers_dict[name].clone() + \
                                      (1 - tau) * target_critic_buffers_dict[name].clone()

        for name in actor_buffers_dict:
            actor_params_dict[name] = tau * actor_buffers_dict[name].clone() + \
                                     (1 - tau) * target_actor_buffers_dict[name].clone()

        self.critic_target.load_state_dict(critic_params_dict)
        self.actor_target.load_state_dict(actor_params_dict)

    def train_step(self, states, actions, rewards, next_states, dones):
        # print("states shape: {}, actions shape: {}, rewards shape: {}, next_states shape: {}, dones shape: {}".format(
        #     states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape
        # ))
        states = torch.tensor(states, dtype=torch.float).to(self.device)  # [n, n_states]
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)  # [n, n_actions]
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)  # [n]
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)  # [n, n_states]
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)  # [n]

        # Normalize rewards
        # if len(rewards) > 1:
            # print(rewards)
            # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)  # normalize discounted rewards

        self.actor_target.eval()
        self.actor.eval()
        self.critic_target.eval()
        self.critic.eval()

        target_actions = self.actor_target.forward(next_states)
        target_critic_value = self.critic_target.forward(next_states, target_actions)
        critic_value = self.critic.forward(states, actions).squeeze(1)

        target = []
        for idx in range(len(dones)):
            if not dones[idx]:
                target.append(rewards[idx] + self.gamma * target_critic_value[idx][0])
            else:  # If done
                target.append(rewards[idx])

        target = torch.stack(target).to(self.device)

        # Step Critic
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()

        # Step Actor
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(states)
        self.actor.train()
        actor_loss = -self.critic.forward(states, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.batch_num += 1

        # Update the weights of target networks
        self.update_network_parameters()

        # for n, p in enumerate(self.actor.parameters()):
            # print(n)
            # print(p)
            # if n == 9:
            #     print(p)


if __name__ == "__main__":
    # Test Noise:
    # noise = OUActionNoise(mu=0, sigma=0.8, dt=3e-2, theta=0.1, x0=None)
    noise = OUActionNoise(mu=0, sigma=1.0, dt=4e-2, theta=0.0, x0=None)
    noise.test_noise(interval=10)
