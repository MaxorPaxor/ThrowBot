import numpy as np
import torch
import torch.nn as nn
import os


class MLPBCModel(nn.Module):

    """
    Simple MLP that predicts next action a from past states s.
    """

    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.predict_action_activation_m = nn.Tanh()
        self.predict_action_activation_g = nn.Sigmoid()

        layers = [nn.Linear(max_length*self.state_dim, hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim),
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states, actions, rewards, attention_mask=None, target_return=None):

        states = states[:,-self.max_length:].reshape(states.shape[0], -1)  # concat states
        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)
        actions[:, :, :-1] = self.predict_action_activation_m(actions[:, :, :-1])
        actions[:, :, -1] = self.predict_action_activation_g(actions[:, :, -1])

        return None, actions, None

    def get_action(self, states, actions, rewards, target_return, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        if states.shape[1] < self.max_length:
            states = torch.cat(
                [torch.zeros((1, self.max_length-states.shape[1], self.state_dim),
                             dtype=torch.float32, device=states.device), states], dim=1)
        states = states.to(dtype=torch.float32)
        _, actions, _ = self.forward(states, None, None, **kwargs)
        return actions[0,-1]

    def save(self, file_name='dt.pth'):
        model_folder_path = 'weights'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save({'state_dict': self.state_dict()}, file_name)

    def load(self):
        checkpoint = torch.load("./weights/dt.pth", map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['state_dict'])