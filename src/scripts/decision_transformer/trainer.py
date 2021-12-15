import time
import torch
import numpy as np


class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.diagnostics = dict()

        self.start_time = time.time()
        self.step_number = 0

    def train_iteration(self, num_steps):

        train_losses = []
        for i in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if i % 10 == 0:
                print(f"Step number :{i}, loss: {train_loss}")

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch()
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.model.train()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        self.step_number += 1

        return loss.detach().cpu().item()

