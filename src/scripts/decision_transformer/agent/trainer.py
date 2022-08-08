import time
import torch

from torch.nn import CrossEntropyLoss, MSELoss, BCELoss


class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, device, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.device = device
        self.scheduler = scheduler

        self.start_time = time.time()
        self.step_number = 0

    def loss_fn(self, a_pred, a_real):
        # Classic Loss
        # loss_states = torch.mean((a_pred - a_real) ** 2)
        # loss_gripper = 0

        # Smart Loss
        loss_motors = MSELoss()(
            a_pred[:, :-1],
            a_real[:, :-1])

        # Convert labels from range [-1, 1] to [0, 1]
        z = torch.zeros(a_real[:, -1].shape).to(self.device)
        a_real_act = torch.where(a_real[:, -1] == -1., z, a_real[:, -1])
        a_pred_new = a_pred[:, -1]

        loss_gripper = BCELoss()(
            a_pred_new,
            a_real_act)

        return 0.5*loss_gripper + loss_motors

    def train_iteration(self, num_steps, bc=False):
        train_losses = []
        train_loss = None
        for i in range(num_steps):
            if bc:
                train_loss = self.train_step_bc()
            else:
                train_loss = self.train_step()
            train_losses.append(train_loss)
            self.scheduler.step()
            print("\r \rTraining... step number {0}/{1}".format(str(i+1), str(num_steps)), end='')

        print(" Done.")
        return train_loss

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(batch_size=self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(action_preds, action_target)

        self.model.train()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        self.step_number += 1

        return loss.detach().cpu().item()

    def train_step_bc(self):
        states, actions, rewards, dones, rtg, _, attention_mask = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask, target_return=rtg[:, 0],
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:, -1].reshape(-1, act_dim)

        # loss_fn_bc = lambda a_hat, a: torch.mean((a_hat - a) ** 2)
        loss = self.loss_fn(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_number += 1

        return loss.detach().cpu().item()

