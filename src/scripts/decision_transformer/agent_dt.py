import torch
import random
import numpy as np
from collections import deque
import pickle

from model_dt import DecisionTransformer
from noise import OUActionNoise
from trainer import Trainer


class Agent:
    def __init__(self, arm, load_nn=False, load_mem=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(self.device))

        # Arm
        self.arm = arm

        # HER
        self.her = arm.her
        self.k = 4
        self.generate_targets_factor_radius = 1.0

        # Dynamics
        self.max_length = int(arm.number_steps)
        self.no_rotation = arm.no_rotation
        self.n_actions = len(self.arm.joints)
        self.n_states = self.n_actions * self.arm.number_states
        if self.her:
            self.n_states += 1

        # Buffer
        self.MAX_MEMORY = 50_000
        self.memory = deque(maxlen=int(self.MAX_MEMORY))  # popleft()
        if load_mem:
            self.memory = pickle.load(open('data/memory.pkl', 'rb'))

        # Exploration
        self.noise_strong = OUActionNoise(mu=np.zeros(self.n_actions), sigma=0.8, dt=4e-2, theta=0.0)
        # self.noise_strong = OUActionNoise(mu=np.zeros(self.n_actions), sigma=0.0, dt=0e-2, theta=0.0)
        self.noise_soft = OUActionNoise(mu=np.zeros(self.n_actions), sigma=0.4, dt=2e-2, theta=0.1)
        # self.noise_soft = OUActionNoise(mu=np.zeros(self.n_actions), sigma=0.0, dt=0e-2, theta=0.0)
        self.exploration_flag = True
        self.epsilon_arm = 0  # 50
        self.soft_exploration_rate = 50
        self.epsilon_arm_decay = 1e-05
        self.exploration_open_gripper = 0
        self.update_exploration()

        # Learning Params
        self.LR = 3e-04
        self.BATCH_SIZE = 128
        self.num_mini_batches_per_training = 40  # 40
        self.train_every_n_episode = 16  # 16
        self.n_episodes = 0
        self.episode_length = 0
        self.num_epoch = 0
        self.record = -1.0
        self.best_mean_reward = -1.0
        self.best_mean_distance = 10

        # Online Evaluation
        self.evaluate_every_n_epoch = 10
        self.last_evaluated_epoch = 0
        self.best_evaluation_distance = None

        # Models
        embed_dim = 128
        n_layer = 3
        n_head = 1
        activation_function = 'relu'
        dropout = 0.1
        n_positions = 1024  # 1024

        self.model = DecisionTransformer(
            state_dim=self.n_states,
            act_dim=self.n_actions,
            max_length=self.max_length,
            max_ep_len=self.max_length,
            hidden_size=embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * embed_dim,
            activation_function=activation_function,
            n_positions=n_positions,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.model = self.model.to(device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.LR)  # weight_decay=1e-4

        if load_nn:
            self.model.load()

    def update_exploration(self):
        """
        Update exploration parameters between episodes:
        Decay exploration rate, decide on opening moment
        """

        if self.epsilon_arm > 3:
            self.epsilon_arm = self.epsilon_arm * (1 - self.epsilon_arm_decay)
            # self.soft_exploration_rate = self.soft_exploration_rate * (1 - self.epsilon_arm_decay)
        else:
            self.epsilon_arm = 0
            # self.soft_exploration_rate = 20

            # Decide if next episode will have exploration
        if random.randint(0, 100) < self.epsilon_arm:
            self.exploration_flag = True
            self.exploration_open_gripper = random.randint(0, int(self.arm.number_steps - 1))
        else:
            self.exploration_flag = False

    def remember(self, trajectory):
        """
        Add Markov chain to the buffer
        """

        self.memory.append(trajectory)  # popleft if MAX_MEMORY is reached

    def forget(self):
        """
        Clear buffer
        """

        self.memory.clear()

    def save(self, eval_distance):
        """
        Save weights and memory every n epochs
        """

        if self.num_epoch % 10 == 0:
            self.model.save(file_name='model.pth')
            pickle.dump(self.memory, open('data/memory.pkl', 'wb'))

        if self.best_evaluation_distance is None:
            self.best_evaluation_distance = eval_distance
        elif eval_distance < self.best_evaluation_distance:
            self.best_evaluation_distance = eval_distance
            self.model.save(file_name='model_best.pth')
            pickle.dump(self.memory, open('./data/memory_best.pkl', 'wb'))

    def get_batch(self):

        batch_inds = np.random.choice(
            np.arange(len(self.memory)),
            size=self.BATCH_SIZE,
            replace=True
        )

        s, a, r, d, rtg, ts, mask = [], [], [], [], [], [], []

        for i in range(self.BATCH_SIZE):
            traj = self.memory[batch_inds[i]]
            # (state, action, reward, done)

            # get sequences from dataset
            s.append(np.array([t[0] for t in traj]).reshape(1, -1, self.n_states))
            a.append(np.array([t[1] for t in traj]).reshape(1, -1, self.n_actions))
            r.append(np.array([t[2] for t in traj]).reshape(1, -1, 1))
            # rtg.append(np.array([traj[-1][2]] * (len(traj))).reshape(1, -1, 1))
            rtg.append(np.array([traj[-1][2]] * (len(traj)-1) + [0]).reshape(1, -1, 1))
            d.append(np.array([t[3] for t in traj]).reshape(1, -1))
            ts.append(np.array([t for t in range(len(traj))]).reshape(1, -1))

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_length - tlen, self.n_states)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std
            # a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            a[-1] = np.concatenate([np.zeros((1, self.max_length - tlen, self.n_actions)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, self.max_length - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_length - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_length - tlen, 1)), rtg[-1]], axis=1)  # / scale
            ts[-1] = np.concatenate([np.zeros((1, self.max_length - tlen)), ts[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_length - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        ts = torch.from_numpy(np.concatenate(ts, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return s, a, r, d, rtg, ts, mask

    def train_dt(self):
        """
        Call for a DDPG train step
        """

        if len(self.memory) > self.BATCH_SIZE and self.n_episodes % self.train_every_n_episode == 0:
            for _ in range(self.num_mini_batches_per_training):

                trainer = Trainer(
                    model=self.model,
                    optimizer=self.optimizer,
                    batch_size=self.BATCH_SIZE,
                    get_batch=self.get_batch,
                    loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2)
                )
                trainer.train_iteration(num_steps=100)
            self.num_epoch += 1

    def get_action(self, states, actions, rewards, target_return, timesteps):
        """
        Inference state and get action from the policy.
        Generate noise and use exploration methods
        """

        self.model.eval()

        action = self.model.get_action(
            states,
            actions,
            rewards,
            target_return,
            timesteps,
        )  # get action

        noise_soft = torch.tensor(self.noise_soft(), dtype=torch.float).to(self.device)
        noise_strong = torch.tensor(self.noise_strong(), dtype=torch.float).to(self.device)

        if self.exploration_flag:  # Strong exploration
            mu_prime = noise_strong

            if self.episode_length >= self.exploration_open_gripper:
                mu_prime[-1] = -1.0  # Open gripper
            else:
                mu_prime[-1] = 1.0  # Keep gripper closed

        else:  # Soft exploration
            mu_prime = action + noise_soft
            mu_prime[-1] = action[-1]

            if random.randint(0, 100) <= self.soft_exploration_rate:

                if mu_prime[-1] < 0:
                    mu_prime[-1] = 1.0

                elif self.episode_length + 1 >= self.arm.number_steps:
                    mu_prime[-1] = -1.0

        # final_move = mu_prime.cpu().detach().numpy()
        final_move = torch.clip(mu_prime, min=-1, max=1)
        return final_move

    def generate_her_memory(self, trajectory, obj_final_pos):
        """
        Hindsight Experience Replay (HER) modification to the current buffer.
        Adding actual (state||target, action, reward, state_new||target, done) tuple together with a modified one:
        (state||target`, action, reward`, state_new||target`, done)
        """
        target_list = [self.arm.target, obj_final_pos]

        # Create target list
        for _ in range(self.k - 1):
            rand = np.random.rand() * 2 - 1  # Random [-1, 1]
            x = obj_final_pos[0] + self.arm.target_radius * rand * self.generate_targets_factor_radius
            if x > 0:
                new_target = np.array([x, 0, 0])
                target_list.append(new_target)

        # Create new memory buffer
        for trg in target_list:
            print(f"Target: {trg}")
            new_trajectory = []
            for old_tuple in trajectory:  # (state, action, reward, done, success)
                state = np.append(old_tuple[0], trg[0])
                action = old_tuple[1]
                success = old_tuple[4]
                if success:
                    reward = self.arm.reward_sparse(obj_pos=obj_final_pos, target=trg)
                else:
                    reward = old_tuple[2]
                done = old_tuple[3]

                new_trajectory.append([state, action, reward, done])

            print(new_trajectory)
            self.remember(new_trajectory)

    @staticmethod
    def generate_target_her():
        """
        Generates random target in [0.5, 2.5] meters range
        """

        x = np.random.rand() * 2 + 0.5
        target = np.array([x, 0.0, 0.0])
        return target

    @staticmethod
    def calc_dist_from_goal(obj_pos, target):
        """
        Calculates euclidean distance between final object position and the target
        """

        distance = np.sqrt((obj_pos[0] - target[0]) ** 2 +
                           (obj_pos[1] - target[1]) ** 2)

        return distance


