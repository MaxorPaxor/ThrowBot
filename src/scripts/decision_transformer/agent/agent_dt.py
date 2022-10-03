import torch
import random
import numpy as np
from collections import deque

from agent.model_dt.model_dt import DecisionTransformer
from agent.model_dt.noise import OUActionNoise


class Agent:
    def __init__(self, arm, load_nn=False, load_mem=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(self.device))

        # Arm
        self.arm = arm

        # HER
        self.her = arm.her
        self.k = 0
        self.generate_targets_factor_radius = 2.0

        # Dynamics
        self.max_length = int(arm.number_steps)
        self.no_rotation = arm.no_rotation
        self.n_actions = len(self.arm.joints)
        self.n_states = self.n_actions * self.arm.number_states
        if self.her:
            self.n_states += 1

        # Buffer
        self.MAX_MEMORY = 100_000
        self.memory_raw = deque(maxlen=int(self.MAX_MEMORY))  # popleft()
        self.memory_k0 = deque(maxlen=int(self.MAX_MEMORY))  # popleft()
        self.memory_k1 = deque(maxlen=int(self.MAX_MEMORY))  # popleft()
        self.memory_k3 = deque(maxlen=int(self.MAX_MEMORY))  # popleft()
        self.memory_k5 = deque(maxlen=int(self.MAX_MEMORY))  # popleft()
        self.memory_k11 = deque(maxlen=int(self.MAX_MEMORY))  # popleft()
        self.transitions = 0
        # if load_mem:
        #     import pickle
        #     self.memory = pickle.load(open('data/memory.pkl', 'rb'))

        # Exploration
        self.noise_strong = OUActionNoise(mu=np.zeros(self.n_actions), sigma=0.6, dt=2e-1, theta=0)  # dt=4e-2
        self.noise_soft = OUActionNoise(mu=np.zeros(self.n_actions), sigma=0.4, dt=2e-2, theta=0.1)
        self.exploration_flag = True
        self.epsilon_arm = 30  # 50
        self.soft_exploration_rate = 50
        self.epsilon_arm_decay = 1e-05
        self.exploration_open_gripper = 0
        self.update_exploration()

        # Learning Params
        self.LR = 1e-04
        self.BATCH_SIZE = 128
        self.num_mini_batches_per_training = 100  # 40
        self.train_every_n_episode = 100  # 16
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
            self.epsilon_arm = 3
            # self.soft_exploration_rate = 20

            # Decide if next episode will have exploration
        if random.randint(0, 100) <= self.epsilon_arm:
            self.exploration_flag = True
            self.exploration_open_gripper = random.randint(0, int(self.arm.number_steps - 1))
        else:
            self.exploration_flag = False

    def remember(self, trajectory, k):
        """
        Add Markov chain to the buffer
        """

        if k == -1:
            self.memory_raw.append(trajectory)
        if k == 0:
            self.memory_k0.append(trajectory)  # popleft if MAX_MEMORY is reached
        if k == 1:
            self.memory_k1.append(trajectory)  # popleft if MAX_MEMORY is reached
        if k == 3:
            self.memory_k3.append(trajectory)  # popleft if MAX_MEMORY is reached
        if k == 5:
            self.memory_k5.append(trajectory)  # popleft if MAX_MEMORY is reached
        if k == 11:
            self.memory_k11.append(trajectory)  # popleft if MAX_MEMORY is reached

        self.transitions += len(trajectory)

    def forget(self):
        """
        Clear buffer
        """

        self.memory_k0.clear()
        self.memory_k1.clear()
        self.memory_k3.clear()
        self.memory_k5.clear()
        self.memory_k11.clear()

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

    def get_action_random(self):
        """
        Inference state and get action from the policy.
        Generate noise and use exploration methods
        """

        noise_strong = torch.tensor(self.noise_strong(), dtype=torch.float).to(self.device)
        mu_prime = noise_strong

        if self.episode_length >= self.exploration_open_gripper:
            mu_prime[-1] = -1.0  # Open gripper
        else:
            mu_prime[-1] = 1.0  # Keep gripper closed

        final_move = torch.clip(mu_prime, min=-1, max=1)
        return final_move

    def generate_her_memory(self, trajectory, target, obj_final_pos, k=None):
        """
        Hindsight Experience Replay (HER) modification to the current buffer.
        Adding actual (state||target, action, reward, state_new||target, done) tuple together with a modified one:
        (state||target`, action, reward`, state_new||target`, done)
        """

        if k is None:
            k = self.k

        # Create target list
        if k == -1:
            target_list = [target]

        elif k == 0:
            target_list = [obj_final_pos]

        else:
            target_list = [obj_final_pos]
            # Create another k-1 targets
            for _ in range(k - 1):
                rand = np.random.rand() * 2 - 1  # Random [-1, 1]
                x = obj_final_pos[0] + self.arm.target_radius * rand * self.generate_targets_factor_radius
                if x > 0:
                    new_target = np.array([x, 0, 0])
                    target_list.append(new_target)

        # Create new memory buffer
        for trg in target_list:
            new_trajectory = []
            for old_tuple in trajectory:  # (state, action, reward, done, success)
                state = np.append(old_tuple[0], trg[0])
                action = old_tuple[1]
                done = old_tuple[3]

                if len(old_tuple) == 5:
                    success = old_tuple[4]
                    if success:
                        reward = self.arm.reward_sparse(obj_pos=obj_final_pos, target=trg)
                    else:
                        reward = old_tuple[2]

                else:
                    if done:
                        reward = self.arm.reward_sparse(obj_pos=obj_final_pos, target=trg)
                    else:
                        reward = old_tuple[2]

                new_trajectory.append([state, action, reward, done])

            self.remember(new_trajectory, k)

    @staticmethod
    def generate_target_her():
        """
        Generates random target in [0.5, 2.0] meters range
        """

        x = np.random.rand() * 1.5 + 0.5
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


