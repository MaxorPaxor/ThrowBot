import torch
import random
import numpy as np
from collections import deque
import pickle

from model.model import Critic, Actor, DDPGTrainer
from model.noise import OUActionNoise


class Agent:
    def __init__(self, arm, load_nn=False, load_mem=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(self.device))

        # Arm
        self.arm = arm

        # HER
        self.her = arm.her
        self.k = 10
        self.generate_targets_factor_radius = 1.0

        # Dynamics
        self.max_length = int(arm.number_steps)
        self.no_rotation = arm.no_rotation
        self.n_actions = len(self.arm.joints)
        self.n_states = self.n_actions * self.arm.number_states
        if self.her:
            self.n_states += 1

        # Buffer
        self.MAX_MEMORY = 100_000
        self.memory = deque(maxlen=int(self.MAX_MEMORY))  # popleft()
        if load_mem:
            self.memory = pickle.load(open('data/memory_random_10k_3s.pkl', 'rb'))
            # self.memory = deque(itertools.islice(self.memory, 0, int(0.85*len(self.memory))))

        # Exploration
        self.noise_strong = OUActionNoise(mu=np.zeros(self.n_actions), sigma=0.6, dt=2e-1, theta=0)  # dt=4e-2
        self.noise_soft = OUActionNoise(mu=np.zeros(self.n_actions), sigma=0.4, dt=2e-2, theta=0.1)
        self.exploration_flag = True
        self.epsilon_arm = 0  # 50
        self.soft_exploration_rate = 50
        self.epsilon_arm_decay = 1e-05
        self.exploration_open_gripper = 0
        self.update_exploration()

        # Learning Params
        self.LR_actor = 1e-04
        self.LR_critic = 1e-04  # 1e-04
        self.gamma = 0.95  # discount rate
        self.BATCH_SIZE = 128
        self.num_mini_batches_per_training = 100  # 40
        self.train_every_n_episode = 40  # 16
        self.n_episodes = 0
        self.episode_length = 0
        self.num_epoch = 0
        self.record = -1.0
        self.best_mean_reward = -1.0
        self.best_mean_distance = 10

        # Online Evaluation
        self.evaluate_every_n_epoch = 1
        self.last_evaluated_epoch = 0
        self.best_evaluation_distance = None

        # Models
        # Actor
        self.actor = Actor(self.n_states, 64, self.n_actions, num_hidden_layers=2, bn=False, lr_actor=self.LR_actor)
        # self.actor = ActorRes(self.n_states, 128, self.n_actions, bn=False)

        # Critic
        self.critic = Critic(self.n_states + self.n_actions, 256, 1, num_hidden_layers=2, bn=False, lr_critic=self.LR_critic)
        # self.critic = CriticRes(self.n_states + self.n_actions, 128, 1, bn=False)

        if load_nn:
            self.actor.load()
            self.critic.load()

        # Trainer
        self.trainer_ddpg = DDPGTrainer(self.actor, self.critic, gamma=self.gamma)

    def update_exploration(self):
        """
        Update exploration parameters between episodes:
        Decay exploration rate, decide on opening moment
        """

        if self.epsilon_arm > 3:
            self.epsilon_arm = self.epsilon_arm * (1 - self.epsilon_arm_decay)
        else:
            self.epsilon_arm = 0

            # Decide if next episode will have exploration
        if random.randint(0, 100) <= self.epsilon_arm:
            self.exploration_flag = True
            self.exploration_open_gripper = random.randint(0, int(self.arm.number_steps - 1))
        else:
            self.exploration_flag = False

    def remember(self, state, action, reward, next_state, done):
        """
        Add Markov chain to the buffer
        """

        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

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
            self.actor.save(file_name='actor.pth')
            self.critic.save(file_name='critic.pth')
            pickle.dump(self.memory, open('data/memory.pkl', 'wb'))

        if self.best_evaluation_distance is None:
            self.best_evaluation_distance = eval_distance
        elif eval_distance < self.best_evaluation_distance:
            self.best_evaluation_distance = eval_distance
            self.actor.save(file_name='actor_best.pth')
            self.critic.save(file_name='critic_best.pth')
            pickle.dump(self.memory, open('./data/memory_best.pkl', 'wb'))

    def train_ddpg(self):
        """
        Call for a DDPG train step
        """

        if len(self.memory) > self.BATCH_SIZE and self.n_episodes % self.train_every_n_episode == 0:
            for i in range(self.num_mini_batches_per_training):
                mini_sample = random.sample(self.memory, self.BATCH_SIZE)  # list of tuples
                states, actions, rewards, next_states, dones = zip(*mini_sample)
                self.trainer_ddpg.train_step(states, actions, rewards, next_states, dones)
            self.num_epoch += 1

    def get_action(self, state):
        """
        Inference state and get action from the policy.
        Generate noise and use exploration methods
        """

        self.actor.eval()

        state = np.append(state, self.arm.target[0])
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        state_tensor = torch.unsqueeze(state_tensor, 0)

        action = self.actor.forward(state_tensor).to(self.device)
        action = torch.squeeze(action)

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

            if random.randint(0, 100) < self.soft_exploration_rate:

                if mu_prime[-1] < 0:
                    mu_prime[-1] = 1.0

                elif self.episode_length + 1 >= self.arm.number_steps:
                    mu_prime[-1] = -1.0

        final_move = mu_prime.cpu().detach().numpy()
        final_move = np.clip(final_move, a_min=-1, a_max=1)
        return final_move

    def get_action_eval(self, state):
        """
        Inference state and get action from the policy.
        Use this for evaluation, without noise and exploration.
        """

        self.actor.eval()

        if self.her:
            # state = np.concatenate((state, self.arm.target[0]))
            state = np.append(state, self.arm.target[0])

        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        state_tensor = torch.unsqueeze(state_tensor, 0)

        action = self.actor.forward(state_tensor).to(self.device)
        action = torch.squeeze(action)

        final_move = action.cpu().detach().numpy()
        final_move = np.clip(final_move, a_min=-1, a_max=1)
        return final_move

    def get_action_eval_noise(self, state, arm, noise='soft'):
        """
        Inference state and get action from the policy.
        Use this for evaluation, with noise but without exploration.
        This is used mainly to evaluate noise statistics.
        """

        self.actor.eval()
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        mu = self.actor.forward(state_tensor).to(self.device)

        if noise == 'soft':
            noise_soft = torch.tensor(self.noise_soft(), dtype=torch.float).to(self.device)
            mu_prime = mu + noise_soft
            mu_prime[-1] = mu[-1]

        else:
            noise_strong = torch.tensor(self.noise_strong(), dtype=torch.float).to(self.device)
            mu_prime = mu + noise_strong

            if self.episode_length == arm.number_steps:
                mu_prime[-1] = -1.0  # Open gripper at last step
            else:
                mu_prime[-1] = 1.0  # Keep gripper closed

        final_move = mu_prime.cpu().detach().numpy()
        final_move = np.clip(final_move, a_min=-1, a_max=1)
        return final_move

    def generate_her_memory(self, tmp_mem, obj_final_pos):
        """
        Hindsight Experience Replay (HER) modification to the current buffer.
        Adding actual (state||target, action, reward, state_new||target, done) tuple together with a modified one:
        (state||target`, action, reward`, state_new||target`, done)
        """

        # Create target list
        if self.k == -1:
            target_list = [self.arm.target]

        elif self.k == 0:
            target_list = [self.arm.target, obj_final_pos]

        else:
            target_list = [self.arm.target, obj_final_pos]
            # Create another k-1 targets
            for _ in range(self.k - 1):
                rand = np.random.rand() * 2 - 1  # Random [-1, 1]
                # rand = np.random.normal()
                # rand = np.clip(rand, a_max=1, a_min=-1)
                x = obj_final_pos[0] + self.arm.target_radius * rand * self.generate_targets_factor_radius
                if x > 0:
                    new_target = np.array([x, 0, 0])
                    target_list.append(new_target)

        # Create new memory buffer
        for trg in target_list:
            # print(f"Target: {trg}")

            for old_tuple in tmp_mem:  # (state, action, reward, state_new, done, success)
                state = np.append(old_tuple[0], trg[0])
                action = old_tuple[1]
                success = old_tuple[5]
                if success:
                    reward = self.arm.reward_sparse(obj_pos=obj_final_pos, target=trg)
                else:
                    reward = old_tuple[2]

                # state_new = np.concatenate((old_tuple[3], trg))
                state_new = np.append(old_tuple[3], trg[0])
                done = old_tuple[4]

                # DEBUG PRINTS
                # print(f"State: {state}")
                # print(f"Action: {action}")
                # print(f"Success: {success}")
                # print(f"Reward: {reward}\n")
                # print(f"New State: {state_new}\n")
                # print(f"Done: {done}\n")

                self.remember(state, action, reward, state_new, done)

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


