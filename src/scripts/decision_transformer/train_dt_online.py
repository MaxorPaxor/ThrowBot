import torch
import time
from collections import deque
import numpy as np
from datetime import timedelta

from evaluate_dt import eval_model
from robot_env_dt import RoboticArm
from agent_dt import Agent
from utils import Plot


# LOAD_NN = True
# LOAD_MEMORY = True
LOAD_NN = False
LOAD_MEMORY = False


def train_online():
    """
        Main Loop
        """

    # Metrics
    reward_list = []
    mean_reward_list = []
    distance_from_target_list = []
    mean_distance_from_target_list = []

    # Evaluation
    mean_distance_evaluated_list = []
    mean_distance_eval = None
    evaluated_epochs_list = []

    arm = RoboticArm()
    agent = Agent(arm=arm, load_nn=LOAD_NN, load_mem=LOAD_MEMORY)
    plotter = Plot()

    start_time = time.time()
    total_time = 0
    avg_episode_time = 0

    states = torch.zeros((0, agent.model.state_dim), device=agent.device, dtype=torch.float32)
    actions = torch.zeros((0, agent.model.act_dim), device=agent.device, dtype=torch.float32)
    rewards = torch.zeros(0, device=agent.device, dtype=torch.float32)
    ep_return = 1.
    target_return = torch.tensor(ep_return, device=agent.device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=agent.device, dtype=torch.long).reshape(1, 1)


    target = agent.generate_target_her()
    arm.update_target(target)
    trajectory = []

    while True:
        state_ = arm.get_state()  # get state
        state = np.append(state_, arm.target[0])  # append target
        state = torch.from_numpy(state).reshape(1, agent.model.state_dim).to(device=agent.device, dtype=torch.float32)

        states = torch.cat([states, state], dim=0)
        actions = torch.cat([actions, torch.zeros((1, agent.model.act_dim), device=agent.device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=agent.device)])

        action = agent.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )

        actions[-1] = action
        action = action.detach().cpu().numpy()

        reward, done, termination_reason, obj_pos, success = arm.step(action)  # perform action and get new state
        rewards[-1] = reward
        target_return = torch.cat([target_return, target_return[0, -1].reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=agent.device, dtype=torch.long) * (agent.episode_length + 1)], dim=1)

        trajectory.append([state_, action, reward, done, success])
        # [(state, action, reward, done)_i]

        agent.episode_length += 1

        if done:
            # print("states:")
            # print(states.to(dtype=torch.float32))
            # print("actions:")
            # print(actions.to(dtype=torch.float32))
            # print("target_return:")
            # print(target_return.to(dtype=torch.float32))
            # print("timesteps:")
            # print(timesteps.to(dtype=torch.long))

            arm.reset()
            agent.noise_soft.reset()
            agent.noise_strong.reset()
            agent.n_episodes += 1

            agent.generate_her_memory(trajectory, obj_final_pos=obj_pos)

            if reward > agent.record:
                agent.record = reward

            # Rewards statistics
            reward_list.append(round(reward, 4))
            reward_list_np = np.array(reward_list)
            mean_reward = np.round(np.mean(reward_list_np), 4)
            std_reward = np.round(np.std(reward_list_np), 4)
            mean_reward_list.append(mean_reward)

            # Distances statistics
            distance_from_goal = agent.calc_dist_from_goal(obj_pos, arm.target)
            distance_from_target_list.append(distance_from_goal)
            distance_from_target_list_np = np.array(distance_from_target_list)
            mean_distance = np.round(np.mean(distance_from_target_list_np), 4)
            std_distance = np.round(np.std(distance_from_target_list_np), 4)
            mean_distance_from_target_list.append(mean_distance)

            # Evaluation
            if agent.num_epoch % agent.evaluate_every_n_epoch == 0 and agent.num_epoch > agent.last_evaluated_epoch:
                mean_distance_eval = eval_model(arm=arm, model=agent.model, evaluation_episodes=20, print_info=False)
                mean_distance_evaluated_list.append(mean_distance_eval)
                evaluated_epochs_list.append(agent.num_epoch)
                agent.last_evaluated_epoch = agent.num_epoch

            info = (f' {"GENERAL":30}\n'
                    f' {"  Termination reason:":30} {termination_reason}\n'
                    f' {"  Episode:":30} {agent.n_episodes}\n'
                    f' {"  Episode length:":30} {agent.episode_length}\n'
                    f' {"REWARDS":30}\n'
                    f' {"  Reward:":30} {round(reward, 4)}\n'
                    f' {"  Record reward:":30} {round(agent.record, 4)}\n'
                    f' {"  Mean reward:":30} {mean_reward}\n'
                    f' {"  Variance reward:":30} {std_reward}\n'
                    f' {"DISTANCE":30}\n'
                    f' {"  Target X:":30} {round(arm.target[0], 4)}\n'
                    f' {"  Final object X position:":30} {round(obj_pos[0], 4)}\n'
                    f' {"  Distance From target:":30} {round(distance_from_goal, 4)}\n'
                    f' {"  Mean distance:":30} {round(mean_distance, 4)}\n'
                    f' {"  Variance distance:":30} {round(std_distance, 4)}\n'
                    f' {"EVALUATION":30}\n'
                    f' {"  Mean evaluation distance:":30} {mean_distance_eval}\n'
                    f' {"  Best evaluation distance:":30} {agent.best_evaluation_distance}\n'
                    f' {"  Last evaluated epoch:":30} {agent.last_evaluated_epoch}\n'
                    f' {"EXPLORATION":30}\n'
                    f' {"  Exploration:":30} {agent.exploration_flag}\n'
                    f' {"  Exploration rate:":30} {round(agent.epsilon_arm, 4)}\n'
                    f' {"TRAINING":30}\n'
                    f' {"  Epoch number:":30} {agent.num_epoch}\n'
                    f' {"  Mini-batches per training:":30} {agent.num_mini_batches_per_training}\n'
                    f' {"  Train every n episodes:":30} {agent.train_every_n_episode}\n'
                    f' {"  Memory:":30} {len(agent.memory)}/{agent.MAX_MEMORY}\n'
                    f' {"  Batch Size:":30} {agent.BATCH_SIZE}\n'
                    f' {"  Total time passed:":30} {str(timedelta(seconds=total_time))}\n'
                    f' {"  AVG time per episode:":30} {str(timedelta(seconds=avg_episode_time))}\n')
            print(info)
            print('=======================================================')

            agent.train_dt()
            plotter.plot(reward_list, mean_reward_list, mean_distance_from_target_list,
                         mean_distance_evaluated_list, evaluated_epochs_list)
            agent.save(eval_distance=mean_distance_eval)
            agent.update_exploration()
            agent.episode_length = 0

            target = agent.generate_target_her()
            arm.update_target(target)
            trajectory = []
            states = torch.zeros((0, agent.model.state_dim), device=agent.device, dtype=torch.float32)
            actions = torch.zeros((0, agent.model.act_dim), device=agent.device, dtype=torch.float32)
            rewards = torch.zeros(0, device=agent.device, dtype=torch.float32)
            ep_return = 1.
            target_return = torch.tensor(ep_return, device=agent.device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=agent.device, dtype=torch.long).reshape(1, 1)

            total_time = time.time() - start_time
            avg_episode_time = total_time / agent.n_episodes


if __name__ == '__main__':
    train_online()
