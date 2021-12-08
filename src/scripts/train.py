import numpy as np
import time
from datetime import timedelta

from agent import Agent
from robot_env import RoboticArm
from utils import Plot
from eval_model import eval_model


# LOAD_NN = True
# LOAD_MEMORY = True
LOAD_NN = False
LOAD_MEMORY = False


def train():
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

    if agent.her:
        target = agent.generate_target_her()
        arm.update_target(target)
        temp_mem = []

    while True:
        state = arm.get_3_state()  # get state
        action = agent.get_action(state)  # get action
        reward, done, termination_reason, obj_pos, success = arm.step(action)  # perform action and get new state
        state_new = arm.get_3_state()  # get new state

        if agent.her:
            temp_mem.append((state, action, reward, state_new, done, success))
        else:
            agent.remember(state, action, reward, state_new, done)

        agent.episode_length += 1

        if done:
            arm.reset()
            agent.noise_soft.reset()
            agent.noise_strong.reset()
            agent.n_episodes += 1

            if agent.her:
                agent.generate_her_memory(temp_mem, obj_final_pos=obj_pos)

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
                mean_distance_eval = eval_model(arm=arm, agent=agent, evaluation_episodes=50, print_info=False)
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

            agent.train_ddpg()
            plotter.plot(reward_list, mean_reward_list, mean_distance_from_target_list,
                         mean_distance_evaluated_list, evaluated_epochs_list)
            agent.save(eval_distance=mean_distance_eval)
            agent.update_exploration()
            agent.episode_length = 0

            if agent.her:
                target = agent.generate_target_her()
                arm.update_target(target)
                temp_mem = []

            total_time = time.time() - start_time
            avg_episode_time = total_time / agent.n_episodes


if __name__ == "__main__":
    train()
