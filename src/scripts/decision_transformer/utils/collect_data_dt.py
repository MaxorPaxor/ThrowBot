import numpy as np
import pickle
import time
from datetime import timedelta
from collections import deque

from agent.agent_dt import Agent
from env.robot_env_dt import RoboticArm


# LOAD_NN = True
# LOAD_MEMORY = True
LOAD_NN = False
LOAD_MEMORY = False


def collect_data():
    """
    Main Loop
    """

    # Metrics
    reward_list = []
    mean_reward_list = []
    distance_from_target_list = []
    mean_distance_from_target_list = []

    arm = RoboticArm()
    arm.number_states = 1  # 1 state for decision transformer
    arm.state_mem = deque(maxlen=arm.number_states)

    agent = Agent(arm=arm, load_nn=LOAD_NN, load_mem=LOAD_MEMORY)
    agent.epsilon_arm = 100  # 50
    agent.soft_exploration_rate = 0
    agent.epsilon_arm_decay = 0
    agent.MAX_MEMORY = 1000  # Number of trajectories
    agent.memory = deque(maxlen=int(agent.MAX_MEMORY))  # collect 100k transitions
    agent.k = 3

    start_time = time.time()
    total_time = 0
    avg_episode_time = 0

    target = agent.generate_target_her()
    arm.update_target(target)
    temp_mem = []
    max_dis = 0

    while True:

        t1 = time.time()
        state = arm.get_state()  # get state
        
        action = agent.get_action_random()  # get action
        action = action.detach().cpu().numpy()
        time.sleep(0.003)  # simulates network pass

        t2 = time.time()
        # print(f"dt state to action: {t2-t1}")
        # print(f"State: {state}")
        print(f"Action: {action}")
        reward, done, termination_reason, obj_pos, success = arm.step(action)  # perform action and get new state

        temp_mem.append((state, action, reward, done, success))
        agent.episode_length += 1

        if done:
            arm.reset()
            agent.noise_soft.reset()
            agent.noise_strong.reset()
            agent.n_episodes += 1

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

            if obj_pos[0] > max_dis:
                max_dis = obj_pos[0]

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
                    f' {"  Max distance:":30} {round(max_dis, 4)}\n'
                    f' {"EXPLORATION":30}\n'
                    f' {"  Exploration:":30} {agent.exploration_flag}\n'
                    f' {"  Exploration rate:":30} {round(agent.epsilon_arm, 4)}\n'
                    f' {"DATA":30}\n'
                    f' {"  Memory:":30} {len(agent.memory)}/{agent.MAX_MEMORY}\n'
                    f' {"  Total time passed:":30} {str(timedelta(seconds=total_time))}\n'
                    f' {"  AVG time per episode:":30} {str(timedelta(seconds=avg_episode_time))}\n')
            print(info)
            print('=======================================================')

            pickle.dump(agent.memory, open(f"./data/memory_random_{agent.MAX_MEMORY}traj_{arm.UPDATE_RATE}Hz_k-{agent.k}.pkl", 'wb'))
            if len(agent.memory) == agent.MAX_MEMORY:
                break

            agent.update_exploration()
            agent.episode_length = 0

            target = agent.generate_target_her()
            arm.update_target(target)
            temp_mem = []

            total_time = time.time() - start_time
            avg_episode_time = total_time / agent.n_episodes

    print(f"Done collecting {agent.MAX_MEMORY} transitions")


if __name__ == "__main__":
    collect_data()
