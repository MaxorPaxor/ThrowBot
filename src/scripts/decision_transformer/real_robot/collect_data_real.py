import torch
import numpy as np
import argparse
import pickle
import time
from collections import deque

from real_robot.robot_env_dt_real import RoboticArm
from agent.model_dt.model_dt import DecisionTransformer
from agent.agent_dt import Agent

EXPLORATION = True


def collect_real_data(arm, model):

    # Reset arm
    reset_arm()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    n_episodes = 0

    agent = Agent(arm=arm, load_nn=False, load_mem=False)
    agent.epsilon_arm = 100  # 50
    agent.soft_exploration_rate = 0
    agent.epsilon_arm_decay = 0
    agent.MAX_MEMORY = 1000  # Number of trajectories
    agent.memory = deque(maxlen=int(agent.MAX_MEMORY))  # collect 100k transitions
    agent.k = 3

    temp_mem = []
    # target_list = np.arange(0.3, 2.3, 0.1)
    target_list = np.arange(2.2, 2.3, 0.1)

    # arm.first_step(np.array([0.0, 0.0, 0.0, 1.0]))
    for x in target_list:
        _ = input("Press ENTER to throw.")

        target = np.array([x, 0.0, 0.0])
        arm.update_target(target)

        states = torch.zeros((0, model.state_dim), device=device, dtype=torch.float32)
        actions = torch.zeros((0, model.act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        ep_return = 1.
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        episode_length = 0

        arm.first_step(np.array([0.0, 0.0, 0.0, 1.0]))
        done = False
        print('---')
        while not done:

            state_ = arm.get_state()  # get state
            state = np.append(state_, arm.target[0])  # append target
            state = torch.from_numpy(state).reshape(1, model.state_dim).to(device=device, dtype=torch.float32)
            states = torch.cat([states, state], dim=0)
            actions = torch.cat([actions, torch.zeros((1, model.act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                states.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )  # get action

            actions[-1] = action
            action_ = action.detach().cpu().numpy()

            if EXPLORATION:
                pass
                # TODO: implement real robot exploration

            done, termination_reason = arm.step(action_)  # perform action and get new state
            # print(action_)

            if done:
                object_position = float(input(f'Target: {x}. Input object position '))
                object_position = np.array([object_position, 0, 0])
                distance_from_target = calc_dist_from_goal(object_position, arm.target)

                if distance_from_target < arm.target_radius:
                    reward_ = 1.0  # Reward
                else:
                    reward_ = -1.0  # Reward

            else:
                reward_ = 0.0  # Reward

            rewards[-1] = reward_
            target_return = torch.cat([target_return, target_return[0, -1].reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (episode_length+1)], dim=1)
            episode_length += 1

            temp_mem.append((state_, action_, reward_, done))

        # DONE
        agent.generate_her_memory(temp_mem, obj_final_pos=object_position)
        temp_mem = []
        reset_arm()
        n_episodes += 1

        pickle.dump(agent.memory, open(f"../data/memory_real_"
                                       f"traj-{len(target_list)}_"
                                       f"Hz-{arm.UPDATE_RATE}_"
                                       f"herK-{agent.k}.pkl", 'wb'))

        print('\n')


def reset_arm():
    print("Restarting arm...")
    time.sleep(4)
    arm_new.step(np.array([0.0, 0.0, 0.0, 1.0]))
    arm_new.reset_arm()
    time.sleep(4)
    print("Done.")


def calc_dist_from_goal(obj_pos, target):
    """
    Calculates euclidean distance between final object position and the target
    """

    distance = np.sqrt((obj_pos[0] - target[0]) ** 2 +
                       (obj_pos[1] - target[1]) ** 2)

    return distance


if __name__ == "__main__":

    arm_new = RoboticArm()
    arm_new.number_states = 1  # 1 state for decision transformer
    arm_new.state_mem = deque(maxlen=arm_new.number_states)

    model = DecisionTransformer(
        state_dim=len(arm_new .joints) * arm_new.number_states + 1,
        act_dim=len(arm_new .joints),
        max_length=10,
        max_ep_len=16,
        hidden_size=128,
        n_layer=3,
        n_head=1,
        n_inner=4 * 128,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )

    # checkpoint = torch.load("./weights/dt_random.pth", map_location=torch.device('cpu'))
    checkpoint = torch.load("../weights/dt_trained_simulation_good_1.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)

    collect_real_data(arm=arm_new, model=model)
