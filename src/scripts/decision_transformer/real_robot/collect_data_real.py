import torch
import numpy as np
import argparse
import pickle
import time
from collections import deque

from real_robot.optitrack_detection import OptiTrack
from real_robot.robot_env_dt_real import RoboticArm
from agent.model_dt.model_dt import DecisionTransformer
from agent.agent_dt import Agent

EXPLORATION = False


def collect_real_data(arm, model):

    # Reset arm
    reset_arm()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    n_episodes = 0

    optitrack = OptiTrack()

    agent = Agent(arm=arm, load_nn=False, load_mem=False)
    agent.epsilon_arm = 100  # 50
    agent.soft_exploration_rate = 0
    agent.epsilon_arm_decay = 0
    agent.MAX_MEMORY = 8000  # Number of trajectories
    agent.memory_k0 = deque(maxlen=int(agent.MAX_MEMORY))  # collect agent.MAX_MEMORY transitions
    agent.k = 0

    temp_mem = []
    # target_list = np.arange(0.3, 2.5, 0.1)
    target_list = np.arange(0.5, 2.1, 0.1)
    # amp_range = np.arange(2.0, 4, 0.5)
    amp_range = [1]

    print('---')
    for x in target_list:
        for amp in amp_range:

            confirm(x)

            # Update target
            optitrack.landing_spot = None
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
                    action_ = action_ * np.array([amp, amp, amp, 1])
                    action_ = np.clip(action_, -1.0, 1.0)
                    if episode_length <= 1 and action_[-1] <= arm.gripper_thresh:
                        action_[-1] = 1.0

                done, termination_reason = arm.step(action_)  # perform action and get new state

                if done:
                    time.sleep(2)
                    # object_position_ = float(input(f'Target: {x}. Input object position '))
                    object_position_ = optitrack.landing_spot
                    object_position = np.array([object_position_, 0, 0])
                    distance_from_target = calc_dist_from_goal(object_position, arm.target)
                    print(f"Landing Spot: {object_position_}, Error: {distance_from_target}")

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
            agent.generate_her_memory(temp_mem, target=target,
                                      obj_final_pos=object_position, k=0)
            temp_mem = []
            reset_arm()
            n_episodes += 1

            pickle.dump(agent.memory_k0, open(f"../data/memory_real_finetune2_"
                                           f"traj-{len(target_list) * len(amp_range)}_"
                                           f"herK-{agent.k}.pkl", 'wb'))

            print('\n')


def reset_arm():
    print("Restarting arm...")
    time.sleep(2)
    arm_new.step(np.array([0.0, 0.0, 0.0, 1.0]))
    arm_new.reset_arm()
    time.sleep(2)
    print("Done.")


def confirm(target, angle=0):
    # Confirmation
    confirm = input(f"Distance: {target}, Angle: {angle}\n"
                    f"confirm (y/n)? ")
    if confirm != 'y':
        exit()


def calc_dist_from_goal(obj_pos, target):
    """
    Calculates euclidean distance between final object position and the target
    """

    distance = np.sqrt((obj_pos[0] - target[0]) ** 2 +
                       (obj_pos[1] - target[1]) ** 2)

    return distance


if __name__ == "__main__":

    arm_new = RoboticArm()

    hidden_size = 1024
    n_head = 1
    model = DecisionTransformer(
        state_dim=len(arm_new.joints) * arm_new.number_states + 1,
        act_dim=len(arm_new.joints),
        max_length=20,
        max_ep_len=64,
        hidden_size=hidden_size,
        n_layer=1,
        n_head=n_head,
        n_inner=4 * hidden_size,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )

    # checkpoint = torch.load("../weights/dt_trained_best_pid-high.pth", map_location=torch.device('cpu'))
    checkpoint = torch.load("../weights/dt_trained_simulation_real_cur-best.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)

    collect_real_data(arm=arm_new, model=model)
