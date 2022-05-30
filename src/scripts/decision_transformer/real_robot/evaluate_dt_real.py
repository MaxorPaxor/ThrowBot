"""
Run the ros point streaming interface for the motoman gp8:
roslaunch motoman_gp8_support robot_interface_streaming_gp8.launch controller:='yrc1000' robot_ip:=192.168.255.3

Enable the robot:
rosservice call /robot_enable
"""

import torch
import numpy as np
import time
import argparse
from collections import deque

from agent.model_dt.model_dt import DecisionTransformer
from real_robot.robot_env_dt_real import RoboticArm


def eval_model(arm, model, target, print_info=True):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    target = np.array([target, 0.0, 0.0])
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

    # state_0 = arm.get_state()
    while not done:

        t1 = time.time()
        state = arm.get_state()  # get state
        # state = state_0
        state = np.append(state, arm.target[0])  # append target
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
        action = action.detach().cpu().numpy()
        amp = 1.0
        action = action * np.array([amp, amp, amp, 1])
        action = np.clip(action, -1.0, 1.0)
        # if episode_length <= 1 and action[-1] <= arm.gripper_thresh:
        #     action[-1] = 1.0

        done, termination_reason = arm.step(action)  # perform action and get new state

        rewards[-1] = 0.0  # Reward
        target_return = torch.cat([target_return, target_return[0, -1].reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (episode_length+1)], dim=1)
        episode_length += 1
        t2 = time.time()

        print(f"state: {state}")
        print(f"action: {action}")
        print(f"dt: {t2-t1}")
        # print(done)

        if done:
            time.sleep(2)
            reset_arm()

            # print("states:")
            # print(states.to(dtype=torch.float32))
            # print("actions:")
            # print(actions.to(dtype=torch.float32))
            # print("target_return:")
            # print(target_return.to(dtype=torch.float32))
            # print("timesteps:")
            # print(timesteps.to(dtype=torch.long))
            # t2 = time.time()
            # print(f"dt: {t2 - t1}")

            if print_info:
                print(f' {"EVALUATION:":30}\n'
                      f' {"    Episode Length:":40} {episode_length}\n'
                      f' {"    Target X:":40} {round(arm.target[0], 4)}\n')
                print('=======================================================')

    print('\n')


def reset_arm():
    print("Restarting arm...")
    arm_new.step(np.array([0.0, 0.0, 0.0, 1.0]))
    arm_new.reset_arm()
    print("Done.")


if __name__ == "__main__":
    arm_new = RoboticArm()

    # MODE = 'reset'
    MODE = 'throw'

    if MODE == 'reset':
        reset_arm()

    elif MODE == 'throw':
        hidden_size = 1024
        n_head = 1
        model = DecisionTransformer(
            state_dim=len(arm_new .joints) * arm_new.number_states + 1,
            act_dim=len(arm_new .joints),
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

        # checkpoint = torch.load("../weights/dt_trained_best_pid-tuned.pth", map_location=torch.device('cpu'))
        # checkpoint = torch.load("../weights/dt_trained_best_pid-high.pth", map_location=torch.device('cpu'))
        checkpoint = torch.load("../weights/dt_trained_simulation_real_cur-best.pth", map_location=torch.device('cpu'))
        # checkpoint = torch.load("../weights/dt_trained_simulation_real_iter-0_.pth", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device=device)

        eval_model(arm=arm_new, model=model, target=1.0)

