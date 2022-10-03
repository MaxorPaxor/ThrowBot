"""
Run the ros point streaming interface for the motoman gp8:
roslaunch motoman_gp8_support robot_interface_streaming_gp8.launch controller:='yrc1000' robot_ip:=192.168.255.3

Enable the robot:
rosservice call /robot_enable

Enable gripper:
sudo chmod +777 /dev/ttyUSB0

OptiTrack cameras:
rosrun natnet_ros client
"""

import torch
import numpy as np
import time
import pandas as pd
import os
import argparse
from collections import deque
from os.path import exists

from agent.model_dt.model_dt import DecisionTransformer
from real_robot.robot_env_dt_real import RoboticArm
from real_robot.optitrack_detection import OptiTrack


def eval_model(arm, model, trg=None, print_info=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.eval()
    optitrack = OptiTrack()

    if trg is None:
        trg, angle = optitrack.calculate_target_position_relative()
        trg_calib = calib_distance(trg)
        target = np.array([trg_calib, 0.0, 0.0])
        arm.first_step(np.array([0.0, 0.0, 0.0, 1.0]))
        arm.rotate(angle=angle)

    else:
        angle = 0
        trg_calib = calib_distance(trg)
        # trg_calib = trg
        target = np.array([trg_calib, 0.0, 0.0])

    # Confirm
    confirm(target[0], angle)

    # Update target
    optitrack.landing_spot = None
    arm.update_target(target)

    # Initiate DT Tensors
    states = torch.zeros((0, model.state_dim), device=device, dtype=torch.float32)
    actions = torch.zeros((0, model.act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    ep_return = 1.
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    episode_length = 0

    done = False
    arm.first_step(np.array([0.0, 0.0, 0.0, 1.0]))
    print('---')
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
        amp = 1
        action = action * np.array([amp, amp, amp, 1])
        action = np.clip(action, -1.0, 1.0)

        done, termination_reason = arm.step(action)  # perform action and get new state

        rewards[-1] = 0.0  # Reward
        target_return = torch.cat([target_return, target_return[0, -1].reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (episode_length + 1)], dim=1)
        episode_length += 1
        t2 = time.time()

        # print(f"state: {state}")
        # print(f"action: {action}")
        # print(f"dt: {t2-t1}")
        # print(done)

        if done:
            time.sleep(2)
            reset_arm()
            landing_spot = optitrack.landing_spot
            # for p in optitrack.object_trajectory:
            #     print(p)

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
                      f' {"    Target X:":40} {round(trg, 4)}\n'
                      f' {"    Landing Spot:":40} {landing_spot}')
                print('=======================================================')

    print('\n')
    return landing_spot


def reset_arm():
    print("Restarting arm...")
    arm_new.step(np.array([0.0, 0.0, 0.0, 1.0]))
    arm_new.reset_arm()
    print("Done.")


def confirm(target, angle):
    # Confirmation
    confirm = input(f"Distance: {target}, Angle: {angle}\n"
                    f"confirm (Y/n)? ")
    # if confirm != 'y':
    #     exit()


def calib_distance(distance):
    # Calculate bias
    # x = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2])
    # y = np.array([-0.007577330237326, 0.022894900668332, 0.003860787926171, 0.022042582862457, 0.036951599075063, 0.044572539389809, 0.044888753935297, 0.037753688183563, 0.025685550083397, 0.007367352271927, -0.007775384595554, -0.013623647595499, -0.026159257288058, -0.037559778473997, -0.047748293642911, -0.039395907517661, -0.030966311503301, -0.022664224197008, 0.023809207275412, 0.048979892210608, 0.059140761572595, 0.070079195505347, 0.070550300546806, 0.047801092219306, 0.058861365635787, 0.082658151674321, -0.001135333927866, -0.012561556352557, -0.081727992542677, -0.10445427299573, -0.117063422855222])
    # z = np.polyfit(x, y, deg=6)
    # p = np.poly1d(z)

    if 0.5 <= distance <= 2.0:
        z = np.array([4.18935377, - 30.70556761, 89.15824107, - 130.50426463, 101.03042058, - 39.10580455, 5.93147809])
        p = np.poly1d(z)
        bias = p(distance)
        new_distance = distance - bias
        new_distance = np.clip(new_distance, a_min=0, a_max=2)

        print(f"Calibrating. distance: {distance},\n"
              f"bias: {bias},\n"
              f"new distance: {new_distance}")

        return new_distance

    else:
        return distance


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

        # checkpoint = torch.load("../weights/dt_env-sim_model-big_pid-high.pth", map_location=torch.device('cpu'))
        # checkpoint = torch.load("../weights/dt_env-sim_model-big_pid-low.pth", map_location=torch.device('cpu'))
        checkpoint = torch.load("../weights/dt_trained_simulation_real_cur-best.pth", map_location=torch.device('cpu'))
        # checkpoint = torch.load("../weights/dt_simulation_real_data-3_iter-2.pth", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device=device)

        # target_list = np.arange(0.5, 2.1, 0.1)  # (0.5, 2.05, 0.05)
        # results_cols = [f"{x:.2f}" for x in target_list]
        # data = {}
        # for r in results_cols:
        #     data[r] = []
        #
        # for trg in target_list:
        #     landing = eval_model(arm=arm_new, model=model, trg=trg)
        #     data[f"{trg:.2f}"].append(landing)
        #
        # df = pd.DataFrame(data, columns=results_cols)
        # print(abs(df - target_list).mean(axis=1))

        # csv_path = f'../results/evaluation_results/evaluation_real/eval_real_100.csv'
        # if not exists(csv_path):
        #     df_new = pd.DataFrame(columns=results_cols)
        #     df_new.to_csv(f'../results/evaluation_results/evaluation_real/eval_real_100.csv',
        #                   index=False)
        #
        # df.to_csv(f'../results/evaluation_results/evaluation_real/eval_real_100.csv',
        #           mode='a', index=False, header=False)

        landing = eval_model(arm=arm_new, model=model)
