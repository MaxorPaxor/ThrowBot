import torch
import numpy as np
import time
import argparse
from collections import deque

from agent.model_dt.model_dt import DecisionTransformer
from real_robot.robot_env_dt_real import RoboticArm

from robotiqGripper import RobotiqGripper


def eval_model(arm, model, print_info=True, gripper_bool=False):

    if gripper_bool:
        # Connect to gripper
        gripper_status = None
        gripper = RobotiqGripper("/dev/ttyUSB0", slaveaddress=9)
        gripper._aCoef = -4.7252
        gripper._bCoef = 1086.8131
        gripper.closemm = 0
        gripper.openmm = 860
        gripper.goTomm(270, 255, 255)
        print("Gripper is ready")

    # reset_arm()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    # x = np.random.rand() * 2 + 0.5
    x = 0.6
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
    warmup = False
    done = False
    print('---')
    while not done:

        state = arm.get_state()  # get state
        state = np.append(state, arm.target[0])  # append target
        state = torch.from_numpy(state).reshape(1, model.state_dim).to(device=device, dtype=torch.float32)
        states = torch.cat([states, state], dim=0)
        actions = torch.cat([actions, torch.zeros((1, model.act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        t1 = time.time()
        action = model.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )  # get action
        t2 = time.time()

        actions[-1] = action
        action = action.detach().cpu().numpy()
        done, termination_reason = arm.step(action)  # perform action and get new state

        # print(f"dt: {t2 - t1}")

        rewards[-1] = 0.0  # Reward
        target_return = torch.cat([target_return, target_return[0, -1].reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (episode_length+1)], dim=1)
        episode_length += 1

        if warmup:
            # time.sleep(0.01)
            warmup = False
        else:
            # time.sleep(0.01)
            pass

        if action[-1] < arm.gripper_thresh:
            if gripper_bool:
                # time.sleep(0.01)
                gripper.goTomm(350, 255, 255)
                done = True

        if done:
            time.sleep(2)
            reset_arm()

            # print("states:")
            # print(states.to(dtype=torch.float32))
            print("actions:")
            print(actions.to(dtype=torch.float32))
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
    # time.sleep(2)
    arm_new.step(np.array([0.0, 0.0, 0.0, 1.0]))
    arm_new.reset_arm()
    if GRIPPER:
        gripper = RobotiqGripper("/dev/ttyUSB0", slaveaddress=9)
        gripper._aCoef = -4.7252
        gripper._bCoef = 1086.8131
        gripper.closemm = 0
        gripper.openmm = 860
        gripper.goTomm(270, 255, 255)
    # time.sleep(2)
    print("Done.")


if __name__ == "__main__":
    GRIPPER = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=False)
    args = parser.parse_args()
    print(f"Mode: {args.mode}")

    arm_new = RoboticArm()
    arm_new.number_states = 1  # 1 state for decision transformer
    arm_new.state_mem = deque(maxlen=arm_new.number_states)

    if args.mode == 'reset':
        reset_arm()

    elif args.mode == 'throw':
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
        checkpoint = torch.load("../weights/dt_trained_simulation.pth", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device=device)

        eval_model(arm=arm_new, model=model, gripper_bool=GRIPPER)
