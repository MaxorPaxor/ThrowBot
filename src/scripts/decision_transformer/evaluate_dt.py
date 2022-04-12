import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from collections import deque

from env.robot_env_dt import RoboticArm
from agent.model_dt.model_dt import DecisionTransformer


def eval_model(arm, model, print_info=True, plot=False, target=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    reward_list = []
    distance_from_target_list = []
    average_distance = None
    n_episodes = 0

    states = torch.zeros((0, model.state_dim), device=device, dtype=torch.float32)
    actions = torch.zeros((0, model.act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    ep_return = 1.
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_length = 0

    if target is None:
        target_list = np.arange(0.5, 2.5, 0.1)
    else:
        target_list = np.array([target, target, target])

    for x in target_list:

        target = np.array([x, 0.0, 0.0])
        arm.update_target(target)

        done = False
        while not done:

            t1 = time.time()
            state = arm.get_state()  # get state
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
            # print(action)

            t2 = time.time()
            # print(f"dt state to action: {t2 - t1}")
            reward, done, termination_reason, obj_pos, success = arm.step(action)  # perform action and get new state
            rewards[-1] = reward
            target_return = torch.cat([target_return, target_return[0, -1].reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (episode_length+1)], dim=1)
            episode_length += 1

        # DONE
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

        arm.reset()
        n_episodes += 1

        reward_list.append(reward)
        reward_list_np = np.array(reward_list)
        average_reward = np.mean(reward_list_np)
        std_reward = np.std(reward_list_np)

        distance_from_goal = calc_dist_from_goal(obj_pos, arm.target)
        distance_from_target_list.append(distance_from_goal)
        distance_from_target_list_np = np.array(distance_from_target_list)
        average_distance = np.mean(distance_from_target_list_np)
        std_distance = np.std(distance_from_target_list_np)

        if print_info:
            print(f' {"EVALUATION:":30}\n'
                  f' {"    Episode Number:":40} {n_episodes}\n'
                  f' {"    Episode Length:":40} {episode_length}\n'
                  f' {"    Reward:":40} {reward}\n'
                  f' {"    Average Reward:":40} {average_reward}\n'
                  f' {"    STD Reward:":40} {std_reward}\n'
                  f' {"    Target X:":40} {round(arm.target[0], 4)}\n'
                  f' {"    Final Object X Position:":40} {round(obj_pos[0], 4)}\n'
                  f' {"    Distance From Target:":40} {round(distance_from_goal, 4)}\n'
                  f' {"    Average Distance From Target:":40} {average_distance}\n'
                  f' {"    Variance distance:":40} {std_distance}\n')
            print('=======================================================')

        else:
            # print(f" Evaluating Model... {int(100 * n_episodes / len(target_list))} %", end="\r")
            print("\r \rEvaluating Model... {0}%".format(str(int(100 * n_episodes / len(target_list)))), end='')

        episode_length = 0

        states = torch.zeros((0, model.state_dim), device=device, dtype=torch.float32)
        actions = torch.zeros((0, model.act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        ep_return = 1.
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.title('Evaluation')
            plt.xlabel('Target')
            plt.ylabel('Average distance from target')

            ax.scatter(target_list[:-1], distance_from_target_list)
            plt.show()

    print(' Done.')
    return average_distance


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
        max_length=10,  # K=10
        max_ep_len=16,  # max_ep_length=10
        hidden_size=128,
        n_layer=3,
        n_head=1,
        n_inner=4 * 128,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )

    checkpoint = torch.load("./weights/dt_trained.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)

    eval_model(arm=arm_new, model=model, plot=False)#, target=1.4)
