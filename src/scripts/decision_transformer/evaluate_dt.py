import torch
import numpy as np
import time
from collections import deque

from robot_env_dt import RoboticArm
from model_dt import DecisionTransformer


def eval_model(arm, model, evaluation_episodes=100, print_info=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    reward_list = []
    distance_from_target_list = []
    average_distance = None
    n_episodes = 0

    x = np.random.rand() * 2 + 0.5
    target = np.array([x, 0.0, 0.0])
    arm.update_target(target)

    states = torch.zeros((0, model.state_dim), device=device, dtype=torch.float32)
    actions = torch.zeros((0, model.act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    ep_return = 1.
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_length = 0
    while n_episodes < evaluation_episodes:
        t1 = time.time()
        state = arm.get_state()  # get state
        # state = arm.get_n_state()  # get state
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

        reward, done, termination_reason, obj_pos, success = arm.step(action)  # perform action and get new state
        rewards[-1] = reward
        target_return = torch.cat([target_return, target_return[0, -1].reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (episode_length + 1)], dim=1)
        episode_length += 1

        if done:
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
                print(f" Evaluating Model... {int(100*n_episodes/evaluation_episodes)} %", end="\r")

            episode_length = 0
            x = np.random.rand() * 2 + 0.5
            target = np.array([x, 0.0, 0.0])
            arm.update_target(target)

            states = torch.zeros((0, model.state_dim), device=device, dtype=torch.float32)
            actions = torch.zeros((0, model.act_dim), device=device, dtype=torch.float32)
            rewards = torch.zeros(0, device=device, dtype=torch.float32)
            ep_return = 1.
            target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

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
        max_length=10,
        max_ep_len=10,
        hidden_size=128,
        n_layer=3,
        n_head=1,
        n_inner=4 * 128,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )

    # checkpoint = torch.load("./weights/dt_random.pth", map_location=torch.device('cpu'))
    checkpoint = torch.load("./weights/dt_trained.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)

    eval_model(arm=arm_new, model=model)
