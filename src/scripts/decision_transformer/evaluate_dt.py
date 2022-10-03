import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

from collections import deque

from env.robot_env_dt import RoboticArm
from agent.model_dt.model_dt import DecisionTransformer
from agent.agent_dt import Agent


def eval_model(arm, model, state_mean=None, state_std=None, print_info=True, target=None, bc=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    ### DELETE
    # Agent
    agent = Agent(arm=arm)
    agent.epsilon_arm = 100  # 50
    agent.soft_exploration_rate = 0
    agent.epsilon_arm_decay = 0
    agent.k = 0
    ###

    num_params = count_parameters(model)
    print(f"Number of params: {num_params}")

    reward_list = []
    distance_from_target_list = []
    hit_list = []
    hit_target_list = []
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
        target_list = np.arange(0.5, 2.1, 0.1)
    else:
        target_list = target

    for x in target_list:

        target = np.array([x, 0.0, 0.0])
        arm.update_target(target)

        done = False
        while not done:
            state = arm.get_state()  # get state
            state = np.append(state, arm.target[0])  # append target
            if state_mean is not None and state_std is not None:
                state = (state - state_mean) / state_std

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
            # print(f"dt: {t2 - t1}")

            actions[-1] = action
            action = action.detach().cpu().numpy()

            reward, done, termination_reason, obj_pos, success = arm.step(action)  # perform action and get new state
            rewards[-1] = reward
            target_return = torch.cat([target_return, target_return[0, -1].reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (episode_length+1)], dim=1)
            episode_length += 1

        # DONE
        arm.reset()
        n_episodes += 1

        hit_list.append(1.0 if reward == 1 else 0)
        hit_list_np = np.array(hit_list)
        hit_rate = np.mean(hit_list_np)

        if reward == 1:
            hit_target_list.append(x)

        distance_from_goal = calc_dist_from_goal(obj_pos, arm.target)
        distance_from_target_list.append(distance_from_goal)
        distance_from_target_list_np = np.array(distance_from_target_list)
        average_distance = np.mean(distance_from_target_list_np)
        std_distance = np.std(distance_from_target_list_np)

        if print_info:
            print(f' {"EVALUATION:":30}\n'
                  f' {"    Episode Number:":40} {n_episodes}\n'
                  f' {"    Episode Length:":40} {episode_length}\n'
                  f' {"    Hit-rate:":40} {hit_rate}\n'
                  f' {"    Target X:":40} {round(arm.target[0], 4)}\n'
                  f' {"    Final Object X Position:":40} {round(obj_pos[0], 4)}\n'
                  f' {"    Distance From Target:":40} {round(distance_from_goal, 4)}\n'
                  f' {"    Average Distance From Target:":40} {average_distance}\n'
                  f' {"    STD Distance:":40} {std_distance}\n')
            print('=======================================================')

        else:
            print("\r \rEvaluating Model... {0}%".format(str(int(100 * n_episodes / len(target_list)))), end='')

        episode_length = 0

        states = torch.zeros((0, model.state_dim), device=device, dtype=torch.float32)
        actions = torch.zeros((0, model.act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        ep_return = 1.
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        agent.update_exploration()
        agent.episode_length = 0
        agent.noise_soft.reset()
        agent.noise_strong.reset()

    print(' Done.')
    return average_distance, std_distance, hit_rate, distance_from_target_list, hit_target_list


def calc_dist_from_goal(obj_pos, target):
    """
    Calculates euclidean distance between final object position and the target
    """

    distance = np.sqrt((obj_pos[0] - target[0]) ** 2 +
                       (obj_pos[1] - target[1]) ** 2)

    return distance


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    arm_new = RoboticArm()
    arm_new.state_mem = deque(maxlen=arm_new.number_states)
    arm_new.noise_actions = False
    arm_new.random_delay = 0

    hidden_size = 128  # 1024
    n_head = 1
    model = DecisionTransformer(
        state_dim=len(arm_new .joints) * arm_new.number_states + 1,
        act_dim=len(arm_new .joints),
        max_length=20,  # K=10
        max_ep_len=64,  # max_ep_length=10
        hidden_size=hidden_size,
        n_layer=1,
        n_head=n_head,  # 1
        n_inner=4 * hidden_size,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )

    checkpoint = torch.load("./weights/dt_env-sim_model-big_pid-high.pth", map_location=torch.device('cpu'))
    # checkpoint = torch.load("./weights/dt_best.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)

    target_list = np.arange(0.5, 2.2, 0.2)
    _, _, hit_rate, _, target_list = eval_model(arm=arm_new, model=model, target=target_list)

    # target_list_hits = []
    # successes = 0
    # total_throws = 0
    # while successes < 1000:
    #     target = np.random.rand() * 1.5 + 0.5
    #     _, _, hit_rate, _, target_list = eval_model(arm=arm_new, model=model, target=[target])
    #
    #     if hit_rate == 1:
    #         target_list_hits.append(target)
    #         successes += 1
    #
    #     total_throws += 1
    #
    #     print(hit_rate, target_list, total_throws, successes)

    # df = pd.DataFrame(target_list_hits)
    # df.to_csv("results/data_hist/target_list_hits.csv")

    # results_cols = target_list
    # df = pd.DataFrame(errors, columns=results_cols)
    # df.to_csv(f'results/evaluation_results/eval_random.csv', index=False)
