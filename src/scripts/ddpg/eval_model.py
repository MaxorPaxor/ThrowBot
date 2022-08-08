import torch
import numpy as np

from robot_env_ddpg import RoboticArm
from agent_ddpg import Agent


def eval_model(arm, agent, target=None, print_info=True):
    agent.actor.eval()

    reward_list = []
    distance_from_target_list = []
    average_distance = None
    n_episodes = 0

    if target is None:
        target_list = np.arange(0.5, 2.5, 0.2)
    else:
        target_list = np.array(target)

    for x in target_list:

        target = np.array([x, 0.0, 0.0])
        arm.update_target(target)

        done = False
        while not done:
            state = arm.get_n_state()  # get state
            action = agent.get_action_eval(state)  # get action
            reward, done, termination_reason, obj_pos, success = arm.step(action)  # perform action and get new state
            agent.episode_length += 1

        # DONE
        arm.reset()
        n_episodes += 1

        reward_list.append(reward)
        reward_list_np = np.array(reward_list)
        average_reward = np.mean(reward_list_np)
        std_reward = np.std(reward_list_np)

        distance_from_goal = agent.calc_dist_from_goal(obj_pos, arm.target)
        distance_from_target_list.append(distance_from_goal)
        distance_from_target_list_np = np.array(distance_from_target_list)
        average_distance = np.mean(distance_from_target_list_np)
        std_distance = np.std(distance_from_target_list_np)

        if print_info:
            print(f' {"EVALUATION:":30}\n'
                  f' {"    Episode Number:":40} {n_episodes}\n'
                  f' {"    Episode Length:":40} {agent.episode_length}\n'
                  f' {"    Reward:":40} {reward}\n'
                  f' {"    Average Reward:":40} {average_reward}\n'
                  f' {"    STD Reward:":40} {std_reward}\n'
                  f' {"    Target X:":40} {round(arm.target[0], 4)}\n'
                  f' {"    Final Object X Position:":40} {round(obj_pos[0], 4)}\n'
                  f' {"    Distance From Target:":40} {round(distance_from_goal, 4)}\n'
                  f' {"    Average Distance From Target:":40} {average_distance}\n'
                  f' {"    Variance Distance:":40} {std_distance}\n')
            print('=======================================================')

        else:
            print("\r \rEvaluating Model... {0}%".format(str(int(100 * n_episodes / len(target_list)))), end='')

        agent.episode_length = 0

    return average_distance


if __name__ == "__main__":
    arm_new = RoboticArm()
    agent_new = Agent(arm=arm_new)

    checkpoint = torch.load("./weights/actor_best.pth", map_location=torch.device('cpu'))
    try:
        agent_new.actor.load_state_dict(checkpoint['state_dict'])
    except KeyError:
        agent_new.actor.load_state_dict(checkpoint)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent_new.actor.to(device)

    eval_model(arm=arm_new, agent=agent_new)
