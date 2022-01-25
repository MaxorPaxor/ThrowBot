import numpy as np
import time
from datetime import timedelta
import pandas as pd

from agent import Agent
from robot_env import RoboticArm
from eval_model import eval_model


# LOAD_NN = True
LOAD_MEMORY = True
LOAD_NN = False
# LOAD_MEMORY = False


def train():
    """
    Main Loop
    """

    df = pd.DataFrame()
    number_experiments = 10
    for experiment in range(number_experiments):
        arm = RoboticArm()
        agent = Agent(arm=arm, load_nn=LOAD_NN, load_mem=LOAD_MEMORY)
        agent.num_mini_batches_per_training = 100
        agent.train_every_n_episode = 1
        agent.BATCH_SIZE = 512

        # Evaluation
        mean_distance_evaluated_list = []
        mean_distance_eval = eval_model(arm=arm, agent=agent, evaluation_episodes=20, print_info=False)
        mean_distance_evaluated_list.append(mean_distance_eval)

        max_iters = 75
        for iter in range(max_iters):
            agent.train_ddpg()
            mean_distance_eval = eval_model(arm=arm, agent=agent, evaluation_episodes=20, print_info=False)
            mean_distance_evaluated_list.append(mean_distance_eval)

            info = (f' {"EVALUATION":30}\n'
                    f' {"  Mean evaluation distance:":30} {mean_distance_eval}\n'
                    f' {"  Best evaluation distance:":30} {agent.best_evaluation_distance}\n'
                    f' {"TRAINING":30}\n'
                    f' {"  Experiment:":30} {experiment+1}/{number_experiments}\n'
                    f' {"  Iteration:":30} {iter+1}/{max_iters}\n'
                    f' {"  Epoch number:":30} {iter}\n'
                    f' {"  Mini-batches per training:":30} {agent.num_mini_batches_per_training}\n'
                    f' {"  Train every n episodes:":30} {agent.train_every_n_episode}\n'
                    f' {"  Memory:":30} {len(agent.memory)}/{agent.MAX_MEMORY}\n'
                    f' {"  Batch Size:":30} {agent.BATCH_SIZE}\n')
            print(info)
            print('=======================================================')
            agent.save(eval_distance=mean_distance_eval)

        df[experiment] = mean_distance_evaluated_list
        print(df)

    df.to_csv('results/experiment_ddpg_offline_10k.csv', index=False)


if __name__ == "__main__":
    train()
