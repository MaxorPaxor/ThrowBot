import numpy as np
import time
from datetime import timedelta

from agent import Agent
from robot_env import RoboticArm
from utils import Plot
from eval_model import eval_model


# LOAD_NN = True
LOAD_MEMORY = True
LOAD_NN = False
# LOAD_MEMORY = False


def train():
    """
    Main Loop
    """
    # Evaluation
    mean_distance_evaluated_list = []
    mean_distance_eval = None
    evaluated_epochs_list = []

    arm = RoboticArm()
    agent = Agent(arm=arm, load_nn=LOAD_NN, load_mem=LOAD_MEMORY)
    agent.num_mini_batches_per_training = 150
    agent.train_every_n_episode = 1
    agent.BATCH_SIZE = 512

    while True:
        agent.train_ddpg()
        mean_distance_eval = eval_model(arm=arm, agent=agent, evaluation_episodes=10, print_info=False)
        mean_distance_evaluated_list.append(mean_distance_eval)
        evaluated_epochs_list.append(agent.num_epoch)
        agent.last_evaluated_epoch = agent.num_epoch

        info = (f' {"EVALUATION":30}\n'
                f' {"  Mean evaluation distance:":30} {mean_distance_eval}\n'
                f' {"  Best evaluation distance:":30} {agent.best_evaluation_distance}\n'
                f' {"  Last evaluated epoch:":30} {agent.last_evaluated_epoch}\n'
                f' {"TRAINING":30}\n'
                f' {"  Epoch number:":30} {agent.num_epoch}\n'
                f' {"  Mini-batches per training:":30} {agent.num_mini_batches_per_training}\n'
                f' {"  Train every n episodes:":30} {agent.train_every_n_episode}\n'
                f' {"  Memory:":30} {len(agent.memory)}/{agent.MAX_MEMORY}\n'
                f' {"  Batch Size:":30} {agent.BATCH_SIZE}\n')
        print(info)
        print('=======================================================')
        agent.save(eval_distance=mean_distance_eval)
        # agent.update_exploration()
        # agent.episode_length = 0


if __name__ == "__main__":
    train()
