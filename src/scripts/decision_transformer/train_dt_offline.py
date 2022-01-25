import numpy as np
import torch
import pickle
from collections import deque
import pandas as pd

from model_dt import DecisionTransformer
from trainer import Trainer
from evaluate_dt import eval_model
from robot_env_dt import RoboticArm


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    max_ep_len = 10
    state_dim = 5  # 5 or 13
    act_dim = 4
    batch_size = 512
    K = 10

    # load dataset
    # transitions = pickle.load(open('./data/random_1000trans_1s_k3.pkl', 'rb'))
    # # (state, action, reward, next_state, done)
    #
    # fails = 0
    # successes = 0
    # for transition in transitions:
    #     if transition[2] == -1:
    #         fails += 1
    #     if transition[2] == 1:
    #         successes += 1
    # print(f"Total real throws: {(fails + successes)/2} ,Successes: {successes}, Fails: {fails}")
    #
    # trajectories = []
    # trajectory = []
    # states = []
    # for transition in transitions:
    #     trajectory.append(list(transition))
    #     states.append(transition[0][np.array([0, 1, 2, 3, -1])])
    #     if transition[4] is True:
    #         trajectories.append(trajectory)
    #         trajectory = []
    #
    # states = np.array(states)
    #
    # # For normalization
    # states_ = np.concatenate(states, axis=0)
    # state_mean, state_std = np.mean(states_, axis=0), np.std(states_, axis=0) + 1e-6

    # load dataset
    trajectories = pickle.load(open('./data/memory_random_500traj_1Hz_k-3.pkl', 'rb'))
    # (state, action, reward, done)

    fails = 0
    successes = 0
    for trajectory in trajectories:
        for transition in trajectory:
            if transition[2] == -1:
                fails += 1
            if transition[2] == 1:
                successes += 1
    print(f"Total real throws: {(fails + successes)/4} ,Successes: {successes}, Fails: {fails}")

    # import ipdb;
    # ipdb.set_trace(context=20)

    def get_batch(batch_size=256, max_len=K):

        batch_inds = np.random.choice(
            np.arange(len(trajectories)),
            size=batch_size,
            replace=True
        )

        s, a, r, d, rtg, ts, mask = [], [], [], [], [], [], []

        for i in range(batch_size):
            traj = trajectories[batch_inds[i]]

            # import ipdb; ipdb.set_trace(context=20)
            # get sequences from dataset
            s.append(np.array([t[0][np.array([0, 1, 2, 3, -1])] for t in traj]).reshape(1, -1, state_dim))
            a.append(np.array([t[1] for t in traj]).reshape(1, -1, act_dim))
            r.append(np.array([t[2] for t in traj]).reshape(1, -1, 1))
            rtg.append(np.array([traj[-1][2]] * (len(traj))).reshape(1, -1, 1))
            # rtg.append(np.array([traj[-1][2]] * (len(traj)-1) + [0]).reshape(1, -1, 1))
            d.append(np.array([t[3] for t in traj]).reshape(1, -1))
            ts.append(np.array([t for t in range(len(traj))]).reshape(1, -1))

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            # s[-1] = (s[-1] - state_mean) / state_std
            # a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            a[-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)  # / scale
            ts[-1] = np.concatenate([np.zeros((1, max_len - tlen)), ts[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        ts = torch.from_numpy(np.concatenate(ts, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        # print(ts)
        # import ipdb;
        # ipdb.set_trace(context=20)

        return s, a, r, d, rtg, ts, mask

    embed_dim = 128
    n_layer = 3
    n_head = 1
    activation_function = 'relu'
    dropout = 0.1
    n_positions = 1024  # 1024

    df = pd.DataFrame()
    number_experiments = 1
    for experiment in range(number_experiments):

        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * embed_dim,
            activation_function=activation_function,
            n_positions=n_positions,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
        )
        model = model.to(device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # weight_decay=1e-4

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2)
        )

        arm_new = RoboticArm()
        arm_new.number_states = 1  # 1 state for decision transformer
        arm_new.state_mem = deque(maxlen=arm_new.number_states)

        # First evaluation
        evaluation_list = []
        avg_distance_eval = eval_model(arm=arm_new, model=model, evaluation_episodes=5,
                                       print_info=False,
                                       plot=False)
        evaluation_list.append(avg_distance_eval)
        best_avg_distance_eval = avg_distance_eval

        # Train
        max_iters = 75
        for iter in range(max_iters):
            trainer.train_iteration(num_steps=100)
            avg_distance_eval = eval_model(arm=arm_new, model=model, evaluation_episodes=30,
                                           print_info=False,
                                           plot=False)
            evaluation_list.append(avg_distance_eval)

            if avg_distance_eval < best_avg_distance_eval:
                best_avg_distance_eval = avg_distance_eval
                model.save(file_name='dt_trained.pth')

            print(f"Experiment: {experiment+1}/{number_experiments}")
            print(f"Iteration: {iter+1}/{max_iters}")
            print(f"Evaluation Score: {avg_distance_eval}")
            print(f"Best Evaluation Score: {best_avg_distance_eval}")
            print('=' * 40)

        df[experiment] = evaluation_list
        print(df)

    df.to_csv('results/experiment_dt_offline_1k.csv', index=False)


if __name__ == '__main__':
    run()
