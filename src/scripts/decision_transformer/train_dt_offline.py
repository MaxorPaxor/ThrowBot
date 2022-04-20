import numpy as np
import torch
import pickle
import pandas as pd

from collections import deque

from agent.model_dt.model_dt import DecisionTransformer
from agent.trainer import Trainer
from evaluate_dt import eval_model
from env.robot_env_dt import RoboticArm

FINETUNE = True


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    max_ep_len = 16
    state_dim = 5  # 5 or 13
    act_dim = 4
    batch_size = 256
    K = 16

    timestep_noise = True

    # load dataset
    if FINETUNE:
        session_name = 'memory_real_traj-1000_Hz-20_herK-3_0.5-2.8.pkl'
        trajectories = pickle.load(open(f'./data/{session_name}', 'rb'))
    else:
        session_name = 'memory_random_traj-5000_Hz-20_herK-3_noise-True_noise-max-0.3_noise-prob-0.5.pkl'
        trajectories = pickle.load(open(f'./data/{session_name}', 'rb'))
    # (state, action, reward, done)

    fails = 0
    successes = 0
    for trajectory in trajectories:
        print(len(trajectory))
        for transition in trajectory:
            if transition[2] == -1:
                fails += 1
            if transition[2] == 1:
                successes += 1
    print(f"Total real throws: {(fails + successes)/4} ,Successes: {successes}, Fails: {fails}")

    def get_batch(batch_size=256, max_len=K):

        batch_inds = np.random.choice(
            np.arange(len(trajectories)),
            size=batch_size,
            replace=True
        )

        s, a, r, d, rtg, ts, mask = [], [], [], [], [], [], []

        for i in range(batch_size):
            traj = trajectories[batch_inds[i]]

            # get sequences from dataset
            s.append(np.array([t[0][np.array([0, 1, 2, 3, -1])] for t in traj]).reshape(1, -1, state_dim))
            a.append(np.array([t[1] for t in traj]).reshape(1, -1, act_dim))
            r.append(np.array([t[2] for t in traj]).reshape(1, -1, 1))
            rtg.append(np.array([traj[-1][2]] * (len(traj))).reshape(1, -1, 1))
            # rtg.append(np.array([traj[-1][2]] * (len(traj)-1) + [0]).reshape(1, -1, 1))
            d.append(np.array([t[3] for t in traj]).reshape(1, -1))

            if timestep_noise:
                timestep_noise_int = np.random.randint(3)
                ts.append(np.array([t + timestep_noise_int for t in range(len(traj))]).reshape(1, -1))
            else:
                ts.append(np.array([t for t in range(len(traj))]).reshape(1, -1))

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # print(s[-1])
            # print(f"tlen: {tlen}, max_len: {max_len}")
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

        return s, a, r, d, rtg, ts, mask

    embed_dim = 128
    n_layer = 3
    n_head = 1
    activation_function = 'relu'
    dropout = 0.1
    n_positions = 1024  # 1024

    number_experiments = 1
    for experiment in range(number_experiments):
        results_cols = ['avg_distance_eval', 'hit_rate', 'loss']
        df = pd.DataFrame(columns=results_cols)

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
        if FINETUNE:
            checkpoint = torch.load("./weights/dt_trained_simulation.pth", map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

        model = model.to(device=device)
        model.save(file_name='dt_trained.pth')

        model_eval = DecisionTransformer(
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
            resid_pdrop=0,
            attn_pdrop=0,
        )
        model_eval = model_eval.to(device=device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0*1e-2)  # weight_decay=1e-4
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            device=device
        )

        if not FINETUNE:
            arm_new = RoboticArm()
            arm_new.number_states = 1  # 1 state for decision transformer
            arm_new.state_mem = deque(maxlen=arm_new.number_states)

        # First evaluation
        evaluation_list = []
        hit_rate_list = []
        avg_distance_eval = None
        best_avg_distance_eval = avg_distance_eval

        # Train
        max_iters = 5
        num_of_steps = 50
        loss_list = []

        print('=' * 40)
        for iter in range(max_iters):
            train_loss = trainer.train_iteration(num_steps=num_of_steps)

            if FINETUNE:
                model.save(file_name='dt_trained_simulation_real.pth')
            else:
                model.save(file_name='dt_trained.pth')

                # Evaluate
                checkpoint = torch.load("./weights/dt_trained.pth", map_location=torch.device('cpu'))
                model_eval.load_state_dict(checkpoint['state_dict'])
                avg_distance_eval, hit_rate = eval_model(arm=arm_new, model=model_eval, print_info=False, plot=False)

                hit_rate_list.append(hit_rate)
                evaluation_list.append(avg_distance_eval)
                evaluation_list_np = np.array(evaluation_list)
                evaluation_var = evaluation_list_np.var()

                if best_avg_distance_eval is None or avg_distance_eval < best_avg_distance_eval:
                    best_avg_distance_eval = avg_distance_eval
                    model.save(file_name='dt_trained_best.pth')

            loss_list.append(train_loss)

            print(f"Experiment: {experiment+1}/{number_experiments}")
            print(f"Iteration: {iter+1}/{max_iters}")
            print(f"Loss: {train_loss}")
            if not FINETUNE:
                print(f"Evaluation AVG hit-rate: {hit_rate}")
                print(f"Evaluation AVG distance: {avg_distance_eval}")
                print(f"Var Evaluation: {evaluation_var}")
                print(f"Best Evaluation Score: {best_avg_distance_eval}")

            print('=' * 40)

        df['avg_distance_eval'] = evaluation_list
        df['hit_rate'] = hit_rate_list
        df['loss'] = loss_list
        print(df)

        if FINETUNE:
            df.to_csv(f'results/finetune_results_iters-{max_iters}.csv', index=False)
        else:
            df.to_csv(f'results/{session_name}_experiment-{experiment}_results.csv', index=False)


if __name__ == '__main__':
    run()
