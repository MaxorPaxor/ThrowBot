import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager
import os
import sys
import pandas as pd
import glob

matplotlib.use('TkAgg')


class Plot:
    def __init__(self):
        self.fig, self.axs = plt.subplots(3)
        self.fig.set_facecolor('#DEDEDE')
        self.fig.set_figheight(9)
        self.fig.set_figwidth(8)
        self.axs[0].set_facecolor('#DEDEDE')
        self.axs[1].set_facecolor('#DEDEDE')
        self.axs[2].set_facecolor('#DEDEDE')

    def plot(self, evaluation_list, hit_rate_list):
        plt.ion()

        # New Plot
        with stdout_redirected():
            self.axs[0].cla()
            self.axs[1].cla()
            self.axs[2].cla()

        # Title And Labels
        self.fig.suptitle('Training...')
        self.axs[0].set_xlabel('Number of Episodes', labelpad=0)
        self.axs[0].set_ylabel('Score')
        self.axs[1].set_xlabel('Number of Episodes', labelpad=0)
        self.axs[1].set_ylabel('Mean Distance')
        self.axs[2].set_xlabel('Epochs', labelpad=0)
        self.axs[2].set_ylabel('Mean Evaluation Distance')

        # Plot
        self.axs[0].plot(evaluation_list)
        self.axs[1].plot(hit_rate_list)

        # X-Y Lim
        # plt.ylim(ymin=-1.2, ymax=1.2)
        # plt.xlim(xmax=int(1.2*(len(scores)-1)))
        y_bot_0, y_top_0 = self.axs[0].get_ylim()
        y_bot_1, y_top_1 = self.axs[1].get_ylim()

        # Text
        # axs[0].text(len(scores)-1, scores[-1], str(scores[-1]))
        self.axs[0].annotate("Mean Distance: {}".format(evaluation_list[-1]),
                             xy=(0.05, 0.95),
                             xycoords='axes fraction',
                             size=10,
                             ha="left", va="top",
                             bbox=dict(boxstyle="square",
                                       # ec=(1., 0.5, 0.5),
                                       fc=(0.95, 0.95, 0.8), ))

        self.axs[1].annotate("Hit-Rate: {}".format(hit_rate_list[-1]),
                             xy=(0.05, 0.95),
                             xycoords='axes fraction',
                             size=10,
                             ha="left", va="top",
                             bbox=dict(boxstyle="square",
                                       # ec=(1., 0.5, 0.5),
                                       fc=(0.95, 0.95, 0.8), ))

        plt.ioff()
        # plt.draw()
        # plt.pause(.01)
        plt.show()


def plot_results(result_dt_file, result_ddpg_file):
    # Load files
    df_dt = pd.read_csv(result_dt_file)
    df_ddpg = pd.read_csv(result_ddpg_file)

    # Analyse DT Data
    x_dt = range(df_dt.shape[0])
    dy_dt = []
    y_dt = []
    y_max_dt = []
    y_min_dt = []
    y_err_top_dt = []
    y_err_bot_dt = []
    for index, row in df_dt.iterrows():
        max_dt = row.max()
        min_dt = row.min()
        mean_dt = row.mean()
        # err = row.std()
        y_err_top_dt.append(mean_dt - max_dt)
        y_err_bot_dt.append(min_dt - mean_dt)

        # dy.append(err)
        y_dt.append(mean_dt)
        y_max_dt.append(max_dt)
        y_min_dt.append(min_dt)

    dy_dt = np.array([y_err_top_dt, y_err_bot_dt])

    # Analyse DDPG Data
    x_ddpg = range(df_ddpg.shape[0])
    dy_ddpg = []
    y_ddpg = []
    y_max_ddpg = []
    y_min_ddpg = []
    y_err_top_ddpg = []
    y_err_bot_ddpg = []
    for index, row in df_ddpg.iterrows():
        max_ddpg = row.max()
        min_ddpg = row.min()
        mean_ddpg = row.mean()
        # err = row.std()
        y_err_top_ddpg.append(mean_ddpg - max_ddpg)
        y_err_bot_ddpg.append(min_ddpg - mean_ddpg)

        # dy.append(err)
        y_ddpg.append(mean_ddpg)
        y_max_ddpg.append(max_ddpg)
        y_min_ddpg.append(min_ddpg)

    dy_ddpg = np.array([y_err_top_ddpg, y_err_bot_ddpg])

    # Create fig
    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(12)
    ax_dt = fig.add_subplot(1, 1, 1)

    # Major and minor ticks
    major_ticks_x = np.arange(0, 101, 10)
    minor_ticks_x = np.arange(0, 101, 5)
    major_ticks_y = np.arange(0, 101, 1)
    minor_ticks_y = np.arange(0, 101, 0.2)
    ax_dt.set_xticks(major_ticks_x)
    ax_dt.set_xticks(minor_ticks_x, minor=True)
    ax_dt.set_yticks(major_ticks_y)
    ax_dt.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    ax_dt.grid(which='both')
    ax_dt.grid(which='minor', alpha=0.2)
    ax_dt.grid(which='major', alpha=0.5)

    # Titles
    plt.title('DT Offline Training - 10k Random Samples')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Average distance from target')

    # plt.plot(x, y)
    # plt.plot(x, y_max)
    # plt.plot(x, y_min)
    ax_dt.errorbar(x_dt, y_dt, yerr=dy_dt, fmt='tab:blue', elinewidth=1, capsize=3, capthick=1, label='DT')
    ax_dt.errorbar(x_ddpg, y_ddpg, yerr=dy_ddpg, fmt='tab:red', elinewidth=1, capsize=3, capthick=1, label='DDPG')
    ax_dt.legend()
    # plt.savefig('results/results_ddpg.png')
    # plt.savefig('results/results_dt.png')
    plt.savefig('results/results_combined.png')


def plot_attempts_and_k():
    # Load files
    path = '../results/attempts_and_k_exp/'
    # mem_#attempts_#k
    # mem_100_0 = pd.read_csv(path + 'memory_random_attempts-100_herK-0_experiment-0_results.csv')
    # mem_100_2 = pd.read_csv(path + 'memory_random_attempts-100_herK-2_experiment-0_results.csv')
    # mem_100_4 = pd.read_csv(path + 'memory_random_attempts-100_herK-4_experiment-0_results.csv')
    # mem_100_6 = pd.read_csv(path + 'memory_random_attempts-100_herK-6_experiment-0_results.csv')
    # mem_100_8 = pd.read_csv(path + 'memory_random_attempts-100_herK-8_experiment-0_results.csv')
    # mem_500_0 = pd.read_csv(path + 'memory_random_attempts-500_herK-0_experiment-0_results.csv')
    # mem_500_2 = pd.read_csv(path + 'memory_random_attempts-500_herK-2_experiment-0_results.csv')
    # mem_500_4 = pd.read_csv(path + 'memory_random_attempts-500_herK-4_experiment-0_results.csv')
    # mem_500_6 = pd.read_csv(path + 'memory_random_attempts-500_herK-6_experiment-0_results.csv')
    # mem_500_8 = pd.read_csv(path + 'memory_random_attempts-500_herK-8_experiment-0_results.csv')
    # mem_1000_0 = pd.read_csv(path + 'memory_random_attempts-1000_herK-0_experiment-0_results.csv')
    # mem_1000_2 = pd.read_csv(path + 'memory_random_attempts-1000_herK-2_experiment-0_results.csv')
    # mem_1000_4 = pd.read_csv(path + 'memory_random_attempts-1000_herK-4_experiment-0_results.csv')
    # mem_1000_6 = pd.read_csv(path + 'memory_random_attempts-1000_herK-6_experiment-0_results.csv')
    # mem_1000_8 = pd.read_csv(path + 'memory_random_attempts-1000_herK-8_experiment-0_results.csv')

    mem_100_raw = pd.read_csv(path + 'memory_random_attempts-100_her-raw_experiment-0_results.csv')
    mem_100_0 = pd.read_csv(path + 'memory_random_attempts-100_herK-0_experiment-0_results.csv')
    mem_100_1 = pd.read_csv(path + 'memory_random_attempts-100_herK-1_experiment-0_results.csv')
    mem_100_3 = pd.read_csv(path + 'memory_random_attempts-100_herK-3_experiment-0_results.csv')
    mem_100_5 = pd.read_csv(path + 'memory_random_attempts-100_herK-5_experiment-0_results.csv')
    mem_500_raw = pd.read_csv(path + 'memory_random_attempts-500_her-raw_experiment-0_results.csv')
    mem_500_0 = pd.read_csv(path + 'memory_random_attempts-500_herK-0_experiment-0_results.csv')
    mem_500_1 = pd.read_csv(path + 'memory_random_attempts-500_herK-1_experiment-0_results.csv')
    mem_500_3 = pd.read_csv(path + 'memory_random_attempts-500_herK-3_experiment-0_results.csv')
    mem_500_5 = pd.read_csv(path + 'memory_random_attempts-500_herK-5_experiment-0_results.csv')
    mem_1000_raw = pd.read_csv(path + 'memory_random_attempts-1000_her-raw_experiment-0_results.csv')
    mem_1000_0 = pd.read_csv(path + 'memory_random_attempts-1000_herK-0_experiment-0_results.csv')
    mem_1000_1 = pd.read_csv(path + 'memory_random_attempts-1000_herK-1_experiment-0_results.csv')
    mem_1000_3 = pd.read_csv(path + 'memory_random_attempts-1000_herK-3_experiment-0_results.csv')
    mem_1000_5 = pd.read_csv(path + 'memory_random_attempts-1000_herK-5_experiment-0_results.csv')

    # Create fig
    plt.rcParams.update({'font.size': 15})
    plt.rc('figure', autolayout=True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=17)
    plt.rc('legend', fontsize=13)
    plt.rc('mathtext', fontset='stix')
    plt.rc('font', family='STIXGeneral')
    fig = plt.figure(figsize=(7.2, 4.45))
    data_fig = fig.add_subplot(1, 1, 1)

    # Major and minor ticks
    major_ticks_x = np.array([100, 500, 1000])
    # minor_ticks_x = np.arange(0, 101, 5)
    # major_ticks_y = np.arange(0, 101, 1)
    # minor_ticks_y = np.arange(0, 101, 0.2)
    data_fig.set_xticks(major_ticks_x)
    # ax_dt.set_xticks(minor_ticks_x, minor=True)
    # ax_dt.set_yticks(major_ticks_y)
    # ax_dt.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    data_fig.grid(which='both')
    data_fig.grid(which='minor', alpha=0.2)
    data_fig.grid(which='major', alpha=0.2)

    # Titles
    # plt.title('DT Offline Training - Number of Random Samples and HER Augmentations')
    plt.xlabel('Number of Random Samples')
    plt.ylabel('Average distance from target')

    # data_type = 'mean'
    data_type = 'min'

    # Plot data
    if data_type == 'min':
        # plt.scatter(100, mem_100_0['avg_distance_eval'].min(), c='tab:blue', label='K = 0')
        # plt.scatter(500, mem_500_0['avg_distance_eval'].min(), c='tab:blue')
        # plt.scatter(1000, mem_1000_0['avg_distance_eval'].min(), c='tab:blue')
        # plt.scatter(100, mem_100_2['avg_distance_eval'].min(), c='tab:red', marker="^", label='K = 2')
        # plt.scatter(500, mem_500_2['avg_distance_eval'].min(), c='tab:red', marker="^")
        # plt.scatter(1000, mem_1000_2['avg_distance_eval'].min(), c='tab:red', marker="^")
        # plt.scatter(100, mem_100_4['avg_distance_eval'].min(), c='tab:green', marker="s", label='K = 4')
        # plt.scatter(500, mem_500_4['avg_distance_eval'].min(), c='tab:green', marker="s")
        # plt.scatter(1000, mem_1000_4['avg_distance_eval'].min(), c='tab:green', marker="s")
        # plt.scatter(100, mem_100_6['avg_distance_eval'].min(), c='tab:orange', marker="D", label='K = 6')
        # plt.scatter(500, mem_500_6['avg_distance_eval'].min(), c='tab:orange', marker="D")
        # plt.scatter(1000, mem_1000_6['avg_distance_eval'].min(), c='tab:orange', marker="D")
        # plt.scatter(100, mem_100_8['avg_distance_eval'].min(), c='tab:purple', marker="*", label='K = 8')
        # plt.scatter(500, mem_500_8['avg_distance_eval'].min(), c='tab:purple', marker="*")
        # plt.scatter(1000, mem_1000_8['avg_distance_eval'].min(), c='tab:purple', marker="*")

        dx = 10
        plt.scatter(100, mem_100_raw['avg_distance_eval'].min(), c='tab:purple', marker="x", label='Raw data')
        plt.scatter(500, mem_500_raw['avg_distance_eval'].min(), c='tab:purple', marker="x")
        plt.scatter(1000, mem_1000_raw['avg_distance_eval'].min(), c='tab:purple', marker="x")
        plt.scatter(100+0.5*dx, mem_100_0['avg_distance_eval'].min(), c='tab:blue', label='K = 0')
        plt.scatter(500+0.5*dx, mem_500_0['avg_distance_eval'].min(), c='tab:blue')
        plt.scatter(1000+0.5*dx, mem_1000_0['avg_distance_eval'].min(), c='tab:blue')
        plt.scatter(100+1.5*dx, mem_100_1['avg_distance_eval'].min(), c='tab:red', marker="^", label='K = 1')
        plt.scatter(500+1.5*dx, mem_500_1['avg_distance_eval'].min(), c='tab:red', marker="^")
        plt.scatter(1000+1.5*dx, mem_1000_1['avg_distance_eval'].min(), c='tab:red', marker="^")
        plt.scatter(100-0.5*dx, mem_100_3['avg_distance_eval'].min(), c='tab:green', marker="s", label='K = 4')
        plt.scatter(500-0.5*dx, mem_500_3['avg_distance_eval'].min(), c='tab:green', marker="s")
        plt.scatter(1000-0.5*dx, mem_1000_3['avg_distance_eval'].min(), c='tab:green', marker="s")
        plt.scatter(100-1.5*dx, mem_100_5['avg_distance_eval'].min(), c='tab:orange', marker="D", label='K = 6')
        plt.scatter(500-1.5*dx, mem_500_5['avg_distance_eval'].min(), c='tab:orange', marker="D")
        plt.scatter(1000-1.5*dx, mem_1000_5['avg_distance_eval'].min(), c='tab:orange', marker="D")

    if data_type == 'mean':
        # plt.scatter(100, mem_100_0['avg_distance_eval'].mean(), c='tab:blue', label='K = 0')
        # plt.scatter(500, mem_500_0['avg_distance_eval'].mean(), c='tab:blue')
        # plt.scatter(1000, mem_1000_0['avg_distance_eval'].mean(), c='tab:blue')
        # plt.scatter(100, mem_100_2['avg_distance_eval'].mean(), c='tab:red', marker="^", label='K = 2')
        # plt.scatter(500, mem_500_2['avg_distance_eval'].mean(), c='tab:red', marker="^")
        # plt.scatter(1000, mem_1000_2['avg_distance_eval'].mean(), c='tab:red', marker="^")
        # plt.scatter(100, mem_100_4['avg_distance_eval'].mean(), c='tab:green', marker="s", label='K = 4')
        # plt.scatter(500, mem_500_4['avg_distance_eval'].mean(), c='tab:green', marker="s")
        # plt.scatter(1000, mem_1000_4['avg_distance_eval'].mean(), c='tab:green', marker="s")
        # plt.scatter(100, mem_100_6['avg_distance_eval'].mean(), c='tab:orange', marker="D", label='K = 6')
        # plt.scatter(500, mem_500_6['avg_distance_eval'].mean(), c='tab:orange', marker="D")
        # plt.scatter(1000, mem_1000_6['avg_distance_eval'].mean(), c='tab:orange', marker="D")
        # plt.scatter(100, mem_100_8['avg_distance_eval'].mean(), c='tab:purple', marker="*", label='K = 8')
        # plt.scatter(500, mem_500_8['avg_distance_eval'].mean(), c='tab:purple', marker="*")
        # plt.scatter(1000, mem_1000_8['avg_distance_eval'].mean(), c='tab:purple', marker="*")

        plt.scatter(100, mem_100_raw['avg_distance_eval'].mean(), c='tab:purple', label='Raw data')
        plt.scatter(500, mem_500_raw['avg_distance_eval'].mean(), c='tab:purple')
        plt.scatter(1000, mem_1000_raw['avg_distance_eval'].mean(), c='tab:purple')
        plt.scatter(100, mem_100_0['avg_distance_eval'].mean(), c='tab:blue', label='K = 0')
        plt.scatter(500, mem_500_0['avg_distance_eval'].mean(), c='tab:blue')
        plt.scatter(1000, mem_1000_0['avg_distance_eval'].mean(), c='tab:blue')
        plt.scatter(100, mem_100_1['avg_distance_eval'].mean(), c='tab:red', marker="^", label='K = 2')
        plt.scatter(500, mem_500_1['avg_distance_eval'].mean(), c='tab:red', marker="^")
        plt.scatter(1000, mem_1000_1['avg_distance_eval'].mean(), c='tab:red', marker="^")
        plt.scatter(100, mem_100_3['avg_distance_eval'].mean(), c='tab:green', marker="s", label='K = 4')
        plt.scatter(500, mem_500_3['avg_distance_eval'].mean(), c='tab:green', marker="s")
        plt.scatter(1000, mem_1000_3['avg_distance_eval'].mean(), c='tab:green', marker="s")
        plt.scatter(100, mem_100_5['avg_distance_eval'].mean(), c='tab:orange', marker="D", label='K = 6')
        plt.scatter(500, mem_500_5['avg_distance_eval'].mean(), c='tab:orange', marker="D")
        plt.scatter(1000, mem_1000_5['avg_distance_eval'].mean(), c='tab:orange', marker="D")

    data_fig.legend(title="Number of HER Augmentations K")
    plt.savefig('../results/results_attempts_and_k.png')


def plot_evaluation():
    # Load files
    path = '../results/evaluation_results/'
    eval_results = pd.read_csv(path + 'eval.csv')

    # Create fig
    plt.rcParams.update({'font.size': 15})
    plt.rc('figure', autolayout=True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=17)
    plt.rc('legend', fontsize=13)
    plt.rc('mathtext', fontset='stix')
    plt.rc('font', family='STIXGeneral')
    fig = plt.figure(figsize=(7.2, 4.45))
    error_fig = fig.add_subplot(1, 1, 1)

    # Titles
    # plt.title('Evaluation of throws - Simulation')
    plt.xlabel('Target [m]')
    plt.ylabel('Distance from target [m]')

    # Plot data
    y_dt = []
    y_max_dt = []
    y_min_dt = []
    y_err_top_dt = []
    y_err_bot_dt = []
    y_err_std = []
    for column in eval_results:
        column_values = eval_results[column].values
        max_dt = column_values.max()
        min_dt = column_values.min()
        mean_dt = column_values.mean()
        err = column_values.std()
        y_err_top_dt.append(mean_dt - max_dt)
        y_err_bot_dt.append(min_dt - mean_dt)

        y_err_std.append(err)
        y_dt.append(mean_dt)
        y_max_dt.append(max_dt)
        y_min_dt.append(min_dt)

    dy_dt = np.array([y_err_std, y_err_std])

    rel_dt = []
    rel_max_dt = []
    rel_min_dt = []
    rel_err_top_dt = []
    rel_err_bot_dt = []
    rel_err_std = []
    for column in eval_results:
        column_values = eval_results[column].values / float(column)
        max_dt = column_values.max()
        min_dt = column_values.min()
        mean_dt = column_values.mean()
        err = column_values.std()
        rel_err_top_dt.append(mean_dt - max_dt)
        rel_err_bot_dt.append(min_dt - mean_dt)

        rel_err_std.append(err)
        rel_dt.append(mean_dt)
        rel_max_dt.append(max_dt)
        rel_min_dt.append(min_dt)

    # drel_dt = np.array([rel_err_top_dt, rel_err_bot_dt])
    drel_dt = np.array([rel_err_std, rel_err_std])

    x = eval_results.columns.to_numpy().astype(float)
    error_fig.errorbar(x, y_dt, yerr=dy_dt, fmt='tab:red', linewidth=2, elinewidth=1.5, capsize=4,
                       capthick=1.5, label='Absolute Error')
    error_fig.errorbar(x, rel_dt, yerr=drel_dt, fmt='tab:blue', linewidth=2, elinewidth=1.5, capsize=4,
                       capthick=1.5, label='Relative Error')

    error_fig.legend()
    plt.savefig('../results/evaluation_results/eval_sim.png')


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    Silences function's printings
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout. buffering and flags such as CLOEXEC may be different


if __name__ == '__main__':
    # plot_results(result_dt_file='results/experiment_dt_offline_10k.csv',
    #              result_ddpg_file='results/experiment_ddpg_offline_10k.csv')
    plot_attempts_and_k()
    # plot_evaluation()
