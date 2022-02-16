import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager
import os
import sys
import pandas as pd


class Plot:
    def __init__(self):
        self.fig, self.axs = plt.subplots(3)
        self.fig.set_facecolor('#DEDEDE')
        self.fig.set_figheight(9)
        self.fig.set_figwidth(8)
        self.axs[0].set_facecolor('#DEDEDE')
        self.axs[1].set_facecolor('#DEDEDE')
        self.axs[2].set_facecolor('#DEDEDE')

    def plot(self, scores, mean_scores, mean_distance_from_target_list, mean_distance_evaluated_list,
             evaluated_epochs_list):

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
        self.axs[0].plot(scores)
        self.axs[0].plot(mean_scores)
        self.axs[1].plot(mean_distance_from_target_list)
        self.axs[2].plot(evaluated_epochs_list, mean_distance_evaluated_list)

        # X-Y Lim
        # plt.ylim(ymin=-1.2, ymax=1.2)
        # plt.xlim(xmax=int(1.2*(len(scores)-1)))
        y_bot_0, y_top_0 = self.axs[0].get_ylim()
        y_bot_1, y_top_1 = self.axs[1].get_ylim()

        # Text
        # axs[0].text(len(scores)-1, scores[-1], str(scores[-1]))
        self.axs[0].annotate("Mean score: {}".format(mean_scores[-1]),
                             xy=(0.05, 0.95),
                             xycoords='axes fraction',
                             size=10,
                             ha="left", va="top",
                             bbox=dict(boxstyle="square",
                             # ec=(1., 0.5, 0.5),
                             fc=(0.95, 0.95, 0.8),))

        self.axs[1].annotate("Mean Distance: {}".format(mean_distance_from_target_list[-1]),
                             xy=(0.05, 0.95),
                             xycoords='axes fraction',
                             size=10,
                             ha="left", va="top",
                             bbox=dict(boxstyle="square",
                             # ec=(1., 0.5, 0.5),
                             fc=(0.95, 0.95, 0.8),))

        plt.show(block=False)
        plt.pause(.01)


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
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout. buffering and flags such as CLOEXEC may be different


if __name__ == '__main__':
    plot_results(result_dt_file='results/experiment_dt_offline_10k.csv',
                 result_ddpg_file='results/experiment_ddpg_offline_10k.csv')
    # plot_results(result_dt_file='results/experiment_ddpg_offline_10k.csv')


