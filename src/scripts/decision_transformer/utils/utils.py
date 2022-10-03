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
    """
    Used for live plotting
    """
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


def plot_attempts_and_k():
    # Load files
    path = '../results/attempts_and_k_exp/new_shared_batch/'

    mem_50_raw = pd.read_csv(path + 'memory_random_attempts-50_her-raw_results.csv')
    mem_50_0 = pd.read_csv(path + 'memory_random_attempts-50_herK-0_results.csv')
    mem_50_1 = pd.read_csv(path + 'memory_random_attempts-50_herK-1_results.csv')
    mem_50_3 = pd.read_csv(path + 'memory_random_attempts-50_herK-3_results.csv')
    mem_50_5 = pd.read_csv(path + 'memory_random_attempts-50_herK-5_results.csv')
    mem_100_raw = pd.read_csv(path + 'memory_random_attempts-100_her-raw_results.csv')
    mem_100_0 = pd.read_csv(path + 'memory_random_attempts-100_herK-0_results.csv')
    mem_100_1 = pd.read_csv(path + 'memory_random_attempts-100_herK-1_results.csv')
    mem_100_3 = pd.read_csv(path + 'memory_random_attempts-100_herK-3_results.csv')
    mem_100_5 = pd.read_csv(path + 'memory_random_attempts-100_herK-5_results.csv')
    mem_250_raw = pd.read_csv(path + 'memory_random_attempts-250_her-raw_results.csv')
    mem_250_0 = pd.read_csv(path + 'memory_random_attempts-250_herK-0_results.csv')
    mem_250_1 = pd.read_csv(path + 'memory_random_attempts-250_herK-1_results.csv')
    mem_250_3 = pd.read_csv(path + 'memory_random_attempts-250_herK-3_results.csv')
    mem_250_5 = pd.read_csv(path + 'memory_random_attempts-250_herK-5_results.csv')
    mem_500_raw = pd.read_csv(path + 'memory_random_attempts-500_her-raw_results.csv')
    mem_500_0 = pd.read_csv(path + 'memory_random_attempts-500_herK-0_results.csv')
    mem_500_1 = pd.read_csv(path + 'memory_random_attempts-500_herK-1_results.csv')
    mem_500_3 = pd.read_csv(path + 'memory_random_attempts-500_herK-3_results.csv')
    mem_500_5 = pd.read_csv(path + 'memory_random_attempts-500_herK-5_results.csv')
    mem_750_raw = pd.read_csv(path + 'memory_random_attempts-750_her-raw_results.csv')
    mem_750_0 = pd.read_csv(path + 'memory_random_attempts-750_herK-0_results.csv')
    mem_750_1 = pd.read_csv(path + 'memory_random_attempts-750_herK-1_results.csv')
    mem_750_3 = pd.read_csv(path + 'memory_random_attempts-750_herK-3_results.csv')
    mem_750_5 = pd.read_csv(path + 'memory_random_attempts-750_herK-5_results.csv')
    mem_1000_raw = pd.read_csv(path + 'memory_random_attempts-1000_her-raw_results.csv')
    mem_1000_0 = pd.read_csv(path + 'memory_random_attempts-1000_herK-0_results.csv')
    mem_1000_1 = pd.read_csv(path + 'memory_random_attempts-1000_herK-1_results.csv')
    mem_1000_3 = pd.read_csv(path + 'memory_random_attempts-1000_herK-3_results.csv')
    mem_1000_5 = pd.read_csv(path + 'memory_random_attempts-1000_herK-5_results.csv')

    # Create fig
    plt.rcParams.update({'font.size': 18})
    plt.rc('figure', autolayout=True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=17)
    plt.rc('legend', fontsize=13)
    plt.rc('mathtext', fontset='stix')
    plt.rc('font', family='STIXGeneral')
    fig = plt.figure(figsize=(7., 3))
    data_fig = fig.add_subplot(1, 1, 1)

    # Major and minor ticks
    major_ticks_x = np.array([100, 250, 500, 750, 1000])
    # minor_ticks_x = np.arange(0, 101, 5)
    major_ticks_y = np.arange(10, 140, 20)
    # minor_ticks_y = np.arange(0, 101, 0.2)
    data_fig.set_xticks(major_ticks_x)
    # ax_dt.set_xticks(minor_ticks_x, minor=True)
    data_fig.set_yticks(major_ticks_y)
    # ax_dt.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    data_fig.grid(which='both')
    data_fig.grid(which='minor', alpha=0.2)
    data_fig.grid(which='major', alpha=0.2)

    # Titles
    # plt.title('DT Offline Training - Number of Random Samples and HER Augmentations')
    plt.xlabel('Number of random samples')
    plt.ylabel('Mean error (cm)')

    # data_type = 'mean'
    data_type = 'min'

    # Plot data
    x = np.array([50, 100, 250, 500, 750, 1000])
    dx = 10
    if data_type == 'min':
        x_raw = x
        y_raw = np.array([mem_50_raw['avg_distance_eval'].min(),
                          mem_100_raw['avg_distance_eval'].min(),
                          mem_250_raw['avg_distance_eval'].min(),
                          mem_500_raw['avg_distance_eval'].min(),
                          mem_750_raw['avg_distance_eval'].min(),
                          mem_1000_raw['avg_distance_eval'].min()]) * 100

        x_k0 = x + 0.5 * dx
        y_k0 = np.array([mem_50_0['avg_distance_eval'].min(),
                         mem_100_0['avg_distance_eval'].min(),
                         mem_250_0['avg_distance_eval'].min(),
                         mem_500_0['avg_distance_eval'].min(),
                         mem_750_0['avg_distance_eval'].min(),
                         mem_1000_0['avg_distance_eval'].min()]) * 100

        x_k1 = x + 1.5 * dx
        y_k1 = np.array([mem_50_1['avg_distance_eval'].min(),
                         mem_100_1['avg_distance_eval'].min(),
                         mem_250_1['avg_distance_eval'].min(),
                         mem_500_1['avg_distance_eval'].min(),
                         mem_750_1['avg_distance_eval'].min(),
                         mem_1000_1['avg_distance_eval'].min()]) * 100

        x_k3 = x - 0.5 * dx
        y_k3 = np.array([mem_50_3['avg_distance_eval'].min(),
                         mem_100_3['avg_distance_eval'].min(),
                         mem_250_3['avg_distance_eval'].min(),
                         mem_500_3['avg_distance_eval'].min(),
                         mem_750_3['avg_distance_eval'].min(),
                         mem_1000_3['avg_distance_eval'].min()]) * 100

        x_k5 = x - 1.5 * dx
        y_k5 = np.array([mem_50_5['avg_distance_eval'].min(),
                         mem_100_5['avg_distance_eval'].min(),
                         mem_250_5['avg_distance_eval'].min(),
                         mem_500_5['avg_distance_eval'].min(),
                         mem_750_5['avg_distance_eval'].min(),
                         mem_1000_5['avg_distance_eval'].min()]) * 100

        plt.plot(x_raw, y_raw, c='tab:purple', marker="o", linestyle='solid', linewidth=1.0, label='Baseline (no HER)')
        plt.plot(x_k0, y_k0, c='tab:blue', marker="^", linestyle='solid', linewidth=1.0, label='$K_{her} = 0$')
        plt.plot(x_k1, y_k1, c='tab:red', marker="s", linestyle='solid', linewidth=1.0, label='$K_{her} = 1$')
        plt.plot(x_k3, y_k3, c='tab:green', marker="D", linestyle='solid', linewidth=1.0, label='$K_{her} = 3$')
        plt.plot(x_k5, y_k5, c='tab:orange', marker="*", linestyle='solid', linewidth=1.0, label='$K_{her} = 5$')

    plt.xlim([25, 1025])

    # data_fig.legend(title="Number of HER Augmentations K")
    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    # order = [0, 1, 2, 3, 4, 5]
    order = [0, 1, 2, 3, 4]

    # add legend to plot
    data_fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc=1)

    plt.savefig('../results/attempts_and_k_exp/results_attempts_and_k.png')


def plot_evaluation():
    # Load files
    path = '../results/evaluation_results/'
    eval_results_sim = pd.read_csv(path + 'eval_sim.csv')
    eval_results_real = pd.read_csv(path + 'evaluation_real/eval_real.csv')
    eval_results_real_pre = pd.read_csv(path + 'evaluation_real/eval_real_pre_not-baysian.csv')

    #
    eval_results_sim *= 100
    eval_results_real *= 100
    eval_results_real_pre *= 100

    # Create fig
    plt.rcParams.update({'font.size': 18})
    plt.rc('figure', autolayout=True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=17)
    plt.rc('legend', fontsize=15)
    plt.rc('mathtext', fontset='stix')
    plt.rc('font', family='STIXGeneral')
    fig = plt.figure(figsize=(7, 4.2))
    error_fig = fig.add_subplot(1, 1, 1)

    # Titles
    # plt.title('Evaluation of throws - Simulation')
    plt.xlabel('Distance to goal $d_g$ (cm)')
    plt.ylabel('Mean error (cm)')

    matplotlib.pyplot.yscale("log")

    # major_ticks_x = np.array([100, 500, 1000])
    # minor_ticks_x = np.arange(0, 101, 5)
    # major_ticks_y = np.array([1e1, 1e2, 1e3, 1e4])
    # minor_ticks_y = np.array([1e1, 1e2, 1e3, 1e4])

    # error_fig.set_xticks(major_ticks_x)
    # error_fig.set_xticks(minor_ticks_x, minor=True)
    # error_fig.set_yticks(major_ticks_y)
    # error_fig.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    error_fig.grid(which='both')
    error_fig.grid(which='minor', alpha=0.2)
    error_fig.grid(which='major', alpha=0.2)

    # Process Data
    # DT Sim Error
    y_dt = []
    y_max_dt = []
    y_min_dt = []
    y_err_top_dt = []
    y_err_bot_dt = []
    y_err_std = []
    for column in eval_results_sim:
        column_values = eval_results_sim[column].values
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

    # Real Error
    # target_list = np.arange(.5, 2.05, .05)
    target_list = np.arange(50, 205, 5)
    eval_results_real = abs(eval_results_real - target_list)
    real_error = []
    std_error = []
    for column in eval_results_real:
        column_values = eval_results_real[column].values
        mean_dt = column_values.mean()
        err = column_values.std()

        real_error.append(mean_dt)
        std_error.append(err)

    real_err_str = np.array([std_error, std_error])

    # Real Error Pre fine-tuned
    # target_list_pre = np.arange(.5, 2.1, .1)
    target_list_pre = np.arange(50, 210, 10)
    eval_results_real_pre = abs(eval_results_real_pre - target_list_pre)
    real_error_pre = []
    std_error_pre = []
    for column in eval_results_real_pre:
        column_values = eval_results_real_pre[column].values
        mean_dt = column_values.mean()
        err = column_values.std()

        real_error_pre.append(mean_dt)
        std_error_pre.append(err)

    real_err_str_pre = np.array([std_error_pre, std_error_pre])

    # Plot Data
    x = eval_results_sim.columns.to_numpy().astype(float) * 100
    x_pre = eval_results_real_pre.columns.to_numpy().astype(float) * 100
    # error_fig.errorbar(x, y_dt, yerr=dy_dt, fmt='tab:blue', linewidth=1.5, elinewidth=1.0, capsize=4,
    #                    capthick=1.0, label='Simulation Error')
    # error_fig.errorbar(x, real_error, yerr=real_err_str, fmt='tab:red', linewidth=1.5, elinewidth=1.0, capsize=4,
    #                    capthick=1.0, label='Real Throws Error')

    from scipy.signal import savgol_filter
    window_size = 13
    y_sim_1 = y_dt + dy_dt[0]
    y_sim_2 = y_dt - dy_dt[1]
    y_sim_1 = savgol_filter(y_sim_1, window_size, 3)  # window size, polynomial order
    y_sim_2 = savgol_filter(y_sim_2, window_size, 3)  # window size, polynomial order
    y_dt = savgol_filter(y_dt, window_size, 3)

    y_real_1 = real_error + real_err_str[0]
    y_real_2 = real_error - real_err_str[1]
    y_real_1 = savgol_filter(y_real_1, window_size, 3)  # window size, polynomial order
    y_real_2 = savgol_filter(y_real_2, window_size, 3)  # window size, polynomial order
    real_error = savgol_filter(real_error, window_size, 3)

    y_real_pre_1 = real_error_pre + real_err_str_pre[0]
    y_real_pre_2 = real_error_pre - real_err_str_pre[1]
    y_real_pre_1 = savgol_filter(y_real_pre_1, window_size, 3)  # window size, polynomial order
    y_real_pre_2 = savgol_filter(y_real_pre_2, window_size, 3)  # window size, polynomial order
    real_error_pre = savgol_filter(real_error_pre, window_size, 3)

    error_fig.plot(x, y_dt, color='tab:blue')
    error_fig.plot(x, real_error, color='tab:orange')
    error_fig.plot(x_pre, real_error_pre, color='tab:green')
    error_fig.fill_between(x, y_sim_1, y_sim_2, color='tab:blue', linewidth=0.0, label='Simulated robot', alpha=0.5)
    error_fig.fill_between(x, y_real_1, y_real_2, color='tab:orange', linewidth=0.0, label='Real robot - fine-tuned',
                           alpha=0.6)
    error_fig.fill_between(x_pre, y_real_pre_1, y_real_pre_2, color='tab:green', linewidth=0.0,
                           label='Real robot - no fine-tuning', alpha=0.5)

    plt.xlim([50, 200])
    # plt.xlim([.5, 2.0])
    # plt.ylim([1e-1, 2e1])
    plt.ylim([1e0, 2e2])

    print(f"sim ME: {np.array(y_dt).mean()}, real ME {np.array(real_error).mean()} real-pre ME: {np.array(real_error_pre).mean()}")

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # order = [0, 1]
    order = [2, 1, 0]
    error_fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')

    # Save
    plt.savefig('../results/evaluation_results/evaluation_real/eval_real_both.png')


def plot_target_hist():
    # Load file
    target_list = pd.read_csv('../results/data_hist/target_list_500.csv')
    target_list = target_list.to_numpy()
    target_list = target_list[:, 1] * 100

    target_list_hits = pd.read_csv('../results/data_hist/target_list_hits.csv')
    target_list_hits = target_list_hits.to_numpy()
    target_list_hits = target_list_hits[:, 1] * 100

    # Create fig
    plt.rcParams.update({'font.size': 18})
    plt.rc('figure', autolayout=True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=17)
    plt.rc('legend', fontsize=13)
    plt.rc('mathtext', fontset='stix')
    plt.rc('font', family='STIXGeneral')
    fig = plt.figure(figsize=(7, 3.))
    data_fig = fig.add_subplot(1, 1, 1)
    fig.tight_layout(pad=0.4, w_pad=0.1, h_pad=1.0)

    # Major and minor ticks
    # major_ticks_x = np.array([100, 500, 1000])
    # minor_ticks_x = np.arange(0, 101, 5)
    # major_ticks_y = np.arange(0, 100, 25)
    # minor_ticks_y = np.arange(0, 101, 0.2)
    # data_fig.set_xticks(major_ticks_x)
    # ax_dt.set_xticks(minor_ticks_x, minor=True)
    # data_fig.set_yticks(major_ticks_y)
    # ax_dt.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    data_fig.grid(which='both')
    data_fig.grid(which='minor', alpha=0.2)
    data_fig.grid(which='major', alpha=0.2)

    # Titles
    # plt.title('DT Offline Training - Number of Random Samples and HER Augmentations')
    plt.xlabel('Distance to goal $d_g$ (cm)')
    plt.ylabel('Number of throws')

    # Plot data
    n_bins = 40
    n_bins_hits = 30
    # plt.imshow(target_list[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent)
    print(target_list_hits.shape)
    # n, bins, _ = plt.hist((target_list, target_list_hits), bins=n_bins, range=(0, 2), color=('black', 'tab:red'),
    #                       rwidth=1.0,
    #                       label=('Throws in the dataset', 'Throws in test time'))
    n, bins, _ = plt.hist(target_list_hits, bins=n_bins, range=(0, 200), color='tab:red',
                          rwidth=0.8,
                          label='Successful test throws')
    n, bins, _ = plt.hist(target_list, bins=n_bins, range=(0, 200), color='black',
                          rwidth=0.8,
                          label='Throws in training dataset')

    plt.xlim([0, 200.])
    # matplotlib.pyplot.yscale("log")
    print(n)
    print(bins)
    print(n[10:-1].sum())
    print(n[20:-1].sum())
    print(n[30:-1].sum())
    # plt.plot(target_list)

    # data_fig.legend(title="Number of HER Augmentations K")
    # get handles and labels
    # handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    # order = [5, 0, 1, 2, 3, 4]

    # add legend to plot
    # data_fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    data_fig.legend()

    plt.savefig('../results/data_hist/sample_hist.png')


def plot_real_data():
    # Load files
    path = '../results/real_data/'
    real_data = pd.read_csv(path + 'real_data.csv', header=None)
    real_data = real_data.to_numpy()

    # Create fig
    plt.rcParams.update({'font.size': 18})
    plt.rc('figure', autolayout=True)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=17)
    plt.rc('legend', fontsize=13)
    plt.rc('mathtext', fontset='stix')
    plt.rc('font', family='STIXGeneral')
    fig = plt.figure(figsize=(7, 3.2))
    data_fig = fig.add_subplot(1, 1, 1)

    # Major and minor ticks
    major_ticks_x = np.arange(0, 60, 10)
    minor_ticks_x = np.arange(0, 55, 5)
    # major_ticks_y = np.arange(0, 0.8, 0.1)
    minor_ticks_y = np.arange(0, 800, 100)
    data_fig.set_xticks(major_ticks_x)
    data_fig.set_xticks(minor_ticks_x, minor=True)
    # data_fig.set_yticks(major_ticks_y)
    data_fig.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    data_fig.grid(which='both')
    # data_fig.grid(which='minor', alpha=0.2)
    # data_fig.grid(which='major', alpha=0.2)

    # Titles
    # plt.title("DT Offline Training - Number of Random Samples and HER Augmentations")
    plt.xlabel('Number of real throw samples')
    plt.ylabel('Mean error (cm)')

    # Plot data
    from scipy.signal import savgol_filter
    real_data[2] = savgol_filter(real_data[2], 3, 1)  # windows size, polynomial order (9,7), (3,1)
    plt.semilogy(real_data[0], real_data[1] * 100, c='black', linestyle='solid', linewidth=2.0,
                 label='Fine-tuned (sim. model prior)')
    plt.semilogy(real_data[0], real_data[2] * 100, c='black', linestyle='dashed', linewidth=2.0,
                 label='New model (no prior)')

    plt.xlim([0, 50])
    plt.ylim([0, 100])

    data_fig.legend(title="Model")
    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # specify order of items in legend
    order = [0, 1]

    # add legend to plot
    data_fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    plt.savefig('../results/real_data/real_data.png')


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
    # plot_attempts_and_k()
    # plot_evaluation()
    # plot_target_hist()
    plot_real_data()
