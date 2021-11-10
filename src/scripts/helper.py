import matplotlib.pyplot as plt
from IPython import display
from contextlib import contextmanager
import os
import sys

fig, axs = plt.subplots(3)
fig.set_facecolor('#DEDEDE')
fig.set_figheight(9)
fig.set_figwidth(8)
axs[0].set_facecolor('#DEDEDE')
axs[1].set_facecolor('#DEDEDE')
axs[2].set_facecolor('#DEDEDE')


def plot(scores, mean_scores, mean_distance_from_target_list, mean_distance_evaluated_list, evaluated_epochs_list):

    # New Plot
    with stdout_redirected():
        axs[0].cla()
        axs[1].cla()
        axs[2].cla()

    # Title And Labels
    fig.suptitle('Training...')
    axs[0].set_xlabel('Number of Episodes')
    axs[0].set_ylabel('Score')
    axs[1].set_xlabel('Number of Episodes')
    axs[1].set_ylabel('Mean Distance')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Mean Evaluation Distance')

    # Plot
    axs[0].plot(scores)
    axs[0].plot(mean_scores)
    axs[1].plot(mean_distance_from_target_list)
    axs[2].plot(evaluated_epochs_list, mean_distance_evaluated_list)

    # X-Y Lim
    # plt.ylim(ymin=-1.2, ymax=1.2)
    # plt.xlim(xmax=int(1.2*(len(scores)-1)))
    y_bot_0, y_top_0 = axs[0].get_ylim()
    y_bot_1, y_top_1 = axs[1].get_ylim()

    # Text
    # axs[0].text(len(scores)-1, scores[-1], str(scores[-1]))
    axs[0].annotate("Mean score: {}".format(mean_scores[-1]),
                    xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    size=10,
                    ha="left", va="top",
                    bbox=dict(boxstyle="square",
                    # ec=(1., 0.5, 0.5),
                    fc=(0.95, 0.95, 0.8),))

    axs[1].annotate("Mean Distance: {}".format(mean_distance_from_target_list[-1]),
                    xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    size=10,
                    ha="left", va="top",
                    bbox=dict(boxstyle="square",
                    # ec=(1., 0.5, 0.5),
                    fc=(0.95, 0.95, 0.8),))

    plt.show(block=False)
    plt.pause(.01)


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
