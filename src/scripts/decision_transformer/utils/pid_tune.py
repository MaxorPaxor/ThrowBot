"""
Main script to manually iterate over algo configs and run evaluation
Save results to right directory.
"""

from os import getcwd, path, chdir, mkdir, makedirs, curdir
from os.path import exists, join, dirname
import time
import yaml
import numpy as np
import rospy
import dynamic_reconfigure.client

from hyperopt import fmin, hp, tpe, space_eval

from env.robot_env_dt import RoboticArm

traj_2_l_real = np.array([0.49135804, 0.49135804, 0.49135804, 0.49135804, 0.49135804, 0.49135804, 0.49135804,
                          0.49135804, 0.49135804, 0.49135804, 0.48938784, 0.39135209, 0.2164495, 0.16115429, 0.15337412,
                          0.15488413, 0.15545937, 0.15551689, 0.15545937, 0.15538748, 0.16073723, 0.22904731,
                          0.39841318, 0.48717314, 0.49240786, 0.49173194, 0.49137244, 0.49131489, 0.49131489])

traj_3_u_real = np.array([-0.28836921, -0.28836921, -0.28836921, -0.28836921, -0.28836921, -0.28836921, -0.28836921,
                          -0.28836921, -0.28836921, -0.28836921, -0.28463015, -0.15138473, 0.07690997, 0.15560319,
                          0.16923644, 0.16624518, 0.16532478, 0.16528644, 0.16530561, 0.16534396, 0.15658109,
                          0.06279734, -0.15874784, -0.28144714, -0.29143718, -0.28904033, -0.2883884, -0.28836921,
                          -0.28836921])

traj_5_b_real = np.array([-1.48707461, -1.48707461, -1.48707461, -1.48707461, -1.48707461, -1.48707461, -1.48707461,
                          -1.48707461, -1.48707461, -1.48707461, -1.48458183, -1.34539711, -1.08936608, -1.01273894,
                          -1.00667489, -1.00689065, -1.00705838, -1.00710642, -1.00710642, -1.00710642, -1.01163638,
                          -1.10453808, -1.35884333, -1.4825685, -1.48757792, -1.4872663, -1.48709857, -1.48707461,
                          -1.48705065])

# Extend last position to reduce oscillation errors
traj_2_l_real = np.append(traj_2_l_real, [traj_2_l_real[-1]] * 20)
traj_3_u_real = np.append(traj_3_u_real, [traj_3_u_real[-1]] * 20)
traj_5_b_real = np.append(traj_5_b_real, [traj_5_b_real[-1]] * 20)


def hyper_param_tune():
    def objective(params):
        print(f"Current params:")
        print(params)

        # client_2_l.update_configuration({'p': params['joint_2_l/p']})
        # client_2_l.update_configuration({'i': params['joint_2_l/i']})
        # client_2_l.update_configuration({'d': params['joint_2_l/d']})

        # client_3_u.update_configuration({'p': params['joint_3_u/p']})
        # client_3_u.update_configuration({'i': params['joint_3_u/i']})
        # client_3_u.update_configuration({'d': params['joint_3_u/d']})

        client_5_b.update_configuration({'p': params['joint_5_b/p']})
        client_5_b.update_configuration({'i': params['joint_5_b/i']})
        client_5_b.update_configuration({'d': params['joint_5_b/d']})

        robotic_arm.reset()
        traj_2_l, traj_3_u, traj_5_b = [], [], []

        for i in range(1, 50):

            state = robotic_arm.get_state()
            traj_2_l.append(state[0])
            traj_3_u.append(state[1])
            traj_5_b.append(state[2])

            if i == 10:
                action = np.array([-0.5, 0.5, 0.5, 0.99])
                reward, done, termination_reason, distance, success = robotic_arm.step(action)

            elif i == 20:
                action = np.array([0.5, -0.5, -0.5, 0.99])
                reward, done, termination_reason, distance, success = robotic_arm.step(action)

            else:
                time.sleep(1.0 / robotic_arm.UPDATE_RATE)

        time.sleep(0.5)

        traj_2_l = np.array(traj_2_l)
        traj_3_u = np.array(traj_3_u)
        traj_5_b = np.array(traj_5_b)

        error_2_l = np.sum((traj_2_l_real[:30] - traj_2_l[:30]) ** 2) + 3 * np.sum(
            (traj_2_l_real[30:] - traj_2_l[30:]) ** 2)
        error_3_u = np.sum((traj_3_u_real[:30] - traj_3_u[:30]) ** 2) + 3 * np.sum(
            (traj_3_u_real[30:] - traj_3_u[30:]) ** 2)
        error_5_b = np.sum((traj_5_b_real[:30] - traj_5_b[:30]) ** 2) + 3 * np.sum(
            (traj_5_b_real[30:] - traj_5_b[30:]) ** 2)

        total_error = error_5_b

        print(f"Total Error: {total_error} \n"
              f"error_2_l: {error_2_l}, error_3_u: {error_3_u}, error_5_b: {error_5_b} \n"
              f"-----")

        return total_error

    client_2_l = dynamic_reconfigure.client.Client('/motoman_gp8/gp8_controller/gains/joint_2_l')
    client_3_u = dynamic_reconfigure.client.Client('/motoman_gp8/gp8_controller/gains/joint_3_u')
    client_5_b = dynamic_reconfigure.client.Client('/motoman_gp8/gp8_controller/gains/joint_5_b')
    robotic_arm = RoboticArm()

    # Hyperparams space
    space = {
        # "joint_2_l/p": hp.uniform("joint_2_l/p", 10, 800),
        # "joint_2_l/d": hp.uniform("joint_2_l/d", 10, 800),
        # "joint_2_l/i": hp.uniform("joint_2_l/i", 10, 800),

        # "joint_3_u/p": hp.uniform("joint_3_u/p", 10, 500),
        # "joint_3_u/d": hp.uniform("joint_3_u/d", 10, 500),
        # "joint_3_u/i": hp.uniform("joint_3_u/i", 10, 500),

        "joint_5_b/p": hp.uniform("joint_5_b/p", 1.0, 100),
        "joint_5_b/d": hp.uniform("joint_5_b/d", 1.0, 100),
        "joint_5_b/i": hp.uniform("joint_5_b/i", 1.0, 100),
    }

    # Hyperparams space for fine-tuning around a good working point
    # space_tune = {
    #     "joint_2_l/p": hp.uniform("joint_2_l/p", 10, 800),
    #     "joint_2_l/d": hp.uniform("joint_2_l/d", 10, 800),
    #     "joint_2_l/i": hp.uniform("joint_2_l/i", 10, 800),
    #
    #     "joint_3_u/p": hp.uniform("joint_3_u/p", 10, 500),
    #     "joint_3_u/d": hp.uniform("joint_3_u/d", 10, 500),
    #     "joint_3_u/i": hp.uniform("joint_3_u/i", 10, 500),
    #
    #     "joint_5_b/p": hp.uniform("joint_5_b/p", 1.0, 100),
    #     "joint_5_b/d": hp.uniform("joint_5_b/d", 1.0, 100),
    #     "joint_5_b/i": hp.uniform("joint_5_b/i", 1.0, 100),
    # }

    algo = tpe.suggest
    # spark_trials = SparkTrials()
    best_result = fmin(
        fn=objective,
        space=space,
        algo=algo,
        max_evals=100)

    print(f"Best params: \n")
    print(space_eval(space, best_result))

    with open('./best_params.yaml', 'w') as outfile:
        yaml.dump(space_eval(space, best_result), outfile, default_flow_style=False)

    print("Finished")


if __name__ == '__main__':
    hyper_param_tune()
