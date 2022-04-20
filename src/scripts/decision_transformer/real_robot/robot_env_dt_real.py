#!/usr/bin/env python

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelState, ModelStates, LinkStates

from collections import deque
import numpy as np
import time


class RoboticArm:
    def __init__(self):
        # Global params
        self.UPDATE_RATE = 20  # HZ
        self.total_time = 0.5  # sec
        self.number_steps = int(self.total_time * self.UPDATE_RATE)
        self.no_rotation = True
        self.smooth_factor = 0.0  # 10Hz, 0.5sec, 0.5sf

        # Init attributes
        self.object_distance = 0
        self.curr_time = 0  # sec
        self.curr_step = 0
        self.joint_names = None
        self.angles = None

        # HER attributes
        self.her = True
        self.target_radius = 0.2  # meters

        self.target = np.array([2, 0, 0])
        # Note: Target will be static unless agent class will override it when using HER

        # States
        self.number_states = 1
        self.state_mem = deque(maxlen=self.number_states)  # popleft()

        if self.no_rotation:
            self.velocity = [0, 0, 0, 0]  # rad/s
            self.joints = np.array(['joint_2_l', 'joint_3_u', 'joint_5_b', 'finger_joint'])
            self.max_speed = np.array([385, 520, 550, 1])  # deg/s
        else:
            self.velocity = [0, 0, 0, 0, 0, 0, 0]  # rad/s
            self.joints = np.array(['joint_1_s', 'joint_2_l', 'joint_3_u', 'joint_4_r',
                                    'joint_5_b', 'joint_6_t', 'finger_joint'])
            self.max_speed = np.array([455, 385, 520, 550, 550, 1000, 1])  # deg/s

        self.max_speed_factor = 1.3  # % of max speed for safety reasons
        self.gripper_thresh = 0.8  # Gripper open threshold

        # Init connections
        rospy.init_node('test_rl', anonymous=True)
        self.rate = rospy.Rate(self.UPDATE_RATE)

        # Publish
        self.pub_command = rospy.Publisher('/joint_command', JointTrajectory, queue_size=1)
        while self.pub_command.get_num_connections() == 0:
            self.rate.sleep()

        # Subscribe
        self.sub_joint_states = rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        joints_data = None
        while joints_data is None:
            try:
                joints_data = rospy.wait_for_message('/joint_states', JointState, timeout=5)
            except Exception as e:
                print(e)
                pass

        print("Established Connection")

    def update_target(self, target):
        self.target = target

    def vel_trajectory(self, vel_1=0.0, vel_2=0.0, vel_3=0.0, vel_4=0.0,
                       vel_5=0.0, vel_6=0.0, gripper=1.0, dt=None):
        """
        Commands robotic arm joints velocities
        """

        trajectory = JointTrajectory()
        point = JointTrajectoryPoint()

        trajectory.header.stamp = rospy.Time.now()
        trajectory.header.frame_id = ''
        trajectory.header.seq = 0

        trajectory.joint_names.append("joint_1_s")
        trajectory.joint_names.append("joint_2_l")
        trajectory.joint_names.append("joint_3_u")
        trajectory.joint_names.append("joint_4_r")
        trajectory.joint_names.append("joint_5_b")
        trajectory.joint_names.append("joint_6_t")
        # trajectory.joint_names.append("finger_joint")

        if gripper >= self.gripper_thresh:  # Gripper is closed
            gripper = 0.19

        else:  # Gripper is open
            gripper = 0.05
            # vel_1, vel_2, vel_3, vel_4, vel_5, vel_6 = 0, 0, 0, 0, 0, 0  # stop arm movement
            # dt = 0.02

        # Current_position + NN_velocity_output(-1 < V < +1) * max_speed(rd/s) * time(1/frequency)
        pos_1 = self.angles[0] + vel_1 * 1.0 / self.UPDATE_RATE
        pos_2 = self.angles[1] + vel_2 * 1.0 / self.UPDATE_RATE
        pos_3 = self.angles[2] + vel_3 * 1.0 / self.UPDATE_RATE
        pos_4 = self.angles[3] + vel_4 * 1.0 / self.UPDATE_RATE
        pos_5 = self.angles[4] + vel_5 * 1.0 / self.UPDATE_RATE
        pos_6 = self.angles[5] + vel_6 * 1.0 / self.UPDATE_RATE

        # Boundaries j3 -0.5 - 1.8 / j5 -1.7 - 1.4
        # if pos_2 < -0.5:
        #     pos_2 = -0.5
        # if pos_2 > 0.5:
        #     pos_2 = 0.5
        # if pos_3 < -0.5:
        #     pos_3 = -0.5
        # if pos_3 > 1.8:
        #     pos_3 = 1.8
        # if pos_5 < -1.7:
        #     pos_5 = -1.7
        # if pos_5 > 1.4:
        #     pos_5 = 1.4

        point.positions.append(pos_1)
        point.positions.append(pos_2)
        point.positions.append(pos_3)
        point.positions.append(pos_4)
        point.positions.append(pos_5)
        point.positions.append(pos_6)

        # point.velocities.append(vel_1)
        # point.velocities.append(vel_2)
        # point.velocities.append(vel_3)
        # point.velocities.append(vel_4)
        # point.velocities.append(vel_5)
        # point.velocities.append(vel_6)

        point.velocities.append(0)
        point.velocities.append(0)
        point.velocities.append(0)
        point.velocities.append(0)
        point.velocities.append(0)
        point.velocities.append(0)

        if dt is None:
            dt = 1.0 / self.UPDATE_RATE

        point.time_from_start = rospy.Duration(dt * self.curr_step)
        trajectory.points.append(point)

        return trajectory

    def reset_arm(self):
        """
        Commands robotic arm joints velocities
        """
        self.curr_time = 0  # sec
        self.curr_step = 0

        trajectory = JointTrajectory()
        point = JointTrajectoryPoint()

        trajectory.header.stamp = rospy.Time.now()
        trajectory.header.frame_id = ''
        trajectory.header.seq = 0

        trajectory.joint_names.append("joint_1_s")
        trajectory.joint_names.append("joint_2_l")
        trajectory.joint_names.append("joint_3_u")
        trajectory.joint_names.append("joint_4_r")
        trajectory.joint_names.append("joint_5_b")
        trajectory.joint_names.append("joint_6_t")
        # trajectory.joint_names.append("finger_joint")

        # Current_position + NN_velocity_output(-1 < V < +1) * max_speed(rd/s) * time(1/frequency)
        pos_1 = -1.5707
        pos_2 = 0.5
        pos_3 = -0.3
        pos_4 = 0.0
        pos_5 = -1.5
        pos_6 = -0.785398

        point.positions.append(pos_1)
        point.positions.append(pos_2)
        point.positions.append(pos_3)
        point.positions.append(pos_4)
        point.positions.append(pos_5)
        point.positions.append(pos_6)

        point.velocities.append(0)
        point.velocities.append(0)
        point.velocities.append(0)
        point.velocities.append(0)
        point.velocities.append(0)
        point.velocities.append(0)

        point.time_from_start = rospy.Duration(1)
        trajectory.points.append(point)
        self.pub_command.publish(trajectory)

    def joint_states_callback(self, msg):
        """
        Robotic arm's joints callback function
        """

        if self.joint_names != msg.name:  # Update joints' names if needed
            self.joint_names = msg.name

        self.angles = msg.position
        self.state_mem.append(self.angles)

    def get_state(self):
        """
        Returns the state of the arm to the RL algo
        """

        # rospy.wait_for_message("/motoman_gp8/joint_states", JointState)

        angles = []
        for joint in self.joints[:-1]:
            idx = self.joint_names.index(joint)
            angles.append(self.angles[idx])

        angles.append(1.0)

        state = np.array(angles)

        return state

    def reward_sparse(self, obj_pos=None, target=None):
        """
        Sparse reward function - For HER
        """

        if target is None:
            target = self.target

        distance = np.sqrt((obj_pos[0] - target[0]) ** 2 +
                           (obj_pos[1] - target[1]) ** 2)  # +
        # (obj_pos[2] - target[2]) ** 2)

        if distance <= self.target_radius:  # and obj_pos[0] > self.initial_pos[0]:
            return 1.0
        else:
            return -1.0

    def smooth_velocity(self, new_velocity):
        """
        Smoothens the velocity vector in time, using complimentary filter
        """

        old_velocity = np.array(self.velocity)
        smoothed_velocity = old_velocity * self.smooth_factor + new_velocity * (1 - self.smooth_factor)
        smoothed_velocity[-1] = new_velocity[-1]  # Keep original value for gripper
        # print("Old velocity: {}, New velocity: {}, Smoothened velocity: {}".format(old_velocity,
        #                                                                            new_velocity,
        #                                                                            smoothed_velocity))
        return smoothed_velocity

    def proj_on_max_speed(self, velocity_vector):
        """
        Rescales velocities [0, 1] to [0, max_speed]
        Converts to rad/s from deg/s
        """
        velocity_vector_max = velocity_vector * self.max_speed * self.max_speed_factor
        velocity_vector_max = velocity_vector_max * np.pi / 180
        velocity_vector_max[-1] = velocity_vector[-1]
        return velocity_vector_max

    def step(self, velocity_vector):
        """
        Performs 1 step for the robotic arm and checks if stop conditions are met
        Returns reward, done flag and termination reason
        """
        velocity_vector = self.proj_on_max_speed(velocity_vector)  # Rescale for maximum speed
        if self.curr_step > 0:  # Not for first step
            velocity_vector = self.smooth_velocity(velocity_vector)  # Apply complimentary filter on velocity vector

        if self.no_rotation:
            self.velocity = velocity_vector
            vel_2, vel_3, vel_5, gripper = velocity_vector
            vel_1, vel_4, vel_6 = 0, 0, 0
        else:
            self.velocity = velocity_vector
            vel_1, vel_2, vel_3, vel_4, vel_5, vel_6, gripper = velocity_vector

        trajectory = self.vel_trajectory(vel_1, vel_2, vel_3, vel_4, vel_5, vel_6, gripper)
        self.pub_command.publish(trajectory)
        # print(trajectory)

        # self.rate.sleep()
        time.sleep(1.0 / self.UPDATE_RATE)
        self.curr_time += 1.0 / self.UPDATE_RATE
        self.curr_step += 1

        """
        Stop if
        - Time is over
        - Gripper was opened
        - Object touched the ground
        """
        if self.curr_step >= self.number_steps or gripper < self.gripper_thresh:

            # If object is released, wait for it to fall and reward the action
            if gripper < self.gripper_thresh:
                termination_reason = "Gripper was opened with value: {}".format(gripper)
                done = True  # True

            else:
                termination_reason = "Time is up: {}".format(self.curr_time)
                done = True

        else:  # Gripper is closed and time is not up
            termination_reason = None
            done = False

        return done, termination_reason

    def first_step(self, velocity_vector):
        """
        Performs 1 step for the robotic arm and checks if stop conditions are met
        Returns reward, done flag and termination reason
        """
        velocity_vector = self.proj_on_max_speed(velocity_vector)  # Rescale for maximum speed

        if self.no_rotation:
            self.velocity = velocity_vector
            vel_2, vel_3, vel_5, gripper = velocity_vector
            vel_1, vel_4, vel_6 = 0, 0, 0
        else:
            self.velocity = velocity_vector
            vel_1, vel_2, vel_3, vel_4, vel_5, vel_6, gripper = velocity_vector

        trajectory = self.vel_trajectory(vel_1, vel_2, vel_3, vel_4, vel_5, vel_6, gripper)
        self.pub_command.publish(trajectory)
        # print(trajectory)

        # self.rate.sleep()
        time.sleep(1.0 / self.UPDATE_RATE)
        self.curr_time += 1.0 / self.UPDATE_RATE
        self.curr_step += 1


if __name__ == '__main__':
    robotic_arm = RoboticArm()
    for _ in range(100):
        print(robotic_arm.get_state())
        time.sleep(0.05)
