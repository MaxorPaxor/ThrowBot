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
        self.UPDATE_RATE = 10  # HZ (10)
        self.total_time = 1.0  # sec (1.0)
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
        self.target_radius = 0.1  # meters

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

        self.max_speed_factor = 1.0  # % of max speed for safety reasons
        self.gripper_thresh = 0.8385  # Gripper open threshold (0.82)

        # Init connections
        # Publish
        self.pub_command = rospy.Publisher('/motoman_gp8/gp8_controller/command', JointTrajectory, queue_size=1)
        self.pub_gazebo = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

        # Subscribe
        self.sub_joint_states = rospy.Subscriber("/motoman_gp8/joint_states", JointState, self.joint_states_callback)
        self.sub_model_states = rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_callback)
        self.sub_link_states = rospy.Subscriber("/gazebo/link_states", LinkStates, self.link_states_callback)

        rospy.init_node('test_rl', anonymous=True)
        self.rate = rospy.Rate(self.UPDATE_RATE)

        while self.pub_command.get_num_connections() == 0 or \
                self.pub_gazebo.get_num_connections() == 0:
            self.rate.sleep()

        print("Established Connection")
        print("Restarting environment...")
        self.reset()
        self.initial_pos = self.object_position

    def reset_gazebo_object(self):
        """
        Resets throwing object position and orientation
        """

        model_state = ModelState()
        model_state.model_name = 'simple_box'
        model_state.pose.position.x = 0.28
        model_state.pose.position.y = 0.0
        model_state.pose.position.z = 0.20
        model_state.pose.orientation.x = 0.0
        model_state.pose.orientation.y = 2.2
        model_state.pose.orientation.z = 0.0
        model_state.pose.orientation.w = 1.0

        self.pub_gazebo.publish(model_state)
        self.rate.sleep()

    def reset(self):
        """
        Resets the robotic arm to its initial state
        Resets object
        """

        trajectory = self.trajectory(gripper=1, dt=0.5)  # Place the arm in its origin pose
        self.pub_command.publish(trajectory)
        self.rate.sleep()
        time.sleep(1.0)
        self.reset_gazebo_object()  # Teleport the object to the arm's gripper
        time.sleep(0.4)

        self.state_mem.clear()
        self.curr_time = 0  # sec
        self.curr_step = 0

        if self.no_rotation:
            self.velocity = [0, 0, 0, 0]
        else:
            self.velocity = [0, 0, 0, 0, 0, 0, 0]

    def trajectory(self, j1=0.0, j2=0.5, j3=-0.3, j4=0.0,
                   j5=-1.5, j6=0.0, gripper=-1.0, dt=None):
        """
        Commands robotic arm joints position
        j1=0.0, j2=0.5, j3=-0.3, j4=0.0,
                   j5=-1.5, j6=0.0, gripper=-1.0, dt=None
        """

        trajectory = JointTrajectory()
        point = JointTrajectoryPoint()

        trajectory.header.stamp = rospy.Time.now()
        trajectory.header.frame_id = "/base_link"

        trajectory.joint_names.append("joint_1_s")
        trajectory.joint_names.append("joint_2_l")
        trajectory.joint_names.append("joint_3_u")
        trajectory.joint_names.append("joint_4_r")
        trajectory.joint_names.append("joint_5_b")
        trajectory.joint_names.append("joint_6_t")
        trajectory.joint_names.append("finger_joint")

        if gripper >= 0:
            gripper = 0.625
        else:
            gripper = 0.5

        point.positions.append(j1)  # 0.0
        point.positions.append(j2)  # 0.5
        point.positions.append(j3)
        point.positions.append(j4)
        point.positions.append(j5)
        point.positions.append(j6)
        point.positions.append(gripper)  # 0.05 - 0.7: 0.2=close, 0.05=open.

        if dt is None:
            dt = 1.0 / self.UPDATE_RATE

        point.time_from_start = rospy.Duration(dt)
        trajectory.points.append(point)

        return trajectory

    def vel_trajectory(self, vel_1=0.0, vel_2=0.0, vel_3=0.0, vel_4=0.0,
                       vel_5=0.0, vel_6=0.0, gripper=1.0, dt=None):
        """
        Commands robotic arm joints velocities
        """

        trajectory = JointTrajectory()
        point = JointTrajectoryPoint()

        trajectory.header.stamp = rospy.Time.now()
        trajectory.header.frame_id = "/base_link"

        trajectory.joint_names.append("joint_1_s")
        trajectory.joint_names.append("joint_2_l")
        trajectory.joint_names.append("joint_3_u")
        trajectory.joint_names.append("joint_4_r")
        trajectory.joint_names.append("joint_5_b")
        trajectory.joint_names.append("joint_6_t")
        trajectory.joint_names.append("finger_joint")

        if gripper >= self.gripper_thresh:  # Gripper is closed
            gripper = 0.625

        else:  # Gripper is open
            gripper = 0.5
            # vel_1, vel_2, vel_3, vel_4, vel_5, vel_6 = 0, 0, 0, 0, 0, 0  # stop arm movement
            # dt = 0.02

        # Current_position + NN_velocity_output(-1 < V < +1) * max_speed(rd/s) * time(1/frequency)
        pos_1 = self.angles[1] + vel_1 * 1.0 / self.UPDATE_RATE
        pos_2 = self.angles[2] + vel_2 * 1.0 / self.UPDATE_RATE
        pos_3 = self.angles[3] + vel_3 * 1.0 / self.UPDATE_RATE
        pos_4 = self.angles[4] + vel_4 * 1.0 / self.UPDATE_RATE
        pos_5 = self.angles[5] + vel_5 * 1.0 / self.UPDATE_RATE
        pos_6 = self.angles[6] + vel_6 * 1.0 / self.UPDATE_RATE

        point.positions.append(pos_1)
        point.positions.append(pos_2)
        point.positions.append(pos_3)
        point.positions.append(pos_4)
        point.positions.append(pos_5)
        point.positions.append(pos_6)
        point.positions.append(gripper)

        # point.velocities.append(vel_1)
        # point.velocities.append(vel_2)
        # point.velocities.append(vel_3)
        # point.velocities.append(vel_4)
        # point.velocities.append(vel_5)
        # point.velocities.append(vel_6)
        # point.velocities.append(0)

        if dt is None:
            dt = 1.0 / self.UPDATE_RATE

        point.time_from_start = rospy.Duration(dt)
        trajectory.points.append(point)

        return trajectory

    def joint_states_callback(self, msg):
        """
        Robotic arm's joints callback function
        """

        if self.joint_names != msg.name:  # Update joints' names if needed
            self.joint_names = msg.name

        self.angles = msg.position
        self.state_mem.append(self.angles)

    def link_states_callback(self, msg):
        """
        Robotic arm's links callback function
        """

        idx = msg.name.index('motoman_gp8::link_6_t')
        self.end_link_position_z = msg.pose[idx].position.z

    def model_states_callback(self, msg):
        """
        models callback function
        """

        idx_object = msg.name.index('simple_box')
        idx_target = msg.name.index('target')

        self.object_position = np.array([msg.pose[idx_object].position.x,
                                         msg.pose[idx_object].position.y,
                                         msg.pose[idx_object].position.z])

        self.object_height = msg.pose[idx_object].position.z

        self.target_position = np.array([msg.pose[idx_target].position.x,
                                         msg.pose[idx_target].position.y,
                                         msg.pose[idx_target].position.z])

    def update_target(self, target):
        """
        Updates the target from the agent
        Updates visual target in Gazebo simulation
        """

        self.target = target

        model_state = ModelState()
        model_state.model_name = 'target'
        model_state.pose.position.x = self.target[0]
        model_state.pose.position.y = 0.0
        model_state.pose.position.z = 0.0

        self.pub_gazebo.publish(model_state)
        self.rate.sleep()

    def get_state(self):
        """
        Returns the state of the arm to the RL algo
        """
        angles = []

        for joint in self.joints:
            idx = self.joint_names.index(joint)
            angles.append(self.angles[idx])

        # state = list(angles) + list(self.velocity)
        state = np.array(angles)

        return state

    def get_n_state(self):
        """
        Returns the last 3 states of the arm to the RL algo
        """

        while len(self.state_mem) != self.number_states:
            time.sleep(0.05)  # Make sure the buffer is full with n states

        state = []

        for s in self.state_mem:
            angles = []
            for joint in self.joints:
                idx = self.joint_names.index(joint)
                angles.append(s[idx])
            state += angles

        return np.array(state)

    def reward_sparse(self, obj_pos=None, target=None):
        """
        Sparse reward function - For HER
        """

        if target is None:
            target = self.target

        if obj_pos is None:
            obj_pos = self.object_position

        distance = np.sqrt((obj_pos[0] - target[0]) ** 2 +
                           (obj_pos[1] - target[1]) ** 2)

        if distance <= self.target_radius and obj_pos[0] > self.initial_pos[0]:
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

    def wait_for_object_to_touch_ground(self):
        t1 = time.time()
        while not self.object_height <= 1.2 * 0.5 * 0.015 * np.sqrt(2):  # Make sure object is on the ground
            time.sleep(0.005)
            if time.time() - t1 > 3:  # timeout
                break

    def step(self, velocity_vector):
        """
        Performs 1 step for the robotic arm and checks if stop conditions are met
        Returns reward, done flag and termination reason
        """
        velocity_vector = self.proj_on_max_speed(velocity_vector)  # Rescale for maximum speed
        velocity_vector = self.smooth_velocity(velocity_vector)  # Apply complimentary filter on velocity vector

        if self.no_rotation:
            self.velocity = velocity_vector
            vel_2, vel_3, vel_5, gripper = velocity_vector
            vel_1, vel_4, vel_6 = 0.0, 0.0, 0.0
        else:
            self.velocity = velocity_vector
            vel_1, vel_2, vel_3, vel_4, vel_5, vel_6, gripper = velocity_vector

        trajectory = self.vel_trajectory(vel_1, vel_2, vel_3, vel_4, vel_5, vel_6, gripper)
        self.pub_command.publish(trajectory)

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
            done = True

            # If object is released, wait for it to fall and reward the action
            if gripper < self.gripper_thresh:
                termination_reason = "Gripper was opened with value: {}".format(gripper)
                trajectory = self.vel_trajectory(0, 0, 0, 0, 0, 0, gripper)
                self.pub_command.publish(trajectory)
                self.wait_for_object_to_touch_ground()

                # Successful throw reward:
                reward = self.reward_sparse()
                distance = self.object_position

                success = True

            else:
                termination_reason = "Time is up: {}".format(self.curr_time)
                reward = -1.0
                distance = self.object_position

                success = False

        else:  # Gripper is closed and time is not up
            if self.object_height <= 0.5 * 0.015 * np.sqrt(2):  # if too close to ground
                termination_reason = "Object is too close to ground: {}".format(self.object_height)
                done = True
                reward = -1.0
                distance = self.object_position
                success = False

            else:  # Continue episode
                termination_reason = None
                done = False
                reward = 0.0
                distance = self.object_position
                success = False

        return reward, done, termination_reason, distance, success

    @staticmethod
    def ballistic_model(distance):
        """
        Analytical Ballistic Model Throw
        Input: Distance [0.9 - 2.0]
        Output: Velocity v
        """
        release_x = 0.813
        release_z = 0.349
        position_x = distance
        g = 9.81

        v_nom = g * (position_x - release_x)**2 - 2 * release_z
        v_denom = position_x - release_x
        v = np.sqrt(v_nom / v_denom)

        return v

    def ballistic_throw(self, distance):
        """
        Analytical Ballistic Model Throw
        Input: Distance [0.9 - 2.0]
        """

        # Theoretical velocity
        v = self.ballistic_model(distance)

        # Numerical Mapping between v and dt
        dt = self.calib_distance(v)

        # Throw
        traj1 = self.trajectory(j1=0.0, j2=0.5, j3=-0.3, j4=0.0,
                                j5=-1.5, j6=0.0, gripper=1.0, dt=dt)
        self.pub_command.publish(traj1)
        # self.rate.sleep()
        time.sleep(dt)

        traj2 = self.trajectory(j1=0.0, j2=0.7, j3=0.1, j4=0.0,
                                j5=-1.2, j6=0.0, gripper=1.0, dt=dt)
        self.pub_command.publish(traj2)
        # self.rate.sleep()
        time.sleep(dt)

        traj3 = self.trajectory(j1=0.0, j2=0.8, j3=0.4, j4=0.0,
                                j5=-0.9, j6=0.0, gripper=1.0, dt=dt)
        self.pub_command.publish(traj3)
        # self.rate.sleep()
        time.sleep(dt)

        traj4 = self.trajectory(j1=0.0, j2=0.9, j3=0.7, j4=0.0,
                                j5=-0.586, j6=0.0, gripper=-1.0, dt=dt)
        self.pub_command.publish(traj4)
        # self.rate.sleep()
        time.sleep(dt)

        self.wait_for_object_to_touch_ground()
        distance = self.object_position[0]

        print(f"Theoretical Velocity: {v}")
        print(f"dt: {dt}")
        print(f"Landing Position: {distance}")

    @staticmethod
    def calib_distance(distance):
        # Calculate bias
        # x = np.array([4.55, 3.63, 3.22, 2.86, 2.64, 2.33, 2.19, 1.94, 1.74, 1.66,
        #               1.52, 1.41, 1.34, 1.29, 1.2, 1.16, 1.14, 1.01, 0.76, 0.58])
        # y = np.array([0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
        #               0.16, 0.17, 0.18, 0.19, 0.2 , 0.21, 0.22, 0.25, 0.3, 0.4])
        # z = np.polyfit(x, y, deg=6)
        # p = np.poly1d(z)

        z = np.array(
            [1.49932003e-03, - 2.47497656e-02,  1.68395038e-01, - 6.11142425e-01,
             1.27145297e+00, - 1.50517863e+00,  9.41862115e-01])
        p = np.poly1d(z)
        dt = p(distance)

        return dt


if __name__ == '__main__':
    robotic_arm = RoboticArm()
    robotic_arm.ballistic_throw(distance=2.0)
    # robotic_arm.reset()
