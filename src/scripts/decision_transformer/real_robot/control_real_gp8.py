import rospy
import numpy as np
import time

from robotiqGripper import RobotiqGripper

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class GP8JointCommandDT:
    pass


class GP8JointCommand:
    def __init__(self, gripper_bool=False, rate=20):
        self.rate = rate
        self.duration = 1.0 / self.rate
        self.step = 1
        self.time_from_start = 0
        self.point = None
        self.old_point = None

        joint_states_sub = rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)
        self.joint_names = None
        self.joint_angles = None
        self.joint_velocity = None

        joints_data = None
        while joints_data is None:
            try:
                joints_data = rospy.wait_for_message("/joint_states", JointState, timeout=5)
            except e:
                rospy.logwarn("Time out ")
                pass

        # self.publish_topic = rospy.Publisher('/joint_path_command', JointTrajectory, queue_size=1)
        self.publish_topic = rospy.Publisher('/joint_command', JointTrajectory, queue_size=1)
        # self.publish_topic = rospy.Publisher('/motoman_gp8/gp8_controller/command', JointTrajectory, queue_size=1)
        self.rate_ros = rospy.Rate(self.rate)

        while self.publish_topic.get_num_connections() == 0:
            self.rate_ros.sleep()
        print("Established Connection")

        # Connect to gripper
        self.gripper_bool = gripper_bool
        self.gripper_status = None
        if self.gripper_bool:
            self.gripper = RobotiqGripper("/dev/ttyUSB0", slaveaddress=9)
            self.gripper._aCoef = -4.7252
            self.gripper._bCoef = 1086.8131
            self.gripper.closemm = 0
            self.gripper.openmm = 860
            print("Gripper is ready")

    def joint_states_callback(self, msg):
        self.joint_names = msg.name
        self.joint_angles = msg.position
        self.joint_velocity = msg.velocity

    def move_to_joint(self, point, dt=None):

        if dt is None:
            dt = self.duration

        if len(point) == 7:
            self.point = point[:-1]
            if point[-1] == 0.19:
                self.gripper_status = 'close'
            elif point[-1] == 0.05:
                self.gripper_status = 'open'
            else:
                print('Gripper value error')

        elif len(point) == 6:
            self.point = point
            self.gripper_status = 'close'

        if self.old_point is None:
            self.old_point = self.point

        return self.joint_trajectory_msg(dt)

    def joint_trajectory_msg(self, dt):
        joint_traj = JointTrajectory()

        joint_traj.joint_names = self.joint_names
        joint_traj.header.frame_id = ''
        joint_traj.header.seq = 0

        points = [self.joint_trajectory_point(dt)]
        joint_traj.points = points

        return joint_traj

    def joint_trajectory_point(self, dt):
        next_point = JointTrajectoryPoint()
        next_point.positions = self.point
        next_point.velocities = self.calculate_joint_velocity(dt)

        next_point.time_from_start = rospy.Duration(dt * self.step)
        self.step += 1

        return next_point

    def calculate_joint_velocity(self, dt):
        velocity_vector = []

        for joint in range(len(self.point)):
            joint_velocity = (self.point[joint] - self.old_point[joint])
            joint_velocity = joint_velocity / dt
            velocity_vector.append(joint_velocity)

        return velocity_vector

    def publish(self, trajectory):
        self.publish_topic.publish(trajectory)
        time.sleep(1.0 / self.rate)
        # self.rate_ros.sleep()

        if self.gripper_bool:
            if self.gripper_status == 'close':
                self.gripper.goTomm(270, 255, 255)
            elif self.gripper_status == 'open':
                self.gripper.goTomm(350, 255, 255)

        self.old_point = self.point

    @staticmethod
    def go_to_start():
        # Publish current position at first
        trajectory = manipulator.move_to_joint(manipulator.joint_angles)
        manipulator.publish(trajectory)

        # Go to start
        START_POS = [0.785398, 0.5, -0.3, 0.0, -1.5, -0.785398, 0.19]
        trajectory = manipulator.move_to_joint(START_POS, dt=1)
        manipulator.publish(trajectory)
        time.sleep(1)
        print("Ready")

    @staticmethod
    def throw_recorded():
        PATH = [[0.785398, 0.35009002868640265, -0.07471075628466836, -0.0015012343380202964,
                 -1.1615545296518017, -0.785398, 0.19],
                [0.785398, 0.35536920057449, -0.03200373604862666, -0.002013886892662242,
                 -1.0186948750409055, -0.785398, 0.19],
                [0.785398, 0.2670402373935451, 0.10655824535072317, -0.0025030861799759663,
                 -0.7715734412181068, -0.785398, 0.05], ]

        # PATH = [[0.785398, 0.35009002868640265, -0.07471075628466836, -0.0015012343380202964,
        #          -1.1615545296518017, -0.785398, 0.19],
        #         [0.785398, 0.35536920057449, -0.03200373604862666, -0.002013886892662242,
        #          -1.0186948750409055, -0.785398, 0.19],
        #         [0.785398, 0.2670402373935451, 0.10655824535072317, -0.0025030861799759663,
        #          -0.7715734412181068, -0.785398, 0.19],
        #         [0.785398, 0.2670402373935451, 0.10655824535072317, -0.0025030861799759663,
        #          -0.4715734412181068, -0.785398, 0.05]
        #         ]

        # Publish current position at first
        trajectory = manipulator.move_to_joint(manipulator.joint_angles)
        manipulator.publish(trajectory)

        for _ in range(1):
            for p in PATH:
                time.sleep(0.003)  # Simulates network pass
                trajectory = manipulator.move_to_joint(p)
                manipulator.publish(trajectory)


if __name__ == '__main__':
    ####
    rospy.init_node('move_group_interface', anonymous=True)
    manipulator = GP8JointCommand()

    manipulator.go_to_start()
    # manipulator.throw_recorded()
