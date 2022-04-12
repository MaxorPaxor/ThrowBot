#!/usr/bin/env python
'''
rPR: 0 < rPR < 255 - Psition
rSP: 0 < rSP < 255 - Speed
rFR: 0 < rFR < 255 - Force
'''

import roslib; roslib.load_manifest('robotiq_2f_gripper_control')
import rospy
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output  as outputMsg
from time import sleep


class OpenCloseRobotiqGripper(object):
    def __init__(self):
        rospy.init_node('Robotiq2FGripperSimpleController')
    
        self.pub = rospy.Publisher('Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)
        self.command = outputMsg.Robotiq2FGripper_robot_output()
        # speed = 255
        # force = 150
        self.activatreGripper()
        # self.CloseGripper(self.command,speed, force )
        # self.OpenGripper(self.command,speed, force)
        # self.GoToPose(self.command,0)
    
    def Publish(self):
        g = 0
        while g <= 1:
            self.pub.publish(self.command)
            g = self.pub.get_num_connections()
                # print(g,"pub.get_num_connections()")
            rospy.sleep(0.3)
            if g >1 :
                break            

    def activatreGripper(self):
        print("enter activatreGripper")   
        self.command.rACT = 1
        self.command.rGTO = 1
        self.command.rATR = 0
        self.command.rPR = 0
        self.command.rSP = 255
        self.command.rFR = 150
        self.Publish()
        print("publish")
        return self.command
   
    def CloseGripper(self,speed, force):
        print("enter CloseGripper")
        if speed > 255:
            speed = 255
        if speed < 0:
            speed = 0
        if force > 255:
            force = 255
        if force < 0 :
            force = 0

        self.command.rACT = 1
        self.command.rGTO = 1
        self.command.rATR = 0
        self.command.rPR = 255
        self.command.rSP = speed
        self.command.rFR = force
        self.Publish()
        print("Pub close")
        return self.command

    def OpenGripper(self,speed, force):
        print("enter OpeneGripper")
        if speed > 255:
            speed = 255
        if speed < 0:
            speed = 0
        if force > 255:
            force = 255
        if force < 0 :
            force = 0

        self.command.rACT = 1
        self.command.rGTO = 1
        self.command.rATR = 0
        self.command.rPR = 0
        self.command.rSP = speed
        self.command.rFR = force
        self.Publish()
        print("Pub open")
        return self.command

    def GoToPose(self, nex_pose, speed, force) :
        print("enter GoToPose")
        if nex_pose > 255:
            nex_pose = 255
            rospy.logerr("Next Pose Most be between 0-255")
        if nex_pose < 0:
            nex_pose = 0
            rospy.logerr("Next Pose Most be between 0-255")
        else :
            nex_pose = nex_pose

        if speed > 255:
            speed = 255
        if speed < 0:
            speed = 0
        if force > 255:
            force = 255
        if force < 0 :
            force = 0
        
        self.command.rACT = 1
        self.command.rGTO = 1
        self.command.rATR = 0
        self.command.rPR = nex_pose
        self.command.rSP = speed
        self.command.rFR = force
        self.Publish()
        print("Gripper move")
        return self.command
        
                        
# if __name__ == '__main__':
    # a = OpenCloseRobotiqGripper()
    # a.CloseGripper(255,255)
    # a.OpenGripper(255,255)
