#!/usr/bin/python
import rospy
import actionlib
import control_msgs.msg
 
rospy.init_node('gripper_control')
 
# Create an action client
client = actionlib.SimpleActionClient(
    '/gripper_controller/gripper_cmd',  # namespace of the action topics
    control_msgs.msg.GripperCommandAction  # action type
)
    
# Wait until the action server has been started and is listening for goals
client.wait_for_server()
 
# Create a goal to send (to the action server)
goal = control_msgs.msg.GripperCommandGoal()
goal.command.position = 0.4   # From 0.0 to 0.8
goal.command.max_effort = -1.0  # Do not limit the effort
client.send_goal(goal)
 
client.wait_for_result()
