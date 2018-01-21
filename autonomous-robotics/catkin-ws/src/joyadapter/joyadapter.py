#!/usr/bin/env python
# This file is a modified version of the chatter sample proposed on the ROS wiki page:
# https://raw.github.com/ros/ros_tutorials/kinetic-devel/rospy_tutorials/001_talker_listener/talker.py
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.

"""
this node convert joystick data to twistCommand data for the vrep simulator
"""
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

pub = None # quick and dirty global param

def callback(data):
    """
    function called when joystick data arrives, publish the twist commands directly to /vrep/twistCommand topic
    :param data: the joystick data
    :return:
    """
    # get the global publisher
    global pub
    # build the message
    msg = Twist()
    msg.linear.x = data.axes[1]
    msg.linear.y = data.axes[0]
    msg.angular.z = data.axes[3]
    # log the message
    rospy.loginfo(str(msg))
    # publish the message
    pub.publish(msg)

def joyadapter():
    """
    the behaviour function of the node, init the node, the publisher, and the subscriber.
    :return:
    """
    global pub
    # declare a publisher, that speak on twistCommand topic
    pub = rospy.Publisher('/vrep/twistCommand', Twist, queue_size=10)
    # init the node
    rospy.init_node('joyadapter', anonymous=True)
    # declare the subscriber that listen for joystick
    rospy.Subscriber("/joy", Joy, callback)
    # prevent the program from stopping
    rospy.spin()

if __name__ == '__main__':
    try:
        joyadapter()
    except rospy.ROSInterruptException:
        pass
