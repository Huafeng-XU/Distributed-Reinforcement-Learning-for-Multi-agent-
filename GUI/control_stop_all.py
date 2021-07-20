#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist



class ControlLane():
    def __init__(self):
        self.pub_cmd_vel = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size = 1)
        self.pub_cmd_vel2 = rospy.Publisher('/robot2/cmd_vel', Twist, queue_size=1)
        self.pub_cmd_vel3 = rospy.Publisher('/robot3/cmd_vel', Twist, queue_size=1)
        self.pub_cmd_vel4 = rospy.Publisher('/robot4/cmd_vel', Twist, queue_size=1)
        self.pub_cmd_vel5 = rospy.Publisher('/robot0/cmd_vel', Twist, queue_size=1)

    def main(self):
        rate = rospy.Rate(1)  # 10hz
        while not rospy.is_shutdown():
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0

            twist.linear.z = 0
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = 0

            # pub.publish(hello_str)
            self.pub_cmd_vel.publish(twist)
            self.pub_cmd_vel2.publish(twist)
            self.pub_cmd_vel3.publish(twist)
            self.pub_cmd_vel4.publish(twist)
            self.pub_cmd_vel5.publish(twist)
            rate.sleep()
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('stop_lane_all')
    node = ControlLane()
    node.main()