#!/usr/bin/env python3

import time
import numpy as np

import rospy
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler

WHEELBASE = 1.04
L_front = 0.55
L_rear = WHEELBASE - L_front

TRACK = 0.985

def normalize_angle(angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
            
        return angle
    
class DeadReckoning():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.psi = 0
        
        self.prev_time = None
        
        # ROS
        rospy.init_node('dead_reckoning')
        self.delta_dist_and_steer_sub = rospy.Subscriber('/delta_dist_and_steer', Float32MultiArray, self.dead_reckoning)
        self.dead_reckoning_pub = rospy.Publisher('/dead_reckoning', Odometry, queue_size=10)
        return
    
    def dead_reckoning(self, delta_dist_and_steer_msg):
        if self.prev_time is None:
            self.prev_time = time.time()
            
        delta_dist_and_steer = delta_dist_and_steer_msg.data
        delta_dist_left, steer = delta_dist_and_steer[0], delta_dist_and_steer[1]
        
        # delta_dist_left: 왼쪽 바퀴 이동 거리
        # steer: 조향각
        # beta: 차축 기준으로 차량 중심에서의 속도 방향
        # R: 차량 회전 반경
        # 
        # beta = arctan( L_rear * tan(steer) / WHEELBASE )
        # R = L_rear / sin(beta)
        # delta_dist = delta_dist_left * (R / (R+TRACK/2))
        
        beta = np.arctan2(L_rear * np.tan(steer), WHEELBASE)
        R = L_rear / np.sin(beta)
        print("회전 반경: ", R)
        
        delta_dist = delta_dist_left * (R / (R+TRACK/2))
        print("이동 거리: ", delta_dist)
        
        cur_time = time.time()
        dt = cur_time - self.prev_time
        
        v = delta_dist / dt
        self.x = self.x + delta_dist * np.cos(self.psi + beta)
        self.y = self.y + delta_dist * np.sin(self.psi + beta)
        self.psi = normalize_angle(self.psi + dt * (v / WHEELBASE) * np.cos(beta) * np.tan(steer))
        print("current pose:", (self.x, self.y, self.psi))
        
        # ROS Publish
        dead_reckoning_result = self.make_odometry_msg(self.x, self.y, self.psi)
        self.dead_reckoning_pub.publish(dead_reckoning_result)
        
        print("")
        return
    
    def make_odometry_msg(self, x, y, yaw):
        odometry_msg = Odometry(frame_id='odom', stamp=rospy.Time.now())
        
        odometry_msg.pose.pose.position.x = x
        odometry_msg.pose.pose.position.y = y
        
        odometry_msg.pose.pose.orientation.x, odometry_msg.pose.pose.orientation.y, \
        odometry_msg.pose.pose.orientation.z, odometry_msg.pose.pose.orientation.w = quaternion_from_euler(0, 0, yaw)
        
        return odometry_msg
    
def main():
    node = DeadReckoning()
    
    try:
        rospy.spin()
        
    except KeyboardInterrupt:
        pass
        
if __name__ == '__main__':
    main()