#!/usr/bin/env python3

import numpy as np

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))
from utils.time_plot import TimePlot
from utils.autocar_util import normalize_angle

from control.longitunial_control import RiseTimeImprovement
from control.filter import low_pass_filter

import rospy
import message_filters

from nav_msgs.msg import Path
from geometry_msgs.msg import TwistWithCovarianceStamped
from ackermann_msgs.msg import AckermannDrive
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32

from tf.transformations import euler_from_quaternion
import time

MAX_STEER = np.radians(27)

class PathTracking():
    def __init__(self):
        # 수정해야하는 부분
        self.frame_id = 'velodyne'
        self.k = 1.0 # for stanley controller
        self.scaling_factor_lat = 0.5
        self.k_for_long = 3.0 # for longitunial controller
        self.max_speed = 3.7 # 4.1 or 3.7
        self.min_speed = 2.0 # 2.0 or 1.8
        self.scaling_factor = np.radians(50)

        # 종방향 제어기
        self.long_controller = RiseTimeImprovement(kp=1, ki=0.0, kd=0.0, brake_gain=100)
        
        # Low Pass Filter
        self.lpf = low_pass_filter(alpha = 0.1)
        
        # Time plotting
        # self.time_plot = TimePlot(data_name=['cte'],\
        #                           title='Cross Track Error')
        
        # self.time_plot_2 = TimePlot(data_name=['cur_speed', 'target_speed', 'brake', 'target_speed_filtered'],\
        #                             title='cur_speed vs. target_speed / brake')
        
        # ROS
        rospy.init_node('path_tracker', anonymous=True)

        self.path_sub = message_filters.Subscriber("/local_path", Path) # 지역 경로
        self.speed_sub = message_filters.Subscriber("/ublox_gps/fix_velocity", TwistWithCovarianceStamped) # 차량 현재 속도
        # self.speed_sub = message_filters.Subscriber("/speed", TwistWithCovarianceStamped)

        self.sync = message_filters.ApproximateTimeSynchronizer([self.path_sub, self.speed_sub], queue_size=10, slop=0.5, allow_headerless=True)
        self.sync.registerCallback(self.callback_path_tracking)

        self.check_long_point_pub = rospy.Publisher('/point_for_long_control', Marker, queue_size=10)
        self.cmd_pub = rospy.Publisher("/erp_command", AckermannDrive, queue_size=10) # 목표 속도, 조향, 브레이크
        self.steer_viz_pub = rospy.Publisher("/steering_angle", Float32, queue_size=10)
        self.measure_speed_pub = rospy.Publisher('/measurement_speed', Float32, queue_size=10)
        return
    
    def callback_path_tracking(self, path_msg, speed_msg):
        first_time = time.perf_counter()
        
        # Speed
        speed = np.sqrt(speed_msg.twist.twist.linear.x **2 + speed_msg.twist.twist.linear.y**2)
        
        # 종, 횡방향 Tracking Controller
        # 주시거리 기반 Stanley + 종방향 제어
        target_steer, target_speed = self.path_tracking(0,0,0, speed, path_msg, -0.17)
        
        target_speed_filtered = self.lpf.update(target_speed)
        # print(target_speed, target_speed_filtered)
        final_speed, final_break = self.long_controller.update(target_speed = target_speed_filtered,\
                                                                measurement_speed = speed)
        
        # time plot
        # self.time_plot_2.update([speed, target_speed, final_break/10, target_speed_filtered])
        
        # print(target_speed, target_steer)
        # ROS Publish
        cmd_msg = AckermannDrive()
        cmd_msg.steering_angle = target_steer # 최종 입력 조향각
        cmd_msg.speed = final_speed # 최종 입력 속도
        cmd_msg.jerk = final_break # 최종 입력 브레이크
        self.cmd_pub.publish(cmd_msg)
        
        # for cmd plot
        steering_angle_msg = Float32(data=np.radians(target_steer))
        measurement_speed_msg = Float32(data=speed)
        self.steer_viz_pub.publish(steering_angle_msg)
        self.measure_speed_pub.publish(measurement_speed_msg)
        
        print("현재 차량 속도, 목표 속도, 입력 속도:", speed, target_speed, final_speed)
        print("소요 시간: ", time.perf_counter()-first_time)
        return
    
    def path_tracking(self, car_x, car_y, car_yaw, speed, path_msg, L):
        # Gain
        k = self.k
        k_for_long = self.k_for_long
            
        LD = k_for_long * speed
        if LD < 1.0:
             LD = 1.0
        
        # Stanely
        min_dist_for_lat = np.inf
        min_dist_for_long = np.inf
        
        min_index_for_lat = 0
        min_index_for_long = 0
        
        front_x = car_x + L * np.cos(car_yaw) # 앞 바퀴 축 x 위치
        front_y = car_y + L * np.sin(car_yaw) # 앞 바퀴 축 y 위치
        
        # Path
        for i, p in enumerate(path_msg.poses):
            map_x = p.pose.position.x
            map_y = p.pose.position.y
            map_yaw = euler_from_quaternion([p.pose.orientation.x, p.pose.orientation.y,\
                                                                p.pose.orientation.z, p.pose.orientation.w])[2]
            
            map_x_for_lat = map_x - 2 * np.cos(map_yaw)
            map_y_for_lat = map_y - 2 * np.sin(map_yaw)
            dist_for_lat = np.sqrt((map_x_for_lat-front_x)**2 + (map_y_for_lat-front_y)**2)
            
            # dx = map_x - front_x
            # dy = map_y - front_y
            
            # dist_for_lat = np.sqrt(dx * dx + dy * dy)
            
            if dist_for_lat < min_dist_for_lat:
                min_dist_for_lat = dist_for_lat
                min_index_for_lat = i
            
            map_x_for_long_ = map_x - LD * np.cos(map_yaw)
            map_y_for_long_ = map_y - LD * np.sin(map_yaw)
            
            dist_for_long = np.sqrt((map_x_for_long_-front_x)**2 + (map_y_for_long_-front_y)**2)
            
            if dist_for_long < min_dist_for_long:
                min_dist_for_long = dist_for_long
                min_index_for_long = i

        # target steer
        map_x = path_msg.poses[min_index_for_lat].pose.position.x
        map_y = path_msg.poses[min_index_for_lat].pose.position.y
        map_yaw = euler_from_quaternion([path_msg.poses[min_index_for_lat].pose.orientation.x, path_msg.poses[min_index_for_lat].pose.orientation.y,\
                                                           path_msg.poses[min_index_for_lat].pose.orientation.z, path_msg.poses[min_index_for_lat].pose.orientation.w])[2]
        
        dx = map_x - front_x
        dy = map_y - front_y
        
        perp_vec = [np.cos(map_yaw + np.pi/2), np.sin(car_yaw + np.pi/2)] # yaw값을 내적하기 위해서 임의로90도 돌려줌 오른쪽이 +방향 왼쪽방향이 -방향
        cte = np.dot([dx, dy], perp_vec)
        
        yaw_term = normalize_angle(map_yaw - car_yaw) #heading error
        cte_term = np.arctan2(k*cte, max(1.0, speed))
        
        target_steer = self.scaling_factor_lat * (yaw_term + cte_term)
        target_steer = np.rad2deg(-target_steer)

        # =============================================================================================
        # target_speed
        map_x_for_long = path_msg.poses[min_index_for_long].pose.position.x
        map_y_for_long = path_msg.poses[min_index_for_long].pose.position.y
        map_yaw_for_long = euler_from_quaternion([path_msg.poses[min_index_for_long].pose.orientation.x, path_msg.poses[min_index_for_long].pose.orientation.y,\
                                                           path_msg.poses[min_index_for_long].pose.orientation.z, path_msg.poses[min_index_for_long].pose.orientation.w])[2]
        yaw_term_for_long = normalize_angle(map_yaw_for_long - car_yaw)
        
        target_speed = self.max_speed - abs(yaw_term_for_long)/self.scaling_factor*(self.max_speed-self.min_speed)
        if target_speed <= self.min_speed:
            target_speed = self.min_speed

        # print("yaw값이 조향에 영향을 미친 정도: ", -np.degrees(yaw_term))
        # print("cte값이 조향에 영향을 미친 정도: ", -np.degrees(cte_term))

        print("목표 조향: ", target_steer)
        print("목표 속도: ", target_speed)
        
        print("#"*30)
        
        print("")
        
        marker_for_check = self.make_marker([map_x_for_long, map_y_for_long])
        self.check_long_point_pub.publish(marker_for_check)
        
        # time plot
        # self.time_plot.update([cte])
        return target_steer, target_speed    

    def make_marker(self, pos):
        marker = Marker()
        marker.action = marker.ADD
        marker.type = marker.SPHERE
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        # marker.lifetime = rospy.Duration(0.1)
        marker.id = int(1000000)
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = 0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0
        marker.color.r = 255
        marker.color.g = 0
        marker.color.b = 0
        marker.pose.orientation.w = 1.0
        return marker
    
if __name__ == '__main__':
    path_tracking = PathTracking()
    rospy.spin()
    
    # path_tracking.time_plot.draw()
    # path_tracking.time_plot_2.draw()