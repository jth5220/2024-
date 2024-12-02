#!/usr/bin/env python3
import time
import numpy as np
import math
from CurvesGenerator.dubins_path import calc_dubins_path
from CurvesGenerator.bezier_path import calc_4points_bezier_path

import rospy
import tf
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, Pose
from nav_msgs.msg import Path

from delaunay_path_planning import DelaunayPathPlanning

# color = [(255,255,255), (255,255,0), (255,0,255), (0,255,255), (0,0,255), (0,255,0), (255,0,0), (255,255,122), (255,122,255), (122,255,255), (255,122,122), (122,255,122), (122,122,255)]

class LocalPathPlanning():
    def __init__(self):
        self.path_planner = DelaunayPathPlanning()
        
        # ROS
        rospy.init_node('path_planning', anonymous=True)
        self.cones_sub = rospy.Subscriber("/sensor_fusion/tracked_rubber_cones", MarkerArray, self.callback_cones)
        self.midpoints_pub = rospy.Publisher("/midpoints", Marker, queue_size=10)
        self.path_pub = rospy.Publisher("/local_path", Path, queue_size=10)
        return
    
    def callback_cones(self, cones_msg):
        first_time = time.perf_counter()
        cones, labels = self.get_cones(cones_msg)

        midpoints = self.path_planner.update(cones, labels)
        
        # 주시 거리 L_D 에서 제일 가까운 midpoint 탐색
        L_D = 0 #m
        L_D_2 = 3.0 #m
        closest_midpoint_idx = None
        farthest_midpoint_idx = None
        dist_min = np.inf
        x_ = 0.0
        for idx, midpoint in enumerate(midpoints):
            dist = np.sqrt(midpoint[0]**2+midpoint[1]**2)
            if abs(dist - L_D) < dist_min:
                dist_min = dist
                closest_midpoint_idx = idx
            if midpoint[0] > x_:
                x_ = midpoint[0] 
                farthest_midpoint_idx = idx
                
        #=== dubins_path ===#
        # R_MIN = 1/ (1.7 / np.sin(np.radians(27)))
        # g_x, g_y = midpoints[closest_midpoint_idx]
        # try:
        #     vector = midpoints[closest_midpoint_idx] - midpoints[closest_midpoint_idx-1]
        # except:
        #     vector = midpoints[closest_midpoint_idx+1] - midpoints[closest_midpoint_idx]
        # g_yaw = np.arctan2(vector[1] / vector[0],1)
        
        # g_x, g_y = midpoints[-1]
        # g_vector = midpoints[-1] - midpoints[-2]
        # g_yaw = math.atan2(g_vector[1], g_vector[0])
        
        # s_x, s_y = midpoints[closest_midpoint_idx]
        # s_vector = midpoints[closest_midpoint_idx+1] - midpoints[closest_midpoint_idx]
        # s_yaw = math.atan2(s_vector[1], s_vector[0])
        
        # path_x, path_y, path_yaw = self.get_dubins_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, R_MIN)
        
        #=== Bezier path ===#
        s_x, s_y = midpoints[closest_midpoint_idx]
        v_s = midpoints[closest_midpoint_idx+1]-midpoints[closest_midpoint_idx]
        s_yaw = np.arctan2(v_s[1]/v_s[0], 1)
        
        g_x, g_y = midpoints[farthest_midpoint_idx]
        v_g = midpoints[farthest_midpoint_idx]-midpoints[farthest_midpoint_idx-1]
        g_yaw = np.arctan2(v_g[1]/v_g[0], 1)
        
        # path_x, path_y, path_yaw = self.get_bezier_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, offset=3.0)
        path_x, path_y, path_yaw = self.get_bspline_path(midpoints)
        
        #=== Bspline path ===#
        # path_x, path_y, path_yaw = self.get_bspline_path(midpoints)
        
        # ROS Publish - midpoints
        midpoints_marker = self.make_marker((0.8, 0., 0.8))
        for midpoint in midpoints:
            point = Point()
            point.x, point.y = midpoint[0], midpoint[1]
            midpoints_marker.points.append(point)
        self.midpoints_pub.publish(midpoints_marker)
        
        # ROS Publish - path
        path_msg = self.get_path_msg(path_x, path_y, path_yaw)
            
        self.path_pub.publish(path_msg)
        
        print("Path Planning 소요 시간: ", time.perf_counter() - first_time)
        return   
    
    @staticmethod
    def get_cones(cones_msg):
        cones = []
        labels = []
        for cone_msg in cones_msg.markers:
            color = (cone_msg.color.r, cone_msg.color.g, cone_msg.color.b)
            if color == (0., 0., 1.):
                points = [(point.x, point.y) for point in cone_msg.points]
                cones += points
                labels += [0] * len(points)
            elif color == (1., 1., 0.):
                points = [(point.x, point.y) for point in cone_msg.points]
                cones += points
                labels += [1] * len(points)
            # else:
            #     labels += [None] * len(points)
        return np.array(cones), labels
    
    @staticmethod
    def make_marker(color):
        marker = Marker()
        marker.action = marker.ADD
        marker.type = marker.POINTS
        marker.header.frame_id = "velodyne"
        marker.header.stamp = rospy.Time.now()
        # marker.lifetime = rospy.Duration(0.1)
        marker.id = 50000
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.pose.orientation.w = 1.0
        return marker
    
    @staticmethod
    def get_path_msg(path_x, path_y, path_yaw):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "velodyne"
        
        for x,y,yaw in zip(path_x, path_y, path_yaw):
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "velodyne"
            pose.pose.position.x = x
            pose.pose.position.y = y
            quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
            # quaternion = tf.TransformerROS.transformQuaternion(0,0,yaw)
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]
            path_msg.poses.append(pose)
        return path_msg
    
    @staticmethod
    def get_dubins_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, max_c):
        path_x, path_y, yaw = [], [], []
        path_i = calc_dubins_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, max_c)

        for x, y, iyaw in zip(path_i.x, path_i.y, path_i.yaw):
            path_x.append(x)
            path_y.append(y)
            yaw.append(iyaw)
            
        return path_x, path_y, yaw
    
    @staticmethod
    def get_bezier_path(sx, sy, syaw, gx, gy, gyaw, offset):
        path, control_points = calc_4points_bezier_path(sx, sy, syaw, gx, gy, gyaw, offset=3.0)
        
        path_x, path_y, path_yaw = [], [], []
        for i in range(len(path)-1):
            x, y = path[i]
            x_n, y_n = path[i+1]
            slope = (y_n-y)/(x_n-x)
            yaw = np.arctan2(slope, 1)
            
            path_x.append(x)
            path_y.append(y)
            path_yaw.append(yaw)
        return path_x, path_y, path_yaw
    
    @staticmethod
    def get_bspline_path(midpoints):
        from scipy import interpolate
        
        x=midpoints[:,0]
        y=midpoints[:,1]
        
        l=len(x)
        Order = 3

        t=np.linspace(0,1,l-(Order-1),endpoint=True)
        t=np.append(np.zeros(Order),t)
        t=np.append(t,np.zeros(Order)+1)
        
        tck=[t,[x,y],Order]
        u3=np.linspace(0,1,(max(l*2,70)),endpoint=True)
        out = np.array(interpolate.splev(u3,tck)) # xs: out[0] / ys: out[1]
        out = out.T
        
        path_x, path_y, path_yaw = [], [], []
        for i in range(len(out)-1):
            x, y = out[i]
            x_n, y_n = out[i+1]
            slope = (y_n-y)/(x_n-x)
            yaw = np.arctan2(slope, 1)
            
            path_x.append(x)
            path_y.append(y)
            path_yaw.append(yaw)
            
        return path_x, path_y, path_yaw
    
if __name__ == '__main__':
    path_planning = LocalPathPlanning()
    rospy.spin()