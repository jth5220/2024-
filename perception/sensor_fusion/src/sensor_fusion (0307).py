#!/usr/bin/env python3

import numpy as np
import time

from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks

from sensor_fusion_handler import *

# ROS
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PoseArray
import message_filters

# 올해 파라미터
# self.intrinsic_left = np.array([[381.18310547,   0.,         319.02992935, 0.],
#                             [0. ,        448.0055542, 207.31755552, 0.],
#                             [0.000000, 0.000000, 1.000000, 0.]])

# self.extrinsic_left = self.rtlc(alpha = np.radians(38.0),
#                                 beta = np.radians(31.3),
#                                 gamma = np.radians(4.1), 
#                                 tx = 0.965, ty = -0.23, tz = -0.95)

# self.intrinsic_right = np.array([[378.68261719,   0.,         328.19930137, 0.],
#                     [0. ,        443.68624878, 153.57524293, 0.],
#                     [0.000000, 0.000000, 1.000000, 0.]])
# self.extrinsic_right = self.rtlc(alpha = np.radians(30.8),
#                                     beta = np.radians(-33.2),
#                                     gamma = np.radians(2.3), 
#                                     tx = 0.965, ty = 0.218, tz = -0.95)

# 작년 ROS백 기준
# self.intrinsic_left = np.array([[381.18310547,   0.,         319.02992935, 0.],
#                             [0. ,        448.0055542, 207.31755552, 0.],
#                             [0.000000, 0.000000, 1.000000, 0.]])
        
#         self.extrinsic_left = self.rtlc(alpha = np.radians(34.4),
#                                         beta = np.radians(35.4),
#                                         gamma = np.radians(-4.0), 
#                                         tx = 0.965, ty = -0.23, tz = -0.95)
        
#         self.intrinsic_right = np.array([[378.68261719,   0.,         328.19930137, 0.],
#                             [0. ,        443.68624878, 153.57524293, 0.],
#                             [0.000000, 0.000000, 1.000000, 0.]])
#         self.extrinsic_right = self.rtlc(alpha = np.radians(32.6),
#                                          beta = np.radians(-33.3),
#                                          gamma = np.radians(-0.2), 
#                                          tx = 0.965, ty = 0.218, tz = -0.95)
        
class SensorFusion():
    def __init__(self):
        self.bridge = CvBridge()
        self.intrinsic_left = np.array([[381.18310547,   0.,         319.02992935, 0.],
                            [0. ,        448.0055542, 207.31755552, 0.],
                            [0.000000, 0.000000, 1.000000, 0.]])

        self.extrinsic_left = self.rtlc(alpha = np.radians(38.0),
                                        beta = np.radians(31.3),
                                        gamma = np.radians(4.1), 
                                        tx = 0.965, ty = -0.23, tz = -0.95)

        self.intrinsic_right = np.array([[378.68261719,   0.,         328.19930137, 0.],
                            [0. ,        443.68624878, 153.57524293, 0.],
                            [0.000000, 0.000000, 1.000000, 0.]])
        self.extrinsic_right = self.rtlc(alpha = np.radians(30.8),
                                            beta = np.radians(-33.2),
                                            gamma = np.radians(2.3), 
                                            tx = 0.965, ty = 0.218, tz = -0.95)
                # ROS
        rospy.init_node('sensor_fusion', anonymous=True)
        self.cluster_sub = message_filters.Subscriber('/adaptive_clustering/markers', MarkerArray)
        self.bbox_sub = message_filters.Subscriber("/bounding_boxes/tracked", PoseArray)
        
        self.sync = message_filters.ApproximateTimeSynchronizer([self.cluster_sub, self.bbox_sub], queue_size=10, slop=0.5, allow_headerless=True)
        self.sync.registerCallback(self.callback_fusion)
        
        self.fusion_pub = rospy.Publisher("/sensor_fusion_rubber_cones", MarkerArray, queue_size=10)
        self.clusters_2d_pub = rospy.Publisher("/clusters_2d", PoseArray, queue_size=10)
                
        print("\033[1;33m Starting camera and 3D LiDAR sensor fusion. \033[0m")
        return
    
    def callback_fusion(self, cluster_msg, bbox_msg):
        first_time = time.perf_counter()

        # Clustering points to np array
        clusters = cluster_for_fusion(cluster_msg)
        
        # 2D bounding boxes
        left_bboxes, left_bboxes_label, right_bboxes, right_bboxes_label = bounding_boxes(bbox_msg)
        
        # 3D BBOX to Pixel Frame
        clusters_2d_left, clusters_3d_left = projection_3d_to_2d(clusters, self.intrinsic_left, self.extrinsic_left)
        clusters_2d_right, clusters_3d_right = projection_3d_to_2d(clusters, self.intrinsic_right, self.extrinsic_right)
        
        # Sensor Fusion (Hungarian Algorithm)
        # unmatched_bboxes => 클러스터링으로 안잡힌 경우 발생 => 클러스터링 tracking? => 차라리 sensor fusion 후의 결과를 tracking 하는 편이 낫지 않을까?
        # unmatched_clusters => 웬만하면 거의 other objects인 경우
        matched_left, unmatched_bboxes_left, unmatched_clusters_left = hungarian_match(left_bboxes, clusters_2d_left, distance_threshold=80)
        matched_right, unmatched_bboxes_right, unmatched_clusters_right = hungarian_match(right_bboxes, clusters_2d_right, distance_threshold=80)
        
        # Matching the clusters and bounding boxes
        matched_clusters_3d_left, clusters_labels_left = get_matched_clusters(matched_left, clusters_3d_left, left_bboxes_label)
        matched_clusters_3d_right, clusters_labels_right = get_matched_clusters(matched_right, clusters_3d_right, right_bboxes_label)

        # ROS Publish (Result of sensor fusion)
        fusion_markers = MarkerArray()
        blue_marker = self.make_marker((0.0, 0.0, 1.0))
        yellow_marker = self.make_marker((1.0, 1.0, 0.0))

        label_clusters(matched_clusters_3d_left, clusters_labels_left, blue_marker, yellow_marker)
        label_clusters(matched_clusters_3d_right, clusters_labels_right, blue_marker, yellow_marker)
        
        fusion_markers.markers.extend([blue_marker, yellow_marker])
        self.fusion_pub.publish(fusion_markers)
        
        # ROS Publish (Projected clusters to 2D frame)       
        clusters_2d_right[:,0] = clusters_2d_right[:,0] + 640
        clusters_2d = np.vstack([clusters_2d_left, clusters_2d_right])
        
        clusters_2d_msg = self.make_pose_array(clusters_2d)
        self.clusters_2d_pub.publish(clusters_2d_msg)

        print("소요 시간: {:.5f}".format(time.perf_counter() - first_time))
        return
    
    def rtlc(self, alpha, beta, gamma, tx, ty, tz):              
        Rxa = np.array([[1, 0, 0,0],
                        [0, np.cos(alpha), -np.sin(alpha),0],
                        [0, np.sin(alpha), np.cos(alpha),0],
                        [0,0,0,1]])

        Ryb = np.array([[np.cos(beta), 0, np.sin(beta),0],
                        [0, 1, 0,0],
                        [-np.sin(beta), 0, np.cos(beta),0],
                        [0,0,0,1]])

        Rzg = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                [np.sin(gamma), np.cos(gamma), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
        
        Ry90 = np.array([[np.cos(np.deg2rad(-90)), 0, np.sin(np.deg2rad(-90)),0],
                         [0, 1, 0,0],
                         [-np.sin(np.deg2rad(-90)), 0, np.cos(np.deg2rad(-90)),0],
                         [0,0,0,1]])
                 
        Rx90= np.array([[1, 0, 0,0],
                        [0, np.cos(np.deg2rad(90)), -np.sin(np.deg2rad(90)),0],
                        [0, np.sin(np.deg2rad(90)), np.cos(np.deg2rad(90)),0],
                        [0,0,0,1]])
        
        T = np.array([[1, 0, 0, tx],      
                      [0, 1, 0, ty],                  
                      [0, 0, 1, tz],                       
                      [0, 0, 0, 1]]) 
        
        rtlc = Rzg@Rxa@Ryb@Ry90@Rx90@T
        return rtlc
    
    def make_marker(self, color):
        marker = Marker()
        marker.action = marker.ADD
        marker.type = marker.POINTS
        marker.header.frame_id = "velodyne"
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = rospy.Duration(0.1)
        marker.id = int((color[0] + color[1] + color[2]) * 10000)
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
    def make_pose_array(points):
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'yolo'
        for x, y in points:
            pose = Pose()
            pose.orientation.x = x
            pose.orientation.y = y
            
            pose_array.poses.append(pose)
        return pose_array
    
if __name__ == '__main__':
    sensor_fusion = SensorFusion()
    rospy.spin()