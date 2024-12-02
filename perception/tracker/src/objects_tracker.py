#!/usr/bin/env python3
import numpy as np

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import message_filters

from mot.centroid_kf_tracker import CentroidKF_Tracker

class Object3DTracker():
    def __init__(self):
        # self.tracker = CentroidKF_Tracker(max_lost=10, centroid_distance_threshold=1.0,
        #                                 tracker_output_format='visdrone_challenge',
        #                                 process_noise_scale=0.3,
        #                                 measurement_noise_scale=0.1,
        #                                 time_step=0.1)
        
        # ROS             
        rospy.init_node('rubber_cones_tracker', anonymous=True)
        self.cones_sub = message_filters.Subscriber('/sensor_fusion/rubber_cones', MarkerArray)
        # self.cones_unmatched_sub = message_filters.Subscriber('/sensor_fusion/unmatched_rubber_cones', MarkerArray)
        
        sub_list = [self.cones_sub]
        self.sync = message_filters.ApproximateTimeSynchronizer(sub_list, queue_size=3, slop=0.5, allow_headerless=True)
        self.sync.registerCallback(self.callback_rubber_cones)

        self.tracked_pub = rospy.Publisher("/sensor_fusion/tracked_rubber_cones", MarkerArray, queue_size=10)
        
    def callback_rubber_cones(self, matched_cones_msg):
        cones, labels = self.get_cones(matched_cones_msg)
        
        for i in range(len(labels)):
            if labels[i] is None:
                cone_x = cones[i,0]
                cone_y = cones[i,1]
                if -2.0 <= cone_x < 1.5 and -1.8 < cone_y < 1.8:
                    labels[i] = 0 if cone_y < 0 else 1
                # if cones[i,0] > 0: continue
                # labels[i] = 0 if cones[i,1] < 0 else 1
        
        # ROS Publish (Result of sensor fusion)
        tracked_markers = MarkerArray()
        blue_marker = self.make_marker((0.0, 0.0, 1.0))
        yellow_marker = self.make_marker((1.0, 1.0, 0.0))
        white_marker = self.make_marker((1.0, 1.0, 1.0))
        tracked_markers.markers.extend([blue_marker, yellow_marker, white_marker])
        
        self.label_cones(cones, labels, blue_marker, yellow_marker, white_marker)
        
        self.tracked_pub.publish(tracked_markers)
        return
    
    @staticmethod
    def label_cones(cones_track, labels_track, blue_marker, yellow_marker, white_marker):
        for i in range(len(cones_track)):
            point = Point()
            point.x, point.y, point.z = cones_track[i, 0], cones_track[i, 1], -0.2
                       
            if labels_track[i] == 0:
                blue_marker.points.append(point)
            elif labels_track[i] == 1:
                yellow_marker.points.append(point)
            else:
                white_marker.points.append(point)
        return
    
    @staticmethod
    def get_cones(cones_msg):
        cones = []
        labels = []
        for cone_msg in cones_msg.markers:
            points = [(point.x, point.y) for point in cone_msg.points]
            cones += points
            
            color = (cone_msg.color.r, cone_msg.color.g, cone_msg.color.b)
            if color == (0., 0., 1.):
                labels += [0] * len(points)
            elif color == (1., 1., 0.):
                labels += [1] * len(points)
            else:
                labels += [None] * len(points)
        return np.array(cones), labels
    
    def make_marker(self, color):
        marker = Marker()
        marker.action = marker.ADD
        marker.type = marker.POINTS
        marker.header.frame_id = "velodyne"
        marker.header.stamp = rospy.Time.now()
        # marker.lifetime = rospy.Duration(0.1)
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
    
if __name__ == '__main__':
    tracker = Object3DTracker()
    rospy.spin()