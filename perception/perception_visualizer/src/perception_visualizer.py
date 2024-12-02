#!/usr/bin/env python3

import rospy
import message_filters
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import Image

from visualizer_handler import *

class PerceptionVisualizer():
    def __init__(self):
        self.bridge = CvBridge()
        
        # ROS
        rospy.init_node('perception_visualizer', anonymous=True)
        self.image_sub = message_filters.Subscriber("/image_combined", Image)
        
        self.cluster_2d_sub = message_filters.Subscriber('/clusters_2d', PoseArray)
        self.bbox_sub = message_filters.Subscriber("/bounding_boxes", PoseArray)
        self.bbox_tracked_sub = message_filters.Subscriber("/bounding_boxes/tracked", PoseArray)
        
        self.img_result_pub = rospy.Publisher('/image_perception_result', Image, queue_size=10)
        
        sub_list = [self.image_sub, self.cluster_2d_sub, self.bbox_sub, self.bbox_tracked_sub]
        self.sync = message_filters.ApproximateTimeSynchronizer(sub_list, queue_size=10, slop=0.5, allow_headerless=True)
        self.sync.registerCallback(self.callback_perception)
        
    def callback_perception(self, img_msg, clusters_2d_msg, bboxes_msg, bboxes_tracked_msg):
        # ROS msg transformation
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        clusters_2d = get_cluster_2d(clusters_2d_msg)
        bboxes, bbox_labels = get_bbox(bboxes_msg)
        bboxes_tracked, bbox_tracked_labels = get_bbox(bboxes_tracked_msg)
        
        # Visualization
        visualize_cluster_2d(clusters_2d, img)
        visualize_bbox(bboxes, bbox_labels, img)
        visualize_bbox_tracked(bboxes_tracked, bbox_tracked_labels, img)
        
        # ROS Publish
        img_for_pub = self.bridge.cv2_to_imgmsg(img, "bgr8")
        img_for_pub.header.stamp = rospy.Time.now()
        self.img_result_pub.publish(img_for_pub)
        return
    
if __name__ == '__main__':
    visualizer = PerceptionVisualizer()
    rospy.spin()