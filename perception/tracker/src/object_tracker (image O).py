#!/usr/bin/env python3
import numpy as np

from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks

import rospy
import message_filters
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import Image


class ObjectTracker():
    def __init__(self):
        self.bridge = CvBridge()
        self.tracker = IOUTracker(max_lost=3, iou_threshold=0.4, min_detection_confidence=0.4, max_detection_confidence=0.7,
                             tracker_output_format='visdrone_challenge')
        
        rospy.init_node('bounding_boxes_tracker', anonymous=True)
        self.bbox_sub = rospy.Subscriber('/bounding_boxes', PoseArray, self.callback_tracker)
        # self.img_sub = message_filters.Subscriber('/yolo_visualization', Image)
        
        # self.img_tracked_pub = rospy.Publisher("/image_tracked", Image, queue_size=10)
        self.bbox_tracked_pub = rospy.Publisher('/bounding_boxes/tracked', PoseArray, queue_size=10)
        
        # self.sync = message_filters.ApproximateTimeSynchronizer([self.bbox_sub, self.img_sub], queue_size=100, slop=0.5, allow_headerless=True)
        # self.sync.registerCallback(self.callback_tracker)
        return
    
    def callback_tracker(self, bbox_msg):
        # Image
        # img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        
        bboxes = []
        confidences = []
        labels = []
        for bbox in bbox_msg.poses:
            bboxes.append([int(bbox.orientation.x), int(bbox.orientation.y), int(bbox.orientation.z-bbox.orientation.x), int(bbox.orientation.w-bbox.orientation.y)])
            confidences.append(bbox.position.y)
            labels.append(bbox.position.x)
        
        tracks = self.tracker.update(bboxes, confidences, labels)
        # updated_image = draw_tracks(img, tracks)
        
        poses = PoseArray()
        poses.header.stamp = rospy.Time.now()
        poses.header.frame_id = 'yolo'
        for _, _, x_min, y_min, width, height, _, lbl, _, _ in tracks:
            
                
        # updated_img_msg = self.bridge.cv2_to_imgmsg(updated_image, encoding="bgr8")
        # updated_img_msg.header.stamp = rospy.Time.now()
        # self.img_tracked_pub.publish(updated_img_msg)
        
        return
    

if __name__ == '__main__':
    tracker = ObjectTracker()
    rospy.spin()