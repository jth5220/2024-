#!/usr/bin/env python3
import numpy as np

from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker

import rospy
from geometry_msgs.msg import Pose, PoseArray

import os

class BboxTracker():
    def __init__(self):
        self.tracker = IOUTracker(max_lost=5, iou_threshold=0.4, min_detection_confidence=0.3, max_detection_confidence=1.0,
                             tracker_output_format='visdrone_challenge')
        
        rospy.init_node('bounding_boxes_tracker', anonymous=True)
        self.bbox_sub = rospy.Subscriber('/bounding_boxes', PoseArray, self.callback_tracker)
        
        self.bbox_tracked_pub = rospy.Publisher('/bounding_boxes/tracked', PoseArray, queue_size=10)
        
        os.system('clear')
        print("\033[1;33m Tracking YoLo bounding boxes. \033[0m")
        return
    
    def callback_tracker(self, bbox_msg):
        bboxes = []
        confidences = []
        labels = []
        for bbox in bbox_msg.poses:
            bboxes.append([int(bbox.orientation.x), int(bbox.orientation.y), int(bbox.orientation.z-bbox.orientation.x), int(bbox.orientation.w-bbox.orientation.y)])
            # bboxes.append([bbox.orientation.x, bbox.orientation.y, bbox.orientation.z-bbox.orientation.x, bbox.orientation.w-bbox.orientation.y])
            confidences.append(bbox.position.y)
            labels.append(bbox.position.x)
        
        tracks = self.tracker.update(bboxes, confidences, labels)
        
        bounding_boxes = PoseArray()
        bounding_boxes.header.stamp = rospy.Time.now()
        bounding_boxes.header.frame_id = 'yolo'
        for _, _, x_min, y_min, width, height, _, lbl, _, _ in tracks:
            bbox = Pose()
            bbox.position.x = lbl # 0 blue or 1 yellow
            
            bbox.orientation.x = x_min #xmin
            bbox.orientation.y = y_min #ymin
            bbox.orientation.z = x_min + width #xmax
            bbox.orientation.w = y_min + height #ymax

            bounding_boxes.poses.append(bbox)
            
        self.bbox_tracked_pub.publish(bounding_boxes)

        return

if __name__ == '__main__':
    tracker = BboxTracker()
    rospy.spin()