#!/usr/bin/env python3
import rospy
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters

import os
CAMERA_HEIGHT = 360
CAMERA_WIDTH = 1280

class CombineCamera():
    def __init__(self):
        self.bridge = CvBridge()
        
        # ROS
        rospy.init_node('combine_camera')

        self.left_sub = message_filters.Subscriber("/image_left", Image)
        self.right_sub = message_filters.Subscriber("/image_right", Image)
        # self.left_sub = message_filters.Subscriber("/left_image", Image)
        # self.right_sub = message_filters.Subscriber("/right_image", Image)
        # self.left_sub = message_filters.Subscriber("/camera_left/camera1/usb_cam/image_raw", Image)
        # self.right_sub = message_filters.Subscriber("/camera_right/camera2/usb_cam/image_raw", Image)
        
        self.sync = message_filters.ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=5, slop=0.5, allow_headerless=True)
        self.sync.registerCallback(self.callback_combine)
        
        self.combined_pub = rospy.Publisher("/image_combined", Image, queue_size = 10)
        
        os.system('clear')
        print("\033[1;33m Combining left and right images. \033[0m")
        return
    
    def callback_combine(self, left_msg, right_msg):
        left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        
        combined_img = np.hstack([left_img, right_img])
        image_message = self.bridge.cv2_to_imgmsg(combined_img, encoding="bgr8")
        image_message.header.stamp = rospy.Time.now()
        self.combined_pub.publish(image_message)
        
        return

if __name__ == '__main__':
    try:
        combine_cam = CombineCamera()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass