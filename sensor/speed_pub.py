"""
이것은 rosbag의 GPS velocity 토픽의 time을 동기화해주기 위한 코드이다.
"""
#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import TwistWithCovarianceStamped

class SpeedPub():
    def __init__(self):
        # ROS
        rospy.init_node('speed_test_pub', anonymous=True)
        
        self.speed_sub = rospy.Subscriber('/ublox_gps/fix_velocity', TwistWithCovarianceStamped, self.callback_speed)
        self.speed_pub = rospy.Publisher('/speed', TwistWithCovarianceStamped, queue_size=10)
        
        return
    
    def callback_speed(self, msg):
        msg.header.stamp = rospy.Time.now()
        self.speed_pub.publish(msg)
        
        return
    
if __name__ == '__main__':
    speed = SpeedPub()
    rospy.spin()