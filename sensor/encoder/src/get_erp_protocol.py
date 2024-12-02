#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray

import serial
import time

OVERFLOW = 4294967296
OVERFLOW_2 = OVERFLOW//2
print(OVERFLOW, OVERFLOW_2)

class GetERPProtocol():
    def __init__(self):
        self.protocols = []
        self.buffer = []
        self.is_protocol_on = False
        
        self.encoder = 0
        self.steer = 0
        
        self.prev_encoder = None
        self.prev_time = None
        self.ser = serial.serial_for_url("/dev/ttyERP", baudrate=115200, timeout=0.01)
        
        # ROS
        rospy.init_node('erp_protocol')
        self.protocol_pub = rospy.Publisher('/delta_dist_and_steer', Float32MultiArray, queue_size=10)
        
        return
    
    def get_value(self):
        s = self.ser.read().hex()
        
        if s == '53':
            self.buffer.append(s)
            self.is_protocol_on = True
            return
        
        if self.is_protocol_on:
            self.buffer.append(s)
            
        if self.is_protocol_on and s == '0a':
            if '' in self.buffer or len(self.buffer) < 18:
                self.reset_buffer()
                return

            # encoder
            encoder_val_1 = int(self.buffer[11], 16)
            encoder_val_2 = int(self.buffer[12], 16) << 8
            encoder_val_3 = int(self.buffer[13], 16) << 16
            encoder_val_4 = int(self.buffer[14], 16) << 24
            self.encoder = (encoder_val_1 | encoder_val_2 | encoder_val_3 | encoder_val_4)

            # steering
            steer_1 = int(self.buffer[8], 16)
            steer_2 = int(self.buffer[9], 16) << 8
            
            self.steer = (steer_1 | steer_2)
            if (self.steer > 30000):
              self.steer = (self.steer - 65536)/71
            else:
              self.steer = self.steer/71
              
            self.steer = -self.steer
            self.steer = np.radians(self.steer)
            
            # print(self.encoder, self.steer)

            self.reset_buffer()
        return
    
    def calc_speed(self):
        if self.prev_time is None:
            self.prev_time = time.time()
            return
        
        cur_time = time.time()
        
        dt = cur_time - self.prev_time
        
        if dt >= 0.05:
            if self.prev_encoder is None:
                self.prev_encoder = self.encoder
            else:
                diff = self.encoder - self.prev_encoder
                diff = self.normalize_diff(diff)
                
                delta_dist = diff * 0.06283185307 * 0.265
                print('이동 거리:', delta_dist)
                print('조향(도):', self.steer)
                print()
                self.prev_encoder = self.encoder

                delta_dist_and_steer_msg = Float32MultiArray([delta_dist, self.steer])
                self.protocol_pub.publish(delta_dist_and_steer_msg)
                
            self.prev_time = cur_time
        
    def reset_buffer(self):
        self.buffer = []
        self.is_protocol_on = False
    
    def normalize_diff(self, diff):
        if diff > OVERFLOW_2:
            diff -= OVERFLOW
                
        elif diff < -OVERFLOW_2:
            diff += OVERFLOW
        
        return diff
    
def main():
    node = GetERPProtocol()
    
    try:
        while not rospy.is_shutdown():
            node.get_value()
            node.calc_speed()
            

    except KeyboardInterrupt:
        with open('erp_protocol_received', 'w') as file:
            for line in node.protocols:
                file.write(str(line))
                file.write('\n')
            print("saved the erp protocol record.")
        
if __name__ == '__main__':
    main()