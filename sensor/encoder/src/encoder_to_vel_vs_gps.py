#!/usr/bin/env python3
import numpy as np
from collections import deque

import rospy
import serial
import time
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistWithCovarianceStamped

import matplotlib.pyplot as plt

OVERFLOW = 4294967296
OVERFLOW_2 = OVERFLOW//2

class DataPlot():
    def __init__(self, data_name, title='Basic'):
        self.data = []
        self.timestamp = []
        self.data_name = data_name
        
        self.title = title
        
        self.first_time = None
        
        self.color = ['g','b','r', 'c', 'y', 'm', 'k']
        return
    
    def update(self, data):
        if self.first_time is None:
            self.first_time = time.time()
            
        self.data.append(data)
        self.timestamp.append(time.time() - self.first_time)
        
        return
    
    def draw(self):
        plt.figure(figsize=(10, 8))  # 그래프 크기 설정
        
        datum = np.array(self.data).T
        
        for i, data_name in enumerate(self.data_name):
            plt.plot(self.timestamp, datum[i], '.-', label=data_name, color=self.color[i])
            plt.legend()
        
        plt.xlabel('Time')
        plt.ylabel('Data')
        plt.title(self.title)
        
        plt.tight_layout()  # 그래프 간격 조정
        plt.show()
        return
    
    
class EncoderVelPub():
    def __init__(self):
        rospy.init_node('encoder_velocity')
        
        self.speed = Float32()
        
        self.prev_enc = 0
        self.prev_time = None
        
        self.enc_speed_data = deque(maxlen=5)
        
        # data plot
        self.data_plot = DataPlot(data_name=['gps_speed', 'enc_speed', 'enc_speed_filtered'],\
                                  title='Compare enc_speed with gps_speed')
        
        
        self.ser = serial.serial_for_url('/dev/ttyACM0', baudrate=9600, timeout=0.01)
        
        self.speed_pub = rospy.Publisher('/cur_speed', Float32, queue_size=10)
        self.gps_speed_pub = rospy.Subscriber('/ublox_gps/fix_velocity', TwistWithCovarianceStamped, self.callback_gps_speed)
        return
    
    def callback_gps_speed(self, velocity_msg):
        self.gps_speed = np.sqrt(velocity_msg.twist.twist.linear.x **2 + velocity_msg.twist.twist.linear.y**2)
        
        
    def get_value(self):
        enc_byte = self.ser.readline()
        
        enc_decoded = enc_byte.decode()
        
        if self.prev_time is None:
            self.prev_time = time.time()
            return
        
        cur_time = time.time()
        dt = cur_time - self.prev_time
        if dt < 0.05: # 20Hz
            return
        
        if not enc_decoded: # 빈 값이면
            print('속도: 0 / error 1')
            vel = 0

        else:
            try:
                enc_first, enc_second = enc_decoded.split(',')
                # print(enc_first, enc_second)
                
                # 엔코더 tick value
                cur_enc = (int(enc_first) - int(enc_second)) / 2
                print('first', enc_first, 'second',enc_second)
                
                # 현재 tick - 이전 tick
                delta_enc = self.normalize_diff(cur_enc - self.prev_enc)
                print('delta', delta_enc)
                
                # 속도 값으로 변환
                vel = delta_enc * 0.06283185307 * 0.265 /dt
                print('현재 속도:', vel)
                
                self.prev_enc = cur_enc
                
            except:
                print('속도: 0 / error 2')
                vel = 0
        
        
        # 평균값 필터
        self.enc_speed_data.append(vel)
        vel_filtered = np.mean(self.enc_speed_data)
            
        self.speed_pub.publish(vel_filtered)

        self.data_plot.update([self.gps_speed, vel, vel_filtered])
        
        self.prev_time = cur_time
        # print(enc_byte, ':', enc_decoded) 
        return
    
    def normalize_diff(self, diff):
        if diff > OVERFLOW_2:
            diff -= OVERFLOW
                
        elif diff < -OVERFLOW_2:
            diff += OVERFLOW
        
        return diff
    
def main():
    node = EncoderVelPub()
    try:
        while not rospy.is_shutdown():
            node.get_value()

        node.data_plot.draw()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt')

if __name__ == '__main__':
    main()