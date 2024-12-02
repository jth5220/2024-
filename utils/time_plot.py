#!/usr/bin/env python3
import time
import matplotlib.pyplot as plt
import numpy as np
import csv

class TimePlot():
    def __init__(self, data_name, title='Basic'):
        self.data = []
        self.timestamp = []
        self.data_name = data_name
        
        self.title = title
        
        self.first_time = None
        
        self.color = ['g','b','r', 'c', 'y', 'm', 'k']
        return
    
    def set_color(self, color_list):
        self.color = color_list
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
    
    def save_csv(self, filename):
        """Saves the timestamps and data to a CSV file."""
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', self.data_name])
            writer.writerows(zip(self.timestamp, self.data))
        print(f"Data saved to {filename}")

    