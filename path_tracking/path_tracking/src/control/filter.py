from collections import deque

class average_filter():
    def __init__(self, window_size=3):
        self.window = deque(maxlen = window_size)
        return
    
    def update(self, value):
        self.window.append(value)
        
        value_filtered = sum(self.window) / len(self.window)
        return value_filtered
    
class low_pass_filter():
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value_prev = 0
        return
    
    def update(self, value):
        value_filtered = (1-self.alpha)*self.value_prev + self.alpha*value
        self.value_prev = value_filtered
        return value_filtered

if __name__ == '__main__':
    avg_fltr = average_filter(window_size=5)
    lpf = low_pass_filter(alpha=0.2)
    
    datum = [0, 5, 2, 3, 4, 5, 10, 6, 7, 8 , 9, 15, 11, 12]
    
    for data in datum:
        data_filtered = avg_fltr.update(data)
        print(data_filtered)
        
        data_filtered = lpf.update(data)
        print(data_filtered)
        print()