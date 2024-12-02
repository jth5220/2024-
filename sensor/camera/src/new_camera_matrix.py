#!/usr/bin/env python3
import numpy as np
import cv2

mtx = np.array([[474.2048 , 0  , 319.6910 ], 
                        [0  , 476.7333 , 204.9360    ], 
                        [  0.        ,   0.        ,   1.        ]])
dist = np.array([[-0.3557 , 0.1115, 0. , 0.0 , 0.0 ]])  
# mtx = np.array([[635.762102 , 0  , 310.528472 ], 
#                 [0  , 635.872939 ,  205.602578   ], 
#                 [  0.        ,   0.        ,   1.        ]])
# dist = np.array([[-0.38 , 0.11 , 0.001682 , 0.003889 , 0.0 ]])  


new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 360), 0, (640,360))
print(new_camera_mtx)