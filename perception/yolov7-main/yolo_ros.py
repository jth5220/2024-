import rospy
import torch
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray

from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

import os
import time

WEIGHTS = 'weights/0819.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.60
IOU_THRES = 0.35
CLASSES = None
AGNOSTIC_NMS = False

device = select_device(DEVICE)
half = device.type != 'cpu'  # half precision only supported on CUDA
print('device:', device)

# Load model
model = attempt_load(WEIGHTS, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(IMG_SIZE, s=stride)  # check img_size
if half:
    model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu' :
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

# ROS topic to image
bridge = CvBridge()

# 이미지 메시지를 받는 콜백 함수
def image_callback(image_msg):
    # main
    with torch.no_grad():
        first_time = time.perf_counter()
        cap = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        
        # Detecting
        rubbers = detect(cap)      
        
        if rubbers is None:
            return
        
        # ROS publish
        poses = PoseArray()
        poses.header.stamp = rospy.Time.now()
        poses.header.frame_id = 'yolo'
        
        for sublist in rubbers:
            pose = Pose()
            # Set the position of the pose
            pose.position.x = sublist[0] # 0 blue or 1 yellow
            pose.position.y = sublist[5] # confidence
            # pose.position.z = 0.
                
            pose.orientation.x = sublist[1] #xmin
            pose.orientation.y = sublist[2] #ymin
            pose.orientation.z = sublist[3] #xmax
            pose.orientation.w = sublist[4] #ymax

            poses.poses.append(pose)
            
        pose_array_pub.publish(poses)
        
        # result = detect(cap)
        # image_message = bridge.cv2_to_imgmsg(result, encoding="bgr8")
        # image_message.header.stamp = rospy.Time.now()
        # cam_pub.publish(image_message)
        
        print("YoLo Detecting 소요 시간: {:.5f}".format(time.perf_counter() - first_time))

# Detect function
def detect(img0):
    # # Load image
    # img0 = frame

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # t0 = time_synchronized()
    pred = model(img, augment=AUGMENT)[0]

    # Apply NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

    # Process detections
    det = pred[0]
    numClasses = len(det)

    # s = ''
    # s += '%gx%g ' % img.shape[2:]  # print string

    if numClasses:
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # # Print results
        # for c in det[:, -1].unique():
        #     n = (det[:, -1] == c).sum()  # detections per class
        #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        rubbers_ =[]
        
        # Write results
        for *xyxy, conf, cls in reversed(det):
            # label = f'{names[int(cls)]} {conf:.2f}'
            # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
            xmin, ymin, xmax, ymax = [int(tensor.item()) for tensor in xyxy]
            
            rubber =[int(cls), xmin, ymin, xmax, ymax, conf]
            rubbers_.append(rubber)

        return rubbers_

    return None

if __name__ == '__main__':
    check_requirements(exclude=('pycocotools', 'thop'))
    
    rospy.init_node('yolo')   
    # 이미지 메시지를 받을 때마다 image_callback 함수 호출
    # cam_pub = rospy.Publisher("/yolo_visualization",Image,queue_size =1)
    pose_array_pub= rospy.Publisher('/bounding_boxes', PoseArray, queue_size=10)
    image_subscriber = rospy.Subscriber("/image_combined", Image, image_callback)
    
    os.system('clear')
    print("\033[1;33m Starting YoLo detection. \033[0m")
    rospy.spin()