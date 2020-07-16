#!/usr/bin/env python2.7
from math import *
import numpy as np 
import cv2
from cv2 import aruco
from cv_bridge import CvBridge, CvBridgeError 

import rospy 

#### Class Import
import Object_Detection.C_ObjectDetection as cod
#### Message Import
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


def main():
########################################################################################
#### -- Initialize Node --
    rospy.init_node('CoppeliaSim_ObjectDetection')  
########################################################################################
#### -- Subscribing --
    rgb_frame_cb = cod.Subscribe('/coppeliaSim/image/rgb',Image,(480,640,3))
    depth_frame_cb = cod.Subscribe('/coppeliaSim/image/depth',Float32MultiArray,(480,640))

    current_pbvs_cb = cod.Subscribe('/coppeliaSim/c_M_O1',Float32MultiArray,(4,4))
    desired_pbvs_cb = cod.Subscribe('/coppeliaSim/c_M_O2',Float32MultiArray,(4,4))
########################################################################################
#### -- Publishing --
    pub_current_features = rospy.Publisher('/visionSensor_currentFeatures',Float32MultiArray,queue_size=10)
    pub_desired_features = rospy.Publisher('/visionSensor_desiredFeatures',Float32MultiArray,queue_size=10)  
    pub_current_features_pbvs = rospy.Publisher('/poseEst_currentFeatures',Float32MultiArray,queue_size=10)
    pub_desired_features_pbvs = rospy.Publisher('/poseEst_desiredFeatures',Float32MultiArray,queue_size=10)  
    rospy.sleep(2)
    n = 12

    bridge = CvBridge()
    imageFeaturesID = np.zeros([n+1,4,3])
    fix_imageFeature = []
    fix_feature = np.zeros([2,4,3])

    while not rospy.is_shutdown():
        key = cv2.waitKey(1) & 0xFF
        
        rgb_frame_cb.S_data_callback(bridge.imgmsg_to_cv2(rgb_frame_cb.msg, 'bgr8'))
        rgb_frame = rgb_frame_cb.data
        # cv2.imshow('rgb_Image',rgb_frame)
        depth_frame_cb.S_data_callback(depth_frame_cb.msg.data)
        depth_frame = depth_frame_cb.data

        # current_pbvs_cb.S_data_callback(current_pbvs_cb.msg.data)
        # current_pbvs_features = current_pbvs_cb.data

        # desired_pbvs_cb.S_data_callback(desired_pbvs_cb.msg.data)
        # desired_pbvs_features = desired_pbvs_cb.data

        #### --Class_instance_method
        imageFeatures_cb = cod.Operations(rgb_frame,depth_frame)
        imageFeaturesID = imageFeatures_cb.ImageFeatures_IBVS(imageFeaturesID)
        
        # print(imageFeaturesID)
        imageFeaturesID_current = np.append([imageFeaturesID[1,:,:]],[imageFeaturesID[11,:,:]],axis = 0)
        imageFeaturesID_desired = np.append([imageFeaturesID[2,:,:]],[imageFeaturesID[12,:,:]],axis = 0)
        # print(imageFeaturesID_current)
        #### --Class_instance_pub
        desired_features = cod.Publish(pub_desired_features,Float32MultiArray)

        
        if key == ord('s'):
            fix_imageFeature = imageFeaturesID_desired.flatten()
            fix_feature = np.reshape(fix_imageFeature,(2,4,3))
            

        imageFeatures_cb.Draw_ImageFeatures_IBVS(imageFeaturesID_current,fix_feature)
        # np.append(fix_imageFeature,imageFeaturesID[1,:,:].flatten())
        
        desired_features._msg.data = fix_imageFeature
        desired_features.P_data(desired_features._msg)

        # desired_features_pbvs = cod.Publish(pub_desired_features_pbvs,Float32MultiArray)
        # desired_features_pbvs._msg.data = O2
        # desired_features_pbvs.P_data(desired_features_pbvs._msg)


        current_features = cod.Publish(pub_current_features,Float32MultiArray)
        current_features._msg.data = imageFeaturesID_current.flatten()
        current_features.P_data(current_features._msg)


        # current_features_pbvs = cod.Publish(pub_current_features_pbvs,Float32MultiArray)
        # current_features_pbvs._msg.data = current_pbvs_features.flatten()
        # current_features_pbvs.P_data(current_features_pbvs._msg)

        
        cv2.imshow('Object_Detection',imageFeatures_cb.rgb_frame)



########################################################################################

if __name__ =='__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

