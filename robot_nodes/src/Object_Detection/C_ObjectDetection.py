#!/usr/bin/env python2.7
from math import *
import numpy as np 
import cv2
from cv2 import aruco
from cv_bridge import CvBridge, CvBridgeError 

import rospy 
class Publish(object):
    def __init__(self,pub,msg_type):
        self._pub = pub
        self._msg = msg_type()
    def P_data(self,msg):
        self._msg = msg
        self._pub.publish(self._msg)

        
class Subscribe(object):
    def __init__(self,topicName,msgType,size):
        self.size = size
        self.data = np.zeros(self.size)
        self.msg = 0
        rospy.Subscriber(topicName,msgType,self.S_msg_callback)  
    def S_msg_callback(self,msg):
        self.msg = msg
    def S_data_callback(self,msg_parameter):
        self.data = np.reshape(np.asarray(msg_parameter),self.size,order='C')
        
        
class Operations(object):
    
    def __init__(self,rgb_frame,depth_frame):
        self.rgb_frame = rgb_frame
        self.depth_frame = depth_frame

        
    def ImageFeatures_IBVS(self,imageFeaturesID):
       
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(self.rgb_frame,aruco_dict,parameters=parameters)
        
        if ids is not None:
            
            ids=list(ids.flatten(order='C'))
            imageFeatureDictionary = {}
            for i in range(len(ids)):
                imageFeatureDictionary[ids[i]] = corners[i]
            imageFeatures = sorted(imageFeatureDictionary.iteritems()) 
            # print(imageFeatures)  
            for j in range(len(ids)):
                imageFeaturesID[imageFeatures[j][0],:,:2] = np.reshape(imageFeatures[j][1],(4,2))
            for j in range(len(imageFeaturesID)):
                for k in range(4):
                    imageFeaturesID[j,k,2] = self.depth_frame[int(imageFeaturesID[j,k,1]),int(imageFeaturesID[j,k,0])]
        # imageFeaturesID = imageFeaturesID[1:,:,:]
        return imageFeaturesID

    def Draw_ImageFeatures_IBVS(self,imageFeaturesID,fix_imageFeatures):
        color = [(255,0,0,0),(0,255,0,0),(0,0,255,0),(200,0,180,0)]
        for j in range(2):
            cv2.putText(self.rgb_frame,str(j),tuple((imageFeaturesID[j,0,:2]).astype('int32')),cv2.FONT_HERSHEY_SIMPLEX ,1,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(self.rgb_frame,str(j),tuple((fix_imageFeatures[j,0,:2]).astype('int32')),cv2.FONT_HERSHEY_SIMPLEX ,1,(250,150,0),2,cv2.LINE_AA)
            
            for k in range(4):
                cv2.circle(self.rgb_frame, tuple((imageFeaturesID[j,k,:2]).astype('int32')), 5, color[k], -1)
                cv2.drawMarker(self.rgb_frame,tuple((fix_imageFeatures[j,k,:2]).astype('int32')), color[k],cv2.MARKER_SQUARE,8,2,8)

        
