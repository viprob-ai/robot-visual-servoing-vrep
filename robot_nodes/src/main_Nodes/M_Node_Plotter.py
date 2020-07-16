#!/usr/bin/env python2.7
from math import *
import numpy as np 
import cv2
from cv2 import aruco
from cv_bridge import CvBridge, CvBridgeError 

import rospy 

#### Class Import
import C_ObjectDetection as cod
import K_Kinematics as ks
import VS_VisualServoing as vs
#### Message Import
from armtorso_nodes.msg import std_mssg
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

import subprocess
import webbrowser as wb
import sys
import matplotlib.pyplot as plt

from std_msgs.msg import Header

from mpl_toolkits.mplot3d import *
from matplotlib.pylab import rcParams 
from matplotlib.backends.backend_pdf import PdfPages
from math import *
             
def main():

    rospy.init_node('VS_Plotter',anonymous=True)

    featuresError_cb = pls.Subscribe(4,2)
    jointVelocity_cb = pls.Subscribe(1,6) 
    currentFeatures_cb = pls.Subscribe(4,3)
    desiredFeatures_cb = pls.Subscribe(4,3)
    transJ6_cb = pls.Subscribe(3,4)
    rospy.Subscriber('/features_error', std_mssg,featuresError_cb.S_error_callback)
    rospy.Subscriber('/jointRates/coppeliaSim', Float32MultiArray,jointVelocity_cb.S_data_callback)
    rospy.Subscriber('/urKinect_currentFeaturesPoints', std_mssg,currentFeatures_cb.S_data_callback)
    rospy.Subscriber('/urKinect_desiredFeaturePoints', std_mssg,desiredFeatures_cb.S_data_callback)
    rospy.Subscriber('/coppeliaSim/transJ6',Float32MultiArray, transJ6_cb.S_data_callback)
    a = str(sys.argv[1])
    b = int(sys.argv[2])

    pdf = PdfPages('./Workspaces/ros_ws/eye_to_hand_vs_ws/logs/Saved/PLOTS_FORMAT/PDF/Fig_Output_'+str(b)+'_'+a+'.pdf')
    while not rospy.is_shutdown() :


        transJ6 = transJ6_cb.data
        transJ6 = np.append(transJ6,[[0,0,0,1]],axis = 0)
        print(featuresError_cb.error)
        # plotter = pls.Plotter(featuresError_cb.error,featuresError_cb.error_timestamp)
        # plotter.Plot_data(jointVelocity_cb.data,currentFeatures_cb.data,transJ6_cb.data)
    
    plot_show()
        

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
