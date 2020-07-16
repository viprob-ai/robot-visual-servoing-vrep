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

class Publish(object):
    def __init__(self,pub,msg_type):
        self._pub = pub
        self._msg = msg_type()
    def P_joint_rates(self,data):
        self._msg.data = data
        self._pub.publish(self._msg)

        
class Subscribe(object):
    def __init__(self,row,col):
        self.row = row
        self.col = col
        self.data = np.zeros([self.row,self.col])
        self.error = np.zeros([self.row,self.col])

    def S_data_callback(self,msg):
        self.data = np.reshape(np.asarray(msg.data),[self.row,self.col])

    def S_error_callback(self,msg):
        self.error = np.reshape(np.asarray(msg.data),[self.row,self.col])
        self.error_timestamp = msg.header.stamp.secs

    def S_velocity_callback(self,msg):
        self.joint_position = msg.position
        self.joint_velocity = msg.velocity
        self.velocity_timestamp = msg.header.stamp.secs


class Plotter(object):
    def __init__(self,error,error_timestamp):
        self.error = error
        self.error_features_mean = np.zeros([4,1])
        self.error_timestamp = error_timestamp 
        self.velocity_timestamp = velocity_timestamp

        self.t_error_axis = []
        self.t_velocity_axis = []
        self.error_1_axis = []
        self.error_2_axis = []
        self.error_3_axis = []
        self.error_4_axis = []
        self.mean_error_axis = []
        self.jv_1_axis = []
        self.jv_2_axis = []
        self.jv_3_axis = []
        self.jv_4_axis = []
        self.jv_5_axis = []
        self.jv_6_axis = []
        self.feature_1_x_axis = []
        self.feature_1_y_axis = []
        self.feature_2_x_axis = []
        self.feature_2_y_axis = []
        self.feature_3_x_axis = []
        self.feature_3_y_axis = []
        self.feature_4_x_axis = []
        self.feature_4_y_axis = []
        self.end_eff_x_axis = []
        self.end_eff_y_axis = []
        self.end_eff_z_axis = []

    def Plot_data(self,joint_velocity,current_features,transJ6):

            
        #######  mean_error : for  each feature
        for i in range(4):
            self.error_features_mean[i] = sqrt(np.sum(np.square(self.error[i])))
        ####### mean_error : overall
        self.error = np.reshape(self.error,(8,1))
        mean_error = sqrt(np.sum(np.square(self.error)))
        ########### XAXIS _____################ append
        t_error_axis.append(self.error_timestamp)
        t_velocity_axis.append(self.velocity_timestamp)


        ########### YAXIS_______############### append
        ########  error_mean vs time
        self.error_1_axis.append(self.error_features_mean[0])
        self.error_2_axis.append(self.error_features_mean[1])
        self.error_3_axis.append(self.error_features_mean[2])
        self.error_4_axis.append(self.error_features_mean[3])
        self.mean_error_axis.append(mean_error)
        ########### YAXIS_______############### append
        #########  joint_Vel vs time
        self.jv_1_axis.append(joint_velocity[0])
        self.jv_2_axis.append(joint_velocity[1])
        self.jv_3_axis.append(joint_velocity[2])
        self.jv_4_axis.append(joint_velocity[3])
        self.jv_5_axis.append(joint_velocity[4])
        self.jv_6_axis.append(joint_velocity[5])

        ########### YAXIS_______############### append
        # ########  points vs time
        self.feature_1_x_axis.append(current_features[0,0])
        self.feature_1_y_axis.append(current_features[0,1])
        self.feature_2_x_axis.append(current_features[1,0])
        self.feature_2_y_axis.append(current_features[1,1])
        self.feature_3_x_axis.append(current_features[2,0])
        self.feature_3_y_axis.append(current_features[2,1])
        self.feature_4_x_axis.append(current_features[3,0])
        self.feature_4_y_axis.append(current_features[3,1])

        ########### YAXIS_______############### append
        # ########  end_effector vs time
        self.end_eff_x_axis.append(transJ6[0,3])
        self.end_eff_y_axis.append(transJ6[1,3])
        self.end_eff_z_axis.append(transJ6[2,3])

    def Plot_show(self):

        ######################  FIGURE_WINDOW         ######################################################
        ##########__FIGURE_1
        fig1 = plt.figure(1)
        X_ax = self.t_error_axis
        Y_ax = self.mean_error_axis

        plt.plot(X_ax,Y_ax,label = "Mean Error")
        ####__PROPERTIES :: ###########
        plt.legend()
        plt.xlabel('Time(s) ------> ')
        plt.ylabel('Mean Error ------>')
        ######################  FIGURE_WINDOW         ######################################################

        ##########   __FIGURE_2
        fig2 = plt.figure(2)
        X_ax = self.t_velocity_axis
        Y_ax = [jv_1_axis,jv_2_axis,jv_3_axis,jv_4_axis,jv_5_axis,jv_6_axis]

        plt.plot(X_ax,Y_ax[0],label= "base")
        plt.plot(X_ax,Y_ax[1],label= "shoulder")
        plt.plot(X_ax,Y_ax[2],label= "elbow")
        plt.plot(X_ax,Y_ax[3],label= "wrist 1")
        plt.plot(X_ax,Y_ax[4],label= "wrist 2")
        plt.plot(X_ax,Y_ax[5],label= "wrist 3")
        ####__PROPERTIES :: ###########
        plt.xlabel('Time(s) ------>')
        plt.ylabel('Joint Velocity ------>')
        plt.legend(prop={'size':6})
        ######################  FIGURE_WINDOW         ######################################################

        ##########   __FIGURE_3
        fig3 = plt.figure(3)

        ######################
        l_abel = ['blue','green','red','orange']
        c_olor = ['blue','green','red','orange']
        s_ize = 50
        alp = 0.8
        c_map = ['b','g','r','orange']
        m_arker =['s','o']
        objx = [obj_1x,obj_2x,obj_3x,obj_4x]
        objy = [obj_1y,obj_2y,obj_3y,obj_4y]
        desiredPointsx = desiredPoints[:,0]
        desiredPointsy = desiredPoints[:,1]

        obj_nx = objx[0]
        obj_ny = objy[0]
        plt.plot(obj_nx,obj_ny,color=c_olor[0])
        plt.scatter(obj_nx[-1],obj_ny[-1],s=s_ize,c=c_map[0],marker=m_arker[0],alpha=alp,label='Final Position')
        plt.scatter(obj_nx[0],obj_ny[0],s=s_ize,c=c_map[0],marker=m_arker[1],alpha=alp,label='Start Position')
        plt.scatter(desiredPointsx[0],desiredPointsy[0],s=150,c=c_map[0],marker='*',alpha=alp,label='Desired Position')
        plt.legend(scatterpoints=1,fontsize=8)

        # print('obj',obj)
        ######################
        for i in range(1,4):
        
            obj_nx = objx[i]
            obj_ny = objy[i]

            plt.plot(obj_nx,obj_ny,color=c_olor[i])
            plt.scatter(obj_nx[-1],obj_ny[-1],s=s_ize,c=c_map[i],marker=m_arker[0],alpha=alp)
            plt.scatter(obj_nx[0],obj_ny[0],s=s_ize,c=c_map[i],marker=m_arker[1],alpha=alp)
            plt.scatter(desiredPointsx[i],desiredPointsy[i],s=150,c=c_map[i],marker='*',alpha=alp)

        plt.xlim((0,640))
        plt.ylim((0,480))
        plt.xlabel('u axis --------------->')
        plt.ylabel('<--------------- v axis') 
        plt.ylim(plt.ylim()[::-1])
        plt.title('Image Plane')


        for i in range(4):
            figi1 = plt.figure(i+11)
            obj_nx = objx[i]
            obj_ny = objy[i]
            plt.plot(obj_nx,obj_ny,color=c_olor[i])
            plt.scatter(obj_nx[-1],obj_ny[-1],s=s_ize,c=c_map[i],marker=m_arker[0],alpha=alp,label='Final Position')
            plt.scatter(obj_nx[0],obj_ny[0],s=s_ize,c=c_map[i],marker=m_arker[1],alpha=alp,label='Start Position')
            plt.scatter(desiredPointsx[i],desiredPointsy[i],s=150,c=c_map[i],marker='*',alpha=alp,label='Desired Position')
            plt.legend(scatterpoints=1,fontsize=8)

            plt.xlim((0,640))
            plt.ylim((0,480))

            plt.ylim(plt.ylim()[::-1])
            plt.title('Image Plane')

            plt.savefig('./ws/tumb_ws/saved/PLOTS_FORMAT/PLOTS_EPS/figINDVI_'+ str(f) + str(i)+'_'+Wp+'.eps', format = 'eps')


        ####__PROPERTIES :: ###########
        plt.xlabel('X-axis ------>')
        plt.ylabel('Y-Axis ------>')
        # plt.legend(prop={'size':6})
        ######################  FIGURE_WINDOW         ######################################################

        ##########   __FIGURE_4
        fig4 = plt.figure(4)
        ax = plt.axes(projection="3d")
        ax.set_title('Endeffector Position')
        # ax.view_init(90,0)
        ax.plot3D(end_eff_x,end_eff_y,end_eff_z,'blue')
        ax.scatter(end_eff_x[0],end_eff_y[0],end_eff_z[0],s=80,c ='g', marker='o',label='Start Position')
        ax.scatter(end_eff_x[-1],end_eff_y[-1],end_eff_z[-1],s=80,c ='r', marker='s',label='Final Position')
        ax.set_xlabel('X-axis ------>')
        ax.set_ylabel('Y-axis ------>')
        ax.set_zlabel('Z-axis ------>')
        plt.legend(scatterpoints=1,fontsize=8)
        

        ##########  __FIGURE_HANDLE_ ################################################################

        figr = [fig1,fig2,fig3,fig4]
        for fig in xrange(4):
            pdf.savefig(figr[fig])
            plt.figure(fig+1)
            plt.savefig('./ws/tumb_ws/saved/PLOTS_FORMAT/PLOTS_EPS/fig'+ str(b)+ str(fig)+'_'+a+'.eps', format = 'eps')
        pdf.close()    

        wb.open_new(r'./ws/tumb_ws/saved/PLOTS_FORMAT/PDF/Fig_Output_'+str(b) +'_'+a+'.pdf')

