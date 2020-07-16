#!/usr/bin/env python2.7
from math import *
import numpy as np 
import cv2
from cv2 import aruco
from cv_bridge import CvBridge, CvBridgeError 

import rospy 

#### Class Import
import Kinematics.K_Kinematics as ks

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
        # print(self.data.shape)

           

class VisualServoing_IBVS(object):
    def __init__(self,n,current_features,desired_features,delta):
        self.n = n
        self.uc = 320
        self.vc = 240
        self.L = np.zeros([self.n*2,6])
        self.Ld = np.zeros([self.n*2,6])
        self.fl = 530
        self.delta = delta
        self.current_image_features = current_features[:,:2]
        self.desired_image_features = desired_features[:,:2]
        self.current_depth_features = current_features[:,2]
        self.desired_depth_features = desired_features[:,2]
        self.error = self.current_image_features - self.desired_image_features


    def Interaction_Matrix_IBVS(self):
        
        self.error = np.reshape(self.error,(8,1))
        mean_error = sqrt(np.sum(np.square(self.error)))
        j=0
        for i in range(self.n):
            u = self.current_image_features[i,0]
            v = self.current_image_features[i,1]
            Z = round(self.current_depth_features[i],5) 

            if abs(Z)!=0:
                pass
            else:
                Z = 1
            
            _u = u - self.uc  #should be applied
            _v = v - self.vc

            self.L[j:j+2,:] = np.array([[ -self.fl/Z,  0,    _u/Z,     _u*_v/self.fl,     -(self.fl*self.fl+_u*_u)/self.fl,  _v],
                                        [    0,  -self.fl/Z, _v/Z, (self.fl*self.fl+_v*_v)/self.fl,      -(_u*_v)/self.fl,    -_u]])


            u = self.desired_image_features[i,0]
            v = self.desired_image_features[i,1]
            Z = round(self.desired_depth_features[i],5) 
            if abs(Z)!=0:
                pass
            else:
                Z = 1
            
            _u = u - self.uc
            _v = v - self.vc

            self.Ld[j:j+2,:] = np.array([[ -self.fl/Z,  0,    _u/Z,     _u*_v/self.fl,     -(self.fl*self.fl+_u*_u)/self.fl,  _v],
                                        [    0,  -self.fl/Z, _v/Z, (self.fl*self.fl+_v*_v)/self.fl,      -(_u*_v)/self.fl,    -_u]])

            j=j+2 
        return self.L,self.Ld,self.error,mean_error


class VisualServoing_PBVS(object):
    def __init__(self,current_features,desired_features,delta):
        self.fl = 530
        self.delta = delta

        self.c_M_O = current_features
        self.cd_M_Od = desired_features
        Od_M_cd_cb = ks.Kinematics(self.cd_M_Od)
        Od_M_cd = Od_M_cd_cb.Inverse_T()
        self.desired_M_current = np.matmul(Od_M_cd,self.c_M_O)
        self.desired_t_current = self.desired_M_current[:3,3]
        self.desired_R_current = self.desired_M_current[:3,:3]

    def Interaction_Matrix_PBVS(self):

        R = self.desired_R_current

        th = acos((np.round(np.trace(R),2)-1)/2)
        if th == 0:
            U = np.zeros([3,1])
        else:
            U = (1/(2*sin(th)))*np.asarray([[R[2,1] - R[1,2]],[R[0,2] - R[2,0]],[R[1,0] - R[0,1]]])

        error = np.append(self.desired_t_current,(th*U).flatten())
        mean_error = sqrt(np.sum(np.square(error)))

        skewU_cb = ks.Kinematics(U)
        skewU = skewU_cb.Skew_T()
        L_th_u = np.eye(3) - (th/2)*(skewU)  + (1-(np.sinc(th)/(np.sinc(th/2))**2)) * np.square(skewU)
        self.L = np.append(np.append(R,np.zeros([3,3]),axis=1),np.append(np.zeros([3,3]),L_th_u,axis=1),axis=0)
                                            
        return self.L,error,mean_error
class CylindricalObject_PBVS(VisualServoing_PBVS):
    
    def __init__(self,current_features,desired_features,delta):
        # super(VisualServoing_PBVS,self).__init__()
        VisualServoing_PBVS.__init__(self,current_features,desired_features,delta)
        self.Od_z_O = self.desired_R_current[:,2]
        self.zd = np.asarray([[0],[0],[1]])
    
    def Interaction_Matrix_cylPBVS(self):
        self.error_c = np.append(self.desired_t_current,np.cross(self.Od_z_O.flatten(),self.zd.flatten()),axis=0)
        self.mean_error_c = sqrt(np.sum(np.square(self.error_c)))
        skew_Od_z_O_cb = ks.Kinematics(self.Od_z_O) 
        skew_Od_z_O = skew_Od_z_O_cb.Skew_T()
        skew_zd_cb = ks.Kinematics(self.zd)
        skew_zd = skew_zd_cb.Skew_T() 
        L_cyl = np.matmul(skew_Od_z_O,skew_zd)
        R = self.desired_R_current
        self.L_ec = np.append(np.append(R,np.zeros([3,3]),axis=1),np.append(np.zeros([3,3]),-L_cyl,axis=1),axis=0)
        return self.L_ec,self.error_c,self.mean_error_c


    
class Controllers(object):
    def __init__(self,J_e,error,mean_error):
        self.J_e = J_e
        self.error = error
        self.mean_error = mean_error

    def LM_Controller(self,var_lambda,var_mu):
        #### CONTROLLERS LEVENBERG MARQUART
        J_e_transpose = np.transpose(self.J_e)
        H_e = np.matmul(J_e_transpose,self.J_e)

        H_e_diagonalised =np.diag(np.diag(H_e)) ## diag just gives diagonal elements array but diag.diag gives matrix with non diagonal elem zero
        
    ####################################################################################################################################################
        # if mean_error >= self.delta * 15:
        #     var_mu = 1
        #     var_lambda = 0.1

        # elif mean_error < self.delta * 15:
        #     var_mu = 1
        #     var_lambda = 0.2
        # elif mean_error < self.delta * 10:
        #     var_mu = 0.5
        #     var_lambda = 0.5
        # elif mean_error < self.delta * 6:
        #     var_mu = 0.01
        #     var_lambda = 0.8   
        # elif mean_error < self.delta * 4:
        #     var_mu = 0.005
        #     var_lambda = 0.9
        # elif mean_error < self.delta * 3:
        #     var_mu = 0.01
        #     var_lambda = 0.9
        # elif mean_error < self.delta * 2.5:
        #     var_mu = 0.008
        #     var_lambda = 0.9

        # elif mean_error <=self.delta * 2:
        #     var_mu = 0.001
        #     var_lambda = 1
        # elif mean_error < self.delta * 1.5:
        #     var_mu = 0.0005
        #     var_lambda = 1   

        # if mean_error >= self.delta:
        # if mean_error >= 200:
        #     var_mu = 1
        #     var_lambda = 0.5
        # var_mu = 0.01
        # fac_1 = 3/4 
        # var_mu = fac_1*var_mu
        if self.mean_error > 200:
            var_mu = 0.001
            var_lambda = 0.2

        elif self.mean_error <=200:
            var_mu = 0.0001
            var_lambda = 0.5

        elif self.mean_error <=100:
            var_mu = 0.00001
            var_lambda = 1

    ##################################################################################        
        H_e_sum = H_e + var_mu*H_e_diagonalised
        H_e_sum_inverse = np.linalg.pinv(H_e_sum)
        #### CONTROL LAW
        J_e_inverse = np.matmul(H_e_sum_inverse,J_e_transpose)
        return J_e_inverse

    def P_Controller(self,var_lambda):
        J_e_transpose = np.transpose(self.J_e)
        J_e_inverse = np.linalg.pinv(self.J_e)

        #### L_manupability_index = sqrt(np.linalg.det(np.matmul(L,L_transpose)))

        return J_e_inverse

    def Control_Law(self,J_e_inverse,var_lambda,delta):
        ##############################################################################################################################################
        # if self.mean_error >= delta:
        jointVelocity = - var_lambda * np.matmul(J_e_inverse,self.error)
        # else:
        #     jointVelocity = np.zeros([6,1])

        return jointVelocity