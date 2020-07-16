#!/usr/bin/env python2.7
from math import *
import numpy as np 
import cv2
from cv2 import aruco
from cv_bridge import CvBridge, CvBridgeError 

import rospy 

#### Class Import
import Object_Detection.C_ObjectDetection as cod
import Kinematics.K_Kinematics as ks
import Visual_Servoing.VS_VisualServoing as vs
import Avoidance_Task.T_Joint_avoidance as ja
#### Message Import

from sensor_msgs.msg import Image,JointState
from std_msgs.msg import Float32MultiArray


def main():
########################################################################################
#### -- Initialize Node --
    rospy.init_node('VisualServoing')  
########################################################################################
#### -- Subscribing --
#### --Image--
    current_ibvs_cb = vs.Subscribe('/visionSensor_currentFeatures',Float32MultiArray,(2,4,3))
    desired_ibvs_cb = vs.Subscribe('/visionSensor_desiredFeatures',Float32MultiArray,(2,4,3))

#### --Transformation--JointState--JointRates{Publish}
    n = [6,6,2]
    jtree = ['rj','lj','hj']
    eetree =['rl','ll','hl']

    o_M_jn = np.zeros([len(n),6,4,4])
    o_M_e = np.zeros([len(n),4,4])
    o_M_jn_e = np.zeros([len(n),7,4,4])
    jointState_position = np.zeros([len(n),6,1])

    o_M_j_cb = [object,object,object]
    o_M_e_cb = [object,object,object]

    jointState_position_cb = [object,object,object]
    pub_jointRates = [object,object,object]

    for i in range(len(n)):
        o_M_j_cb[i] = vs.Subscribe('/coppeliaSim/baseTransformation_'+jtree[i],Float32MultiArray,(n[i],4,4)) 
        o_M_e_cb[i] = vs.Subscribe('/coppeliaSim/baseTransformation_'+eetree[i] +'EndEffector',Float32MultiArray,(4,4))         
        jointState_position_cb[i] = vs.Subscribe('/coppeliaSim/jointState_' + jtree[i] ,JointState,(6,1))

        pub_jointRates[i] = rospy.Publisher('/jointRates_'+ jtree[i]+'/coppeliaSim',Float32MultiArray,queue_size=10)

########################################################################################
#### -- Publishing --
    rospy.sleep(2)

    while not rospy.is_shutdown():
        key = cv2.waitKey(1) & 0xFF
        ###########################################################################################################
        #### --Image--
        current_ibvs_cb.S_data_callback(current_ibvs_cb.msg.data)
        current_ibvs_features = current_ibvs_cb.data

        desired_ibvs_cb.S_data_callback(desired_ibvs_cb.msg.data)
        desired_ibvs_features = desired_ibvs_cb.data

        #### --Kinematics--
        for i in range(2):
            o_M_j_cb[i].S_data_callback(o_M_j_cb[i].msg.data)
            o_M_e_cb[i].S_data_callback(o_M_e_cb[i].msg.data)
            o_M_jn[i] = o_M_j_cb[i].data
            o_M_e[i] = o_M_e_cb[i].data
            o_M_jn_e[i] = np.append(o_M_jn[i],[o_M_e[i]],axis = 0)
            jointState_position_cb[i].S_data_callback(jointState_position_cb[i].msg.position)
            jointState_position[i] = jointState_position_cb[i].data

        o_M_e_cb[-1].S_data_callback(o_M_e_cb[-1].msg.data)
        o_M_c = o_M_e_cb[-1].data

        #############################################################################################################
        o_J_e = np.zeros([2,6,6])
        e_J_e = np.zeros([2,6,6])
        for i in range(2):
            fk_jacobian_cb = ks.Kinematics(o_M_jn_e[i])
            fk_jacobian = fk_jacobian_cb.Manipulator_FK_Jacobian()
            o_J_e[i] = np.round(fk_jacobian,3)
            e_M_o = ks.Kinematics(o_M_jn_e[i,-1,:,:]).Inverse_T()
            e_J_o_cb = ks.Kinematics(e_M_o)
            e_J_o = e_J_o_cb.Jacobian_T()
            e_J_e[i] = np.matmul(e_J_o,o_J_e[i])

        # print(o_J_e[0])
        condnumJ = np.linalg.cond(o_J_e[0],'fro')
        print('condition number',condnumJ)

#### --Class_instance_method
        #### Visual Servoing Class -----------------------------------------------------------------------
        dot_q1_ibvs = np.zeros([2,6,1])
        delta = 10
        for i in range(2):
            #############################################################################################
            vs_IBVS_cb = vs.VisualServoing_IBVS(4,current_ibvs_features[i],desired_ibvs_features[i],delta)         # --$$--i -- current_ibvs_features[i],desired_ibvs_features[i]
            vs_IBVS = vs_IBVS_cb.Interaction_Matrix_IBVS()

            """
            USE EITHER TASK SPACE OR JOINT SPACE CONTROL
            COMMMENT / UNCOMMENT ALL THE CORRESPONDING LINES ACCORDINGLY
            """

            L_IBVS = vs_IBVS[0]
            Ld_IBVS = vs_IBVS[1]
            error_IBVS = vs_IBVS[2]
            mean_error_IBVS = vs_IBVS[3]

            #### 1. L_e = L_IBVS
            # L_e_IBVS = L_IBVS
            # #### 2. L_e = Ld_IBVS
            # L_e_IBVS = Ld_IBVS
            # #### 3. L_e = (L_IBVS + Ld_IBVS)/2
            L_e_IBVS = (L_IBVS + Ld_IBVS)/2
            ###############################################################################################

            c_M_o = ks.Kinematics(o_M_c).Inverse_T()
            c_V_o = ks.Kinematics(c_M_o).Adjoint_T()

            J_e_IBVS = np.matmul(L_e_IBVS,np.matmul(c_V_o,o_J_e[i]))                                                 # --$$--i -- o_j_e[i]

            controller_cb = vs.Controllers(J_e_IBVS,error_IBVS,mean_error_IBVS)

            var_lambda = 0.5 
            inv_J_e_IBVS = controller_cb.LM_Controller(var_lambda,0.5)
            dot_q1_ibvs[i] = controller_cb.Control_Law(-inv_J_e_IBVS,var_lambda,delta)                               # --$$--i -- dot_q1_ibvs[i]
            

#### --Class_instance_pub ------------------------------------------------------
        for i in range(2):
            jointRates = vs.Publish(pub_jointRates[i],Float32MultiArray)
            jointRates._msg.data = dot_q1_ibvs[i].flatten()
            jointRates.P_data(jointRates._msg)
            
########################################################################################

if __name__ =='__main__':
    main()

