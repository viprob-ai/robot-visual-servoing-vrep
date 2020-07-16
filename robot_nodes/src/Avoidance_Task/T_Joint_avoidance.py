#!/usr/bin/env python2.7
from math import *
import numpy as np 

# q = q1 + q2

class Joint_limit_avoidance_task(object):
    def __init__(self,q1,q_i,J_e,error):
        self.q1 = q1[:,0]
        self.q_i = q_i[:,0]
        self.J_e = J_e
        self.e = error
        #### JOINT LIMITS IN DEGREES
        self.q_limit = np.asarray([[-180,90],
                                   [-90,5],
                                   [-90,90],
                                   [-125,0],
                                   [-150,150],
                                   [-70,70]]) 
        self.q_limit = np.deg2rad(self.q_limit)
        self.q_limit = np.reshape(self.q_limit,[6,2])
                                    

    def task(self):
        n = 6
        J_e = self.J_e
        e = self.e
        q1 = self.q1
        q_i = np.round(self.q_i,4)

        q_limit = self.q_limit

        min_q_i = q_limit[:,0]
        max_q_i = q_limit[:,1]
        delta_q_i = max_q_i - min_q_i
        rho = 0.1 #[0-0.5]
        min_q_l0_i = min_q_i + rho*delta_q_i
        max_q_l0_i = max_q_i - rho*delta_q_i

        rho_1 = 0.5 #[0-1]
        min_q_l1_i = min_q_l0_i - (rho_1 * rho * delta_q_i)
        max_q_l1_i = max_q_l0_i + (rho_1 * rho * delta_q_i)

        P_q_max,P_q_min,Range_q = np.zeros([6,3]),np.zeros([6,3]),np.zeros([6,3])
        P_q_max[:,0] = max_q_l0_i
        P_q_max[:,1] = max_q_l1_i
        P_q_max[:,2] = max_q_i 
        P_q_min[:,0] = min_q_i
        P_q_min[:,1] = min_q_l1_i 
        P_q_min[:,2] = min_q_l0_i
         
        Range_q[:,0] = min_q_l0_i
        Range_q[:,1] = q_i
        Range_q[:,2] = max_q_l0_i 

        # if q_i<min_q_l0_i:
        #     delta_i = q_i - min_q_l0_i
        # elif q_i>max_q_l0_i:
        #     delta_i = q_i - max_q_l0_i
        # else:
        #     delta_i = 0


        # g_i = delta_i/delta_q_i
        # P_e = 
        #### ---
        # d_q2_i = -P_e*g_i

        #### ----
        # P=P_e or P_norm_e
        # d_q2_i = -lamda_sec_i * lamda_l_i * P * l_g_i

#### -- activation sign function --
        l_g_i = np.zeros([n,n])
        for i in range(n):
            for i_0 in range(n):
                if (q_i[i] < min_q_l0_i[i]) and (i==i_0):
                    l_g_i[i,i_0] = -1
                elif (q_i[i] > max_q_l0_i[i]) and (i==i_0):
                    l_g_i[i,i_0] = 1
                else:
                    l_g_i[i,i_0] = 0


#### --PROJECTION OPERATOR
        I_n = np.eye(n)

        T_J_e = np.transpose(J_e)
        T_e = np.transpose(e)
        P_norm_e = I_n - (1/(np.matmul(np.matmul(T_e,J_e),np.matmul(T_J_e,e))))*(np.matmul(np.matmul(T_J_e,e),np.matmul(T_e,J_e)))
        P = P_norm_e

        lamda_i = 0.5

#### -- TUNING FUNCTION
        max_L_l_i = np.zeros([6,1])
        min_L_l_i = np.zeros([6,1]) 

        for i in range(n):
            max_L_l_i[i] = 1/(1+exp(-12*((q_i[i] - max_q_l0_i[i])/(min_q_l1_i[i] - max_q_l0_i[i]))+6))
            min_L_l_i[i] = 1/(1+exp(-12*((q_i[i] - min_q_l0_i[i])/(min_q_l1_i[i] - min_q_l0_i[i]))+6))

        min_L_l0_i = np.zeros([n,1])
        max_L_l0_i = np.zeros([n,1])

        min_L_l1_i = np.ones([n,1])
        max_L_l1_i = np.ones([n,1])

        lamda_l_i = np.zeros([6,1])

        for i in range(n):
                
            if (q_i[i] < min_q_l1_i[i]) or (max_q_l1_i[i] < q_i[i]):
                lamda_l_i[i] = 1 
            elif min_q_l1_i[i] <= q_i[i] <= min_q_l0_i[i]:
                lamda_l_i[i] = (min_L_l_i[i] - min_L_l0_i[i])/(min_L_l1_i[i] - min_L_l0_i[i])
            elif max_q_l0_i[i] <= q_i[i] <= max_q_l1_i[i]:
                lamda_l_i[i] = (max_L_l_i[i] - max_L_l0_i[i])/(max_L_l1_i[i] - max_L_l0_i[i])
            elif min_q_l0_i[i] < q_i[i] < max_q_l0_i[i]:
                lamda_l_i[i] = 0 



        d_q2_i = np.zeros([6,6])
        for i in range(n):
            P_g_i = np.matmul(P,np.transpose(l_g_i[i]))

            if (q_i[i] < min_q_l1_i)[i] or (max_q_l1_i[i] < q_i[i]):
                d_q2_i[:,i] = -(1+lamda_i)*(abs(q1[i])/abs(P_g_i[i]))* P_g_i

            elif (min_q_l1_i[i] <= q_i[i] <= min_q_l0_i[i]) or (max_q_l0_i[i] <= q_i[i] <=max_q_l1_i[i]):

                d_q2_i[:,i] = - lamda_l_i[i] * (1+lamda_i) * (abs(q1[i])/abs(P_g_i[i]))* P_g_i
            elif min_q_l0_i[i] < q_i[i] < max_q_l0_i[i]:
                d_q2_i[:,i] = 0
        
        
        q2 = np.sum(d_q2_i,axis=1)
        # q = q1 + q2 


        # print('joint angles',q_i)
        # print('max range',P_q_max)
        # print('min range',P_q_min)
        # print('range joint',Range_q)

        # print('Activation and sign function',l_g_i)

        # print('Projection Operator',P)

        # print('Tuning Func',lamda_l_i)

        # print('d_q2_i',d_q2_i)

        # print('q1',q1)
        # print('q2',q2)

        # print('q',q)
        return q2