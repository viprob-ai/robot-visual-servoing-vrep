#!/usr/bin/env python2.7
from math import *
import numpy as np 


class Kinematics(object):
    def __init__(self,T):
        self.T= T

    def Inverse_T(self):

        R = self.T[:3,:3]
        t = self.T[:3,3]
        MAPPING_R1 = np.concatenate((np.transpose(R), np.matmul(-np.transpose(R),t).reshape([3,1])),axis=1)
        MAPPING_R2 = np.concatenate((np.zeros([1,3]), [[1]]),axis=1)
        MAPPING = np.concatenate((MAPPING_R1,MAPPING_R2),axis=0)
        Inverse_T = MAPPING
        return Inverse_T


    def Adjoint_T(self):

        R = self.T[:3,:3]
        t = self.T[:3,3]
        SkewVec = t
        SkewM = np.array([[          0, -SkewVec[2],  SkewVec[1]],
                        [ SkewVec[2],           0, -SkewVec[0]],
                        [-SkewVec[1],  SkewVec[0],           0]
                        ])

        MAPPING_R1 = np.append(R, np.matmul(SkewM,R),axis=1)
        MAPPING_R2 = np.append(np.zeros([3,3]), R,axis=1)
        MAPPING = np.append(MAPPING_R1,MAPPING_R2,axis=0)
        return MAPPING

    def Skew_T(self):
        SkewVec = self.T.flatten()
        SkewM = np.array([[          0, -SkewVec[2],  SkewVec[1]],
                        [ SkewVec[2],           0, -SkewVec[0]],
                        [-SkewVec[1],  SkewVec[0],           0]
                        ])
        return SkewM

    def Jacobian_T(self):  # o_J_e  = Jacobian_o_T_e * e_J_e
        R = self.T[:3,:3]
        MAPPING_R1 = np.append(R,               np.zeros([3,3]), axis=1)
        MAPPING_R2 = np.append(np.zeros([3,3]), R,               axis=1)
        MAPPING = np.append(MAPPING_R1,MAPPING_R2,axis=0)       
        Jacobian_T = MAPPING
        return Jacobian_T 


    def Manipulator_FK_Jacobian(self):
        Cross = np.zeros([3,1])
        H = np.round(self.T,4)
        J = np.zeros([6,6])
        for i in range(6):
            O_e = H[6,:3,3]
            O_i = H[i,:3,3] 
            Z_i = H[i,:3,2]

            O_p = O_e - O_i

            Cross[0,0] = Z_i[1]*O_p[2]-Z_i[2]*O_p[1]
            Cross[1,0] = Z_i[2]*O_p[0]-Z_i[0]*O_p[2]
            Cross[2,0] = Z_i[0]*O_p[1]-Z_i[1]*O_p[0]

            J[:3,i] = Cross[:,0]
            J[3:,i] = Z_i

        return J
