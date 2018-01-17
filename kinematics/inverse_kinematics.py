'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity, array
from numpy.linalg import norm, inv
import numpy as np
from math import cos, sin, atan2, pi, asin, sqrt, acos, asin


class InverseKinematicsAgent(ForwardKinematicsAgent):

    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''

        foot_hip = transform - [[0,0,0,0],[0,0,0,50],[0,0,0,-85],[0,0,0,0]]
        transform_foot_hip = foot_hip[0:3, 3]
        h_ankle = norm(transform_foot_hip)
        knee_pitch = pi - np.arccos((10000 + 102.9*102.9 - h_ankle*h_ankle) / (200*102.9))

        c = cos(pi/4)
        s = sin(pi/4)
        transform_foot_hiporthogonal = np.dot([[1,0,0],[0,c,-s],[0,s,c]], transform_foot_hip)

        #Ankle Pitch
        anle_pitch_1 = np.arccos((102.9*102.9 + h_ankle*h_ankle - 10000) / (2*102.9*h_ankle))
        ankle_pitch_2 = atan2(transform_foot_hiporthogonal[0], norm(transform_foot_hiporthogonal[1:3]))
        ankle_pitch = anle_pitch_1 + ankle_pitch_2

        #Ankle Roll
        ankle_roll = atan2(transform_foot_hiporthogonal[1], transform_foot_hiporthogonal[2])

        #Hip
        hip_thigh = self.local_trans('LKneePitch', knee_pitch)\
            .dot(self.local_trans('LAnklePitch', ankle_pitch))\
            .dot(self.local_trans('LAnkleRoll', ankle_roll))\
            .dot(inv(foot_hip))

        hip_yaw_pitch = atan2(hip_thigh[1,0] * sqrt(2), hip_thigh[1,2])

        M = hip_thigh[0:3,0:3].dot(inv(self.calculate_matrix(hip_yaw_pitch, [0, 1, 1], False)))

        return [hip_yaw_pitch,
                atan2(-M[1, 2], M[1, 1]),
                atan2(-M[2, 0], M[0, 0]) - pi,
                knee_pitch,
                ankle_pitch,
                ankle_roll]


    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        angles = self.inverse_kinematics(effector_name, transform)
        name = list()
        times = list()
        keys = list()

        for chain in self.chains:
            if chain == 'LLeg':
                for i, joint in enumerate(self.chains[chain]):
                    name.append(joint)
                    keys.append([[angles[i], [0., 0., 0.], [0., 0., 0.]]])
                    times.append([10.0])
            else:
                for joint in self.chains[chain]:
                    name.append(joint)
                    keys.append([[0, [0., 0., 0.], [0., 0., 0.]]])
                    times.append([1.0])


        self.keyframes = (name, times, keys)  # the result joint angles have to fill in

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 1.0
    T[-1, 2] = -0.26
    agent.set_transforms('LLeg', T)
    agent.run()