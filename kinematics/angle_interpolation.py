'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''

import numpy as np
from joint_control.pid import PIDAgent
from joint_control.keyframes import *


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.start = -1

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}

        if (self.start == -1):
            self.start = perception.time
        curr_time = perception.time - self.start

        for i, name in enumerate(keyframes[0]):
            if name not in perception.joint: continue

            time = keyframes[1][i]
            keys = keyframes[2][i]

            #find right key:
            j = -1
            for k, t in enumerate(time):
                if (t > curr_time):
                    j = k
                    break
            if j == -1:
                print('end')
                target_joints[name] = 0
                continue

            last_time = keys[j][1][1]
            last_angle = keys[j][1][2]
            bez_end = (time[j], keys[j][0])
            bez_end_handle =  np.add(bez_end, (last_time, last_angle))

            if (j <= 0):
                start_time = - last_time
                start_angle = 0
                bez_start = (0, perception.joint[name])
                bez_start_handle = np.add(bez_start, (start_time, start_angle))

            else:
                start_time =  keys[j-1][2][1]
                start_angle = keys[j-1][2][2]
                bez_start = (time[j-1], keys[j-1][0])
                bez_start_handle =  np.add(bez_start, (start_time, start_angle))

            target_angle = self.value_bez(bez_start[1],
                                                      bez_start_handle[1],
                                                      bez_end_handle[1],
                                                      bez_end[1],
                                                      self.root_bez(bez_start[0],
                                                     bez_start_handle[0],
                                                     bez_end_handle[0],
                                                     bez_end[0],
                                                     curr_time))

            target_joints[name] = target_angle

        return target_joints

    @staticmethod
    def value_bez(x0, x1, x2, x3, val):
        return np.polyval([-     x0 + 3 * x1 - 3 * x2 + x3,
                             3 * x0 - 6 * x1 + 3 * x2,
                           - 3 * x0 + 3 * x1,
                                 x0],
                          val)

    @staticmethod
    def root_bez(x0, x1, x2, x3, t):
        roots = np.roots([-   x0 + 3 * x1 - 3 * x2 + x3,
                                 3 * x0 - 6 * x1 + 3 * x2,
                               - 3 * x0 + 3 * x1,
                                -t + x0])
        i = []
        for r in roots:
            if np.isreal(r) and 0 <= np.real(r) <= 1:
                i.append(r)
        for r in roots:
            if np.isreal(r) and 0 <= np.real(r) <= 1:
                return np.real(r)
        return 0


if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = rightBellyToStand() #  # CHANGE DIFFERENT KEYFRAMES
    agent.run()