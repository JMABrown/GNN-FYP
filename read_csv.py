# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 20:35:29 2019

@author: user
"""
import matplotlib.pyplot as plt
import csv
from enum import Enum
import numpy as np
import quaternion

class BodyPart(Enum):
    SPINE_BASE = 0
    SPINE_MID = 1
    NECK = 2
    HEAD = 3
    SHOULDER_L = 4
    ELBOW_L = 5
    WRIST_L = 6
    HAND_L = 7
    SHOULDER_R = 8
    ELBOW_R = 9
    WRIST_R = 10
    HAND_R = 11
    HIP_L = 12
    KNEE_L = 13
    ANKLE_L = 14
    FOOT_L = 15
    HIP_R = 16
    KNEE_R = 17
    ANKLE_R = 18
    FOOT_R = 19
    SPINE_SHOULDER = 20
    HANDTIP_L = 21
    THUMB_L = 22
    HANDTIP_R = 23
    THUMB_R = 24

pos_legend = ["X", "Y", "Z"]
rot_legend = ["W", "X", "Y", "Z"]
eul_legend = ["X", "Y", "Z"]

count = 0

column_header = ""
body_stream = []
for i in range(25):
    body_stream.append([])

body_part = BodyPart.KNEE_L

with open('high_knees.csv', 'rt') as f:
    csv_reader = csv.reader(f)

    for line in csv_reader:
        
        if (count == 0):
            column_header = line
        else:
            temp_line = []
            for col in line:
                temp_line.append(float(col))
            
            body_stream[(count-1)%25].append(temp_line)
        count += 1

########## PLOT THE POSITION ##########

x_plots2 = [body_stream[body_part.value][i][0:3] for i in range(len(body_stream[0]))]
fig = plt.figure()
ax = plt.subplot(111)
x_axis = [i for i in range(len(x_plots2))]
for i in range(len(x_plots2[0])):
    ax.plot(x_axis, [x_plots2[j][i] for j in range(len(x_plots2))], label = pos_legend[i])
ax.legend()
plt.xlabel("Frame")
plt.ylabel("Distance (metres)")
plt.show()

########## PLOT THE QUATERNION ##########

x_plots3 = [body_stream[body_part.value][i][3:7] for i in range(len(body_stream[0]))]

fig = plt.figure()
ax = plt.subplot(111)
x_axis = [i for i in range(len(x_plots3))]
for i in range(len(x_plots3[0])):
    ax.plot(x_axis, [x_plots3[j][i] for j in range(len(x_plots3))], label = rot_legend[i])
ax.legend()
plt.xlabel("Frame")
plt.ylabel("Rotation Quaternion")
plt.show()

########## PLOT THE EULER ANGLES ##########

def VecToQuat(vec):
    return np.quaternion(vec[0], vec[1], vec[2], vec[3])

def QuatToVec(quat):
    return [quat.w, quat.x, quat.y, quat.z]

def QuatToEuler(quat):
    test = quat.x*quat.y + quat.z*quat.w
    if (test > 0.499):
        heading = 2*np.arctan2(quat.x, quat.w)
        attitude = np.pi/2
        bank = 0
    elif (test < -0.499):
        heading = -2 * np.arctan2(quat.x, quat.w)
        attitude = - np.pi / 2
        bank = 0
    else:
        sqx = quat.x*quat.x
        sqy = quat.y*quat.y
        sqz = quat.z*quat.z
        heading = np.arctan2(2*quat.y*quat.w - 2*quat.x*quat.z, 1 - 2*sqy - 2*sqz)
        attitude = np.arcsin(2*test)
        bank = np.arctan2(2*quat.x*quat.w - 2*quat.y*quat.z, 1 - 2*sqx - 2*sqz)
    #return [rot in x, rot in y, rot in z] 
    #return [heading, attitude, bank]
    return [bank, heading, attitude]    #x, y, z

x_plots4 = [QuatToEuler(VecToQuat(body_stream[body_part.value][i][3:7])) for i in range(len(body_stream[0]))]

fig = plt.figure()
ax = plt.subplot(111)
x_axis = [i for i in range(len(x_plots2))]
for i in range(len(x_plots4[0])):
    ax.plot(x_axis, [x_plots4[j][i] for j in range(len(x_plots4))], label = eul_legend[i])
ax.legend()
plt.xlabel("Frame")
plt.ylabel("Rotation (Radians)")
plt.show()