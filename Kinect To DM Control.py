from dm_control import suite
import matplotlib.pyplot as plt
import numpy as np
import csv
from enum import Enum
import quaternion
import cv2
import time
from numpy import linalg as la
import tensorflow as tf
from tensorflow import keras
import glob
import os
import random

k2dm = np.quaternion(0.5, -0.5, 0.5, 0.5)

def KinectToDMControl(body_stream):
    
    dm_body_stream = {}
    dm_body_stream['left_knee'] = []
    dm_body_stream['right_knee'] = []
    dm_body_stream['right_elbow'] = []
    dm_body_stream['left_elbow'] = []
    dm_body_stream['right_shoulder1'] = []
    dm_body_stream['right_shoulder2'] = []
    dm_body_stream['left_shoulder1'] = []
    dm_body_stream['left_shoulder2'] = []
    
    dm_body_stream['left_hip_x'] = []
    dm_body_stream['left_hip_y'] = []
    dm_body_stream['left_hip_z'] = []
    
    dm_body_stream['right_hip_x'] = []
    dm_body_stream['right_hip_y'] = []
    dm_body_stream['right_hip_z'] = []
    
    for i in range(len(body_stream[0])):
        #r_knee_quat = VecToQuat(body_stream[BodyPart.KNEE_R.value][i][3:7])
        #r_ankle_quat = VecToQuat(body_stream[BodyPart.ANKLE_R.value][i][3:7])
        #l_knee_quat = VecToQuat(body_stream[BodyPart.KNEE_L.value][i][3:7])
        #l_ankle_quat = VecToQuat(body_stream[BodyPart.ANKLE_L.value][i][3:7])
        #r_elbow_quat = VecToQuat(body_stream[BodyPart.ELBOW_R.value][i][3:7])
        #r_wrist_quat = VecToQuat(body_stream[BodyPart.WRIST_R.value][i][3:7])
        #l_elbow_quat = VecToQuat(body_stream[BodyPart.ELBOW_L.value][i][3:7])
        #l_wrist_quat = VecToQuat(body_stream[BodyPart.WRIST_L.value][i][3:7])
        #dm_body_stream['right_knee'].append(AngleBetweenQuat(r_knee_quat, r_ankle_quat))
        #dm_body_stream['left_knee'].append(AngleBetweenQuat(l_knee_quat, l_ankle_quat))
        #dm_body_stream['right_elbow'].append(AngleBetweenQuat(r_elbow_quat, r_wrist_quat))
        #dm_body_stream['left_elbow'].append(AngleBetweenQuat(l_elbow_quat, l_wrist_quat))
        
        #dm_body_stream['right_shoulder1'].append(QuatToEuler(VecToQuat(body_stream[BodyPart.SHOULDER_R.value][i][3:7]))[2])
        #dm_body_stream['right_shoulder2'].append(QuatToEuler(VecToQuat(body_stream[BodyPart.SHOULDER_R.value][i][3:7]))[1])
        
        r_hip_pos = np.array(body_stream[BodyPart.HIP_R.value][i][0:3])
        r_knee_pos = np.array(body_stream[BodyPart.KNEE_R.value][i][0:3])
        r_ankle_pos = np.array(body_stream[BodyPart.ANKLE_R.value][i][0:3])
        
        l_hip_pos = np.array(body_stream[BodyPart.HIP_L.value][i][0:3])
        l_knee_pos = np.array(body_stream[BodyPart.KNEE_L.value][i][0:3])
        l_ankle_pos = np.array(body_stream[BodyPart.ANKLE_L.value][i][0:3])
        
        r_shoulder_pos = np.array(body_stream[BodyPart.SHOULDER_R.value][i][0:3])
        r_elbow_pos = np.array(body_stream[BodyPart.ELBOW_R.value][i][0:3])
        r_wrist_pos = np.array(body_stream[BodyPart.WRIST_R.value][i][0:3])
        
        l_shoulder_pos = np.array(body_stream[BodyPart.SHOULDER_L.value][i][0:3])
        l_elbow_pos = np.array(body_stream[BodyPart.ELBOW_L.value][i][0:3])
        l_wrist_pos = np.array(body_stream[BodyPart.WRIST_L.value][i][0:3])
        
#        forearm_bone = r_elbow_pos - r_shoulder_pos
#        forearm_dir = forearm_bone / la.norm(forearm_bone)
#        
#        shoulder_axis1 = np.array([2, 1, 1])
#        shoulder_axis1 = shoulder_axis1 / la.norm(shoulder_axis1)
#        shoulder_axis2 = np.array([0, -1, 1])
#        shoulder_axis2 = shoulder_axis2 / la.norm(shoulder_axis2)
#        
#        shoulder_angle1 = np.arccos(forearm_dir.dot(shoulder_axis1))
#        shoulder_angle2 = np.arccos(forearm_dir.dot(shoulder_axis2))
#        
#        dm_body_stream['right_shoulder1'].append(shoulder_angle1 - np.pi/2)
#        dm_body_stream['right_shoulder2'].append(shoulder_angle2 - np.pi/2)
#        
#        forearm_bone = l_elbow_pos - l_shoulder_pos
#        forearm_dir = forearm_bone / la.norm(forearm_bone)
#        
#        shoulder_axis1 = np.array([2, -1, 1])
#        shoulder_axis1 = shoulder_axis1 / la.norm(shoulder_axis1)
#        shoulder_axis2 = np.array([0, 1, 1])
#        shoulder_axis2 = shoulder_axis2 / la.norm(shoulder_axis2)
#        
#        shoulder_angle1 = np.arccos(forearm_dir.dot(shoulder_axis1))
#        shoulder_angle2 = np.arccos(forearm_dir.dot(shoulder_axis2))
#        
#        dm_body_stream['left_shoulder1'].append(shoulder_angle1 - np.pi/2)
#        dm_body_stream['left_shoulder2'].append(shoulder_angle2 - np.pi/2)
        
        dm_body_stream['right_shoulder1'].append(AngleToAxes(r_shoulder_pos, r_elbow_pos, [2, 1, 1], np.arccos) - np.pi/2)
        dm_body_stream['right_shoulder2'].append(AngleToAxes(r_shoulder_pos, r_elbow_pos, [0, -1, 1], np.arccos) - np.pi/2)
        
        dm_body_stream['left_shoulder1'].append(AngleToAxes(l_shoulder_pos, l_elbow_pos, [2, -1, 1], np.arccos) - np.pi/2)
        dm_body_stream['left_shoulder2'].append(AngleToAxes(l_shoulder_pos, l_elbow_pos, [0, 1, 1], np.arccos) - np.pi/2)
        
        #dm_body_stream['right_shoulder1'].append(AngleToAxes(r_shoulder_pos, r_elbow_pos, [2, 1, 1], np.arcsin))
        #dm_body_stream['right_shoulder2'].append(AngleToAxes(r_shoulder_pos, r_elbow_pos, [0, -1, 1], np.arcsin))
        
        #dm_body_stream['left_shoulder1'].append(AngleToAxes(l_shoulder_pos, l_elbow_pos, [2, -1, 1], np.arcsin))
        #dm_body_stream['left_shoulder2'].append(AngleToAxes(l_shoulder_pos, l_elbow_pos, [0, 1, 1], np.arcsin))
        
        ######
        
        knee_quat = VecToQuat(body_stream[BodyPart.KNEE_R.value][i][3:7])
        
        knee_eul = QuatToEuler(knee_quat * np.quaternion(0, -0.707, -0, -0.707))
        dm_body_stream['right_hip_x'].append(-knee_eul[2])
        dm_body_stream['right_hip_y'].append(-knee_eul[0])
        dm_body_stream['right_hip_z'].append(knee_eul[1])
        
        ######
        
        knee_quat = VecToQuat(body_stream[BodyPart.KNEE_L.value][i][3:7])
        knee_eul = QuatToEuler(knee_quat * np.quaternion(0, 0.707, -0, -0.707))
        
        dm_body_stream['left_hip_x'].append(knee_eul[2])
        dm_body_stream['left_hip_y'].append(-knee_eul[0])
        dm_body_stream['left_hip_z'].append(-knee_eul[1])
        
        ######
        
        dm_body_stream['right_knee'].append(-AngleBetweenBodyPos(r_hip_pos, r_knee_pos, r_ankle_pos))
        dm_body_stream['left_knee'].append(-AngleBetweenBodyPos(l_hip_pos, l_knee_pos, l_ankle_pos))
        dm_body_stream['right_elbow'].append(AngleBetweenBodyPos(r_shoulder_pos, r_elbow_pos, r_wrist_pos) - np.pi/2)
        dm_body_stream['left_elbow'].append(AngleBetweenBodyPos(l_shoulder_pos, l_elbow_pos, l_wrist_pos) - np.pi/2)
        
    return dm_body_stream
    
def AngleBetweenQuat(quat1, quat2):
    knee_rot = (quat1.inverse() * unit_quat) * quat1
    ankle_rot = (quat2.inverse() * unit_quat) * quat2
    knee_rot_vec = np.array([knee_rot.w, knee_rot.x, knee_rot.y, knee_rot.z])
    ankle_rot_vec = np.array([ankle_rot.w, ankle_rot.x, ankle_rot.y, ankle_rot.z])
    angle = np.arccos(knee_rot_vec.dot(ankle_rot_vec)/(np.linalg.norm(knee_rot_vec)*np.linalg.norm(ankle_rot_vec)))
    return angle

def AngleBetweenBodyPos(pos0, pos1, pos2):
    bone1 = pos0 - pos1
    bone2 = pos1 - pos2
    
    bone1 = bone1 / la.norm(bone1)
    bone2 = bone2 / la.norm(bone2)

    # Depending on if you consider the angle inside or outside the lines
    angle = np.arccos(bone1.dot(bone2))
    #angle = np.pi - np.arccos(bone1.dot(bone2))
    
    return angle
    
def AngleToAxes(pos0, pos1, axes, arctrig):
    bone = pos1 - pos0
    bone_dir = bone / la.norm(bone)
    
    np_axes = np.array(axes)
    np_axes = np_axes / la.norm(np_axes)
    
    #angle = np.arccos(bone_dir.dot(np_axes))
    angle = arctrig(bone_dir.dot(np_axes))
    return angle

def VecToQuat(vec):
    #NEED TO ASSERT THAT IT IS DIMENSION 4
    return np.quaternion(vec[0], vec[1], vec[2], vec[3])

# If you're flying a plane (x is forward, y is up, z is to the right)
# Heading is NESW (yaw)
# Attitude is up/down (pitch)
# Banking is twisting (roll)
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

def Test_QuatToEuler():
    quat = np.quaternion(0.707, 0.707, 0, 0)
    eul = QuatToEuler(quat)
    assert(abs(eul[0] - np.pi/2) < 0.001)
    assert(eul[1] == 0)
    assert(eul[2] == 0)
    
    quat = np.quaternion(0.707, 0, 0.707, 0)
    eul = QuatToEuler(quat)
    assert(eul[0] == 0)
    assert(abs(eul[1] - np.pi/2) < 0.001)
    assert(eul[2] == 0)
    
    quat = np.quaternion(0.707, 0, 0, 0.707)
    eul = QuatToEuler(quat)
    assert(eul[0] == 0)
    assert(eul[1] == 0)
    assert(abs(eul[2] - np.pi/2) < 0.001)

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

unit_vec = np.array([0, 0, 1])
unit_quat = np.quaternion(0, 0, 0, 1)

##### LOAD MULTIPLE RECORDINGS FROM ./data #####

working_dir = os.path.abspath(__file__)
data_dir = os.path.join(working_dir, "../data")

all_files = glob.glob(data_dir + "/*.csv")
episodes = {}

for file in all_files:
    
    count = 0
    
    column_header = ""
    body_stream = []
    for i in range(25):
        body_stream.append([])

    with open(file, 'rt') as f:
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
        
    dm_body_stream = KinectToDMControl(body_stream)
    episodes[file.title()] = dm_body_stream

max_frame = 400

width = 480
height = 480
video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)
img = None

# Load one task:
env = suite.load(domain_name="humanoid", task_name="stand")

# MJCF forming humanoid task
#suite.humanoid.get_model_and_assets()

## Iterate over a task set:
#for domain_name, task_name in suite.BENCHMARKING:
#  env = suite.load(domain_name, task_name)

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
action = np.zeros_like(action_spec.minimum)
time_step = env.reset()

##### BODY PART DICT -> SIMULATION STATES CONVERSION #####

for ep in episodes:
    
    current_states = []
    desired_states = []
    
    episode = episodes[ep]

    recording_length = len(next(iter(episode.values())))
    
    for i in range(recording_length-1):
        with env.physics.reset_context():
            env.physics.named.data.qpos[:] = 0
            temp = env.physics.named.data.qpos['root']
            temp[2] = 20
            env.physics.named.data.qpos['root'] = temp
            
            for part_key, part_state in episode.items():
                env.physics.named.data.qpos[part_key] = part_state[i]
                
        current_states.append(np.array(env.physics.named.data.qpos.tolist()))
        
        if (i > 0):
            desired_states.append(np.array(env.physics.named.data.qpos.tolist()))
        
    current_states.pop() #remove last entry
    
    current_states = np.array(current_states)
    desired_states = np.array(desired_states)
    
    episodes[ep] = {"input": current_states, "label": desired_states}

##### FLATTENING SIMULATION STATES EPISODES INTO ONE DATASET #####

all_current_states = []
all_desired_states = []

for ep in episodes:
    episode = episodes[ep]
    
    recording_length = len(next(iter(episode.values())))
    
    for i in range(recording_length):
        all_current_states.append(episode["input"][i])
        all_desired_states.append(episode["label"][i])
        
all_current_states = np.array(all_current_states)
all_desired_states = np.array(all_desired_states)

##### NORMALISE THE DATA #####

current_states_mean = np.mean(all_current_states, axis=0)
current_states_std = np.std(all_current_states, axis=0)
all_current_states = (all_current_states - current_states_mean)/current_states_std
all_current_states = np.nan_to_num(all_current_states, copy=True)   # Removing NaN for fields with 0 variance

desired_states_mean = np.mean(all_desired_states, axis=0)
desired_states_std = np.std(all_desired_states, axis=0)
all_desired_states = (all_desired_states - current_states_mean)/current_states_std
all_desired_states = np.nan_to_num(all_desired_states, copy=True)   # Removing NaN for fields with 0 variance

##### SHUFFLE THE DATASET #####
#https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

all_current_states, all_desired_states = unison_shuffled_copies(all_current_states, all_desired_states)

##### BUILD THE MODEL #####
model = keras.Sequential([
    keras.layers.Dense(512, input_shape=(28,), activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(28, activation='linear')
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

##### TRAIN THE MODEL #####
model.fit(all_current_states, all_desired_states, epochs=100)

##### PREDICT ROLLOUT TRAJECTORY FROM FIRST STATE #####
states_stream = np.zeros([400, 28])
states_stream[0,:] = (model.predict(episode["input"][0:1]))

for i in range(399):
    states_stream[i+1] = model.predict(states_stream[i:i+1]) 
    
##### DENORMALISE THE PREDICTED TRAJECTORY #####
for i in range(400):
    states_stream[i] = (states_stream[i]+desired_states_mean)*desired_states_std

##### RENDER THE PREDICTED STATES WITHOUT SIMULATED PHYSICS #####
frames = 0
while not time_step.last():
  for i in range(max_frame-1):
    #time_step = env.reset()                                            #to reset the env every iteration (generating random states)
    
    #action = np.random.uniform(action_spec.minimum,
    #                         action_spec.maximum,
    #                         size=action_spec.shape)
    
    action = np.zeros_like(action_spec.minimum)     #no action
    
    # BETTER TO TEST ANGLES THIS WAY SO THE GUY ISNT SIMULATED
    with env.physics.reset_context():        
        env.physics.named.data.qpos[:] = states_stream[i]
    
        temp = env.physics.named.data.qpos['root']
        temp[2] = 20
        env.physics.named.data.qpos['root'] = temp
        
        #for part_key, part_state in dm_body_stream.items():
        #    env.physics.named.data.qpos[part_key] = part_state[frames]
    
    time_step = env.step(action)
    
    #print(env.physics.get_state())
    
    print(i)
    video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])
    #print(time_step.reward, time_step.discount, time_step.observation)
    
    frames += 1
    
    if (frames >= len(states_stream)):
        break
  
  print("DISPLAYING BATCH OF VIDEO")
  for i in range(max_frame):
    cv2.imshow('Frame',video[i])
    key = cv2.waitKey(int(1000/30))
  keyboard_input = input("Type q to end (or press enter to continue): ")
  if (keyboard_input == "q"):
      break
plt.close()