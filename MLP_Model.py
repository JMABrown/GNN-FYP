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

EP_FRAMES_CUT = 60
NUM_EPOCHS = 10
NUM_NEURONS = 512
STARTING_FRAME = 30
ROLLOUT_LEN = 100
PLAYBACK_FRAMERATE = 30
MAX_N_STEPS_PRED = 10
ROLLOUT_EPISODE = 0

np.random.seed(3103)

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
        
        dm_body_stream['right_shoulder1'].append(AngleToAxes(r_shoulder_pos, r_elbow_pos, [2, 1, 1], np.arccos) - np.pi/2)
        dm_body_stream['right_shoulder2'].append(AngleToAxes(r_shoulder_pos, r_elbow_pos, [0, -1, 1], np.arccos) - np.pi/2)
        
        dm_body_stream['left_shoulder1'].append(AngleToAxes(l_shoulder_pos, l_elbow_pos, [2, -1, 1], np.arccos) - np.pi/2)
        dm_body_stream['left_shoulder2'].append(AngleToAxes(l_shoulder_pos, l_elbow_pos, [0, 1, 1], np.arccos) - np.pi/2)
        
        ######
        
        knee_quat = VecToQuat(body_stream[BodyPart.KNEE_R.value][i][3:7])
        
        knee_eul = QuatToEuler(knee_quat * np.quaternion(0, -0.707, 0, -0.707))
        dm_body_stream['right_hip_x'].append(-knee_eul[2])
        dm_body_stream['right_hip_y'].append(-knee_eul[0])
        dm_body_stream['right_hip_z'].append(knee_eul[1])
        
        ######
        
        knee_quat = VecToQuat(body_stream[BodyPart.KNEE_L.value][i][3:7])
        knee_eul = QuatToEuler(knee_quat * np.quaternion(0, 0.707, 0, -0.707))
        
        dm_body_stream['left_hip_x'].append(knee_eul[2])
        dm_body_stream['left_hip_y'].append(-knee_eul[0])
        dm_body_stream['left_hip_z'].append(-knee_eul[1])
        
        ######
        
        dm_body_stream['right_knee'].append(-AngleBetweenBodyPos(r_hip_pos, r_knee_pos, r_ankle_pos))
        dm_body_stream['left_knee'].append(-AngleBetweenBodyPos(l_hip_pos, l_knee_pos, l_ankle_pos))
        dm_body_stream['right_elbow'].append(AngleBetweenBodyPos(r_shoulder_pos, r_elbow_pos, r_wrist_pos) - np.pi/2)
        dm_body_stream['left_elbow'].append(AngleBetweenBodyPos(l_shoulder_pos, l_elbow_pos, l_wrist_pos) - np.pi/2)
        
    return dm_body_stream

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

width = 480
height = 480
video = np.zeros((ROLLOUT_LEN, height, 2 * width, 3), dtype=np.uint8)

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
    
    for i in range(recording_length-1-EP_FRAMES_CUT):
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

##### NORMALISED EPISODES #####

normalised_episodes = {}
for ep in episodes:
    episode = episodes[ep]
    episode_input = (episode['input'].copy() - current_states_mean)/current_states_std
    episode_input = np.nan_to_num(episode_input)
    episode_label = (episode['label'].copy() - current_states_mean)/current_states_std
    episode_label = np.nan_to_num(episode_label)
    normalised_episodes[ep] = {"input": episode_input, "label": episode_label}   

##### SHUFFLE THE DATASET #####
#https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

all_current_states, all_desired_states = unison_shuffled_copies(all_current_states, all_desired_states)

##### BUILD THE MODEL #####
#model = keras.Sequential([
#    keras.layers.Dense(512, input_shape=(28,), kernel_regularizer=keras.regularizers.l2(0.01), activation='relu'),
#    keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l2(0.01), activation='relu'),
#    keras.layers.Dense(28, kernel_regularizer=keras.regularizers.l2(0.01), activation='linear')
#])

model = keras.Sequential([
    keras.layers.Dense(NUM_NEURONS, input_shape=(28,), activation='relu'),
    keras.layers.Dense(NUM_NEURONS, activation='relu'),
    keras.layers.Dense(28, activation='linear')
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

##### TRAIN THE MODEL #####
model.fit(all_current_states, all_desired_states, epochs=NUM_EPOCHS)

##### PREDICTION AND TESTING EPISODE #####
episode = normalised_episodes[list(normalised_episodes.keys())[ROLLOUT_EPISODE]]

##### PREDICT ROLLOUT TRAJECTORY FROM FIRST STATE #####
states_stream = np.zeros([ROLLOUT_LEN, 28])
starting_state = episode["input"][STARTING_FRAME:STARTING_FRAME+1]
states_stream[0,:] = model.predict(starting_state)

for i in range(ROLLOUT_LEN-1):
    states_stream[i+1] = model.predict(states_stream[i:i+1])
    
##### DENORMALISE THE PREDICTED TRAJECTORY #####
for i in range(ROLLOUT_LEN):
    states_stream[i] = (states_stream[i]+desired_states_mean)*desired_states_std

##### RENDER THE PREDICTED STATES WITHOUT SIMULATED PHYSICS #####
frames = 0
while not time_step.last():
  for i in range(ROLLOUT_LEN-1):

    
    action = np.zeros_like(action_spec.minimum)     #no action
    
    # BETTER TO TEST ANGLES THIS WAY SO THE GUY ISNT SIMULATED
    with env.physics.reset_context():        
        env.physics.named.data.qpos[:] = states_stream[i]
    
        temp = env.physics.named.data.qpos['root']
        temp[2] = 20
        env.physics.named.data.qpos['root'] = temp
    
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
  for i in range(ROLLOUT_LEN):
    cv2.imshow('Frame',video[i])
    key = cv2.waitKey(int(1000/PLAYBACK_FRAMERATE))
  keyboard_input = input("Type q to end (or press enter to continue): ")
  if (keyboard_input == "q"):
      break
cv2.destroyAllWindows()

##### 1 STEP, 3 STEP, 5 STEP PREDICTION #####
pred_steps = np.zeros([MAX_N_STEPS_PRED, 28])
all_pred_steps = []
for i in range(len(episode["input"])):
    pred_steps[0] = model.predict(episode["input"][i:i+1])
    for step in range(MAX_N_STEPS_PRED-1):
        pred_steps[step+1] = model.predict(pred_steps[step:step+1])
    all_pred_steps.append(pred_steps.copy())
    
def NStepError(all_step_preds, desired_labels, N):
    n_step_pred = [step[N-1] for step in all_pred_steps]
    n_step_error = [(n_step_pred[i] - desired_labels[i+(N-1)])**2 for i in range(len(n_step_pred)-(N-1))]
    mean_n_step_error_bodypart = np.mean(n_step_error, axis = 0)
    mean_n_step_error_frame = np.mean(n_step_error, axis = 1)
    mean_n_step_error = np.mean(n_step_error)
    return mean_n_step_error, mean_n_step_error_frame, mean_n_step_error_bodypart
    
mean_one_step_error, mean_one_step_error_frame, mean_one_step_error_bodypart = NStepError(all_pred_steps, episode["label"], 1)
mean_three_step_error, mean_three_step_error_frame, mean_three_step_error_bodypart = NStepError(all_pred_steps, episode["label"], 3)
mean_five_step_error, mean_five_step_error_frame, mean_five_step_error_bodypart = NStepError(all_pred_steps, episode["label"], 5)
mean_ten_step_error, mean_ten_step_error_frame, mean_ten_step_error_bodypart = NStepError(all_pred_steps, episode["label"], 10)

plt.plot(mean_one_step_error_frame)
plt.plot(mean_three_step_error_frame)
plt.plot(mean_five_step_error_frame)
plt.plot(mean_ten_step_error_frame)
#plt.yscale("log")
plt.xlabel('Frame')
plt.ylabel('MSE')
plt.show()

#ViewEpisode(episodes[all_files[4].title()]['input'])
def ViewEpisode(episode):
    
    env = suite.load(domain_name="humanoid", task_name="stand")
    action_spec = env.action_spec()
    action = np.zeros_like(action_spec.minimum)
    time_step = env.reset()
    
    max_frame = len(episode)
    
    video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)
    
    for i, frame in enumerate(episode):
        
        # BETTER TO TEST ANGLES THIS WAY SO THE GUY ISNT SIMULATED
        with env.physics.reset_context():        
            env.physics.named.data.qpos[:] = frame
            
            temp = env.physics.named.data.qpos['root']
            temp[2] = 20
            env.physics.named.data.qpos['root'] = temp
            
        time_step = env.step(action)
        
        print(i)
        video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                              env.physics.render(height, width, camera_id=1)])
    
    print("DISPLAYING BATCH OF VIDEO")
    for i in range(max_frame):
        cv2.imshow('Frame',video[i])
        key = cv2.waitKey(int(1000/PLAYBACK_FRAMERATE))
    cv2.destroyAllWindows()
