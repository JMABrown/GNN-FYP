
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
import sonnet as snt
import glob
import os
import random
import copy

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

from graph_nets.demos import models

BATCH_SIZE = 1
END_EP_FRAMES_CUT = 60
START_EP_FRAMES_CUT = 30
NUM_EPOCHS = 10
STARTING_FRAME = 30
ROLLOUT_START = 0
ROLLOUT_LEN = 100
PLAYBACK_FRAMERATE = 30
MAX_N_STEPS_PRED = 10
HEIGHT_SUSPENSION = 20
PROCESSING_STEPS_TR = 2
OUTPUT_EDGE_SIZE = 1
OUTPUT_NODE_SIZE = 1
OUTPUT_GLOBAL_SIZE = 1
START_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 3.0
ROLLOUT_EPISODE = 0

np.random.seed(3103)

body_names = ['torso', 'head', 'lower_waist',
              'pelvis', 'right_thigh', 'right_shin',
              'right_foot', 'left_thigh', 'left_shin',
              'left_foot', 'right_upper_arm', 'right_lower_arm',
              'right_hand', 'left_upper_arm', 'left_lower_arm',
              'left_hand']

joint_names = ['abdomen_z', 'abdomen_y', 'abdomen_x',
              'right_hip_x', 'right_hip_z', 'right_hip_y',
              'right_knee',
              'right_ankle_y', 'right_ankle_x',
              'left_hip_x', 'left_hip_z', 'left_hip_y',
              'left_knee',
              'left_ankle_y', 'left_ankle_x',
              'right_shoulder1', 'right_shoulder2',
              'right_elbow',
              'left_shoulder1', 'left_shoulder2',
              'left_elbow']

def DataDictFromEnv(current_env):
    
    global_features = [1.0]
    
    nodes = []

    for body_part in body_names:
        nodes.append([env.physics.named.model.body_mass[body_part],
                          env.physics.named.model.body_inertia[body_part][0],       #inertia isnt exactly necessary but it's there to demonstrate multiple features
                          env.physics.named.model.body_inertia[body_part][1],
                          env.physics.named.model.body_inertia[body_part][2]])
    
    edges = []
    senders = []
    receivers = []
    #for joint in range(1, num_joints):
    for joint in joint_names:
        edges.append(env.physics.named.data.qpos[joint])
        child_body_id = env.physics.named.model.jnt_bodyid[joint]
        senders.append(child_body_id - 1)       #-1 due to not including root
        parent_body_id = env.physics.named.model.body_parentid[child_body_id]
        receivers.append(parent_body_id - 1)    #-1 due to not including root
        
        # Make it bidirectional
        edges.append(env.physics.named.data.qpos[joint])
        senders.append(parent_body_id - 1)
        receivers.append(child_body_id - 1)
        
    data_dict = {
            "globals": np.array(global_features).astype(np.float32),
            "nodes": np.array(nodes).astype(np.float32),
            "edges": np.array(edges).astype(np.float32),
            "senders": np.array(senders).astype(np.int32),
            "receivers": np.array(receivers).astype(np.int32)
    }
    
    return data_dict

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
    
unit_quat = np.quaternion(1, 0, 0, 0)
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
    
    quat1 = np.quaternion(0.707, 0.707, 0, 0)
    quat2 = np.quaternion(0.707, 0, 0.707, 0)
    quat = quat2*quat1
    eul = QuatToEuler(quat)
    print(eul)
    assert(abs(eul[0] - np.pi/2) < 0.001)
    assert(abs(eul[1] - np.pi/2) < 0.001)
    assert(eul[2] == 0)
    
    quat1 = np.quaternion(0.707, 0.0, 0.707, 0)
    quat2 = np.quaternion(0.707, 0.707, 0.0, 0)
    quat = quat2*quat1
    eul = QuatToEuler(quat)
    print(eul)
    assert(eul[0] == 0)
    assert(abs(eul[1] - np.pi/2) < 0.001)
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
img = None

# Load one task:
env = suite.load(domain_name="humanoid", task_name="stand")

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
    
    for i in range(START_EP_FRAMES_CUT, recording_length-1-END_EP_FRAMES_CUT):
        with env.physics.reset_context():
            env.physics.named.data.qpos[:] = 0
            temp = env.physics.named.data.qpos['root']
            temp[2] = HEIGHT_SUSPENSION
            env.physics.named.data.qpos['root'] = temp
            
            for part_key, part_state in episode.items():
                env.physics.named.data.qpos[part_key] = part_state[i]
                
        #current_states.append(np.array(env.physics.named.data.qpos.tolist()))
        current_states.append(DataDictFromEnv(env))
        
        if (i > 0):
            #desired_states.append(np.array(env.physics.named.data.qpos.tolist()))
            desired_states.append(DataDictFromEnv(env))
        
    current_states.pop() #remove last entry for ALL episodes
    
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

def NormaliseGraphAttribute(graphs_list, attribute, attribute_mean = None, attribute_std = None):
    
    num_graphs = graphs_list.size
    normalised_graphs_list = copy.deepcopy(graphs_list)
    
    if (attribute_mean is None):
        attribute_sum = np.zeros_like(graphs_list[0][attribute])
        for g in normalised_graphs_list:
            attribute_sum += g[attribute]
            
        attribute_mean = attribute_sum / num_graphs
     
    if (attribute_std is None):
        attribute_stds = []
        for g in normalised_graphs_list:
            attribute_stds.append(g[attribute] - attribute_mean)
            
        attribute_stds = np.array(attribute_stds)
        attribute_stds = attribute_stds**2
        attribute_std = (np.sum(attribute_stds, axis = 0) / num_graphs)**0.5
        
    for i in range(num_graphs):
        normalised_graphs_list[i][attribute] = (normalised_graphs_list[i][attribute] - attribute_mean) / attribute_std
        normalised_graphs_list[i][attribute] = np.nan_to_num(normalised_graphs_list[i][attribute])
        
    return normalised_graphs_list, attribute_mean, attribute_std

#num_graphs = all_current_states.size
#edges_sum = np.zeros_like(all_current_states[0]['edges'])
#edges_std_sum = np.zeros_like(all_current_states[0]['edges'])
#for g in all_current_states:
#    edges_sum += g['edges']
#    
#edges_mean = edges_sum / num_graphs
#    
#edges_stds = []
#for g in all_current_states:
#    edges_stds.append(g['edges'] - edges_mean)
#    
#edges_stds = np.array(edges_stds)
#edges_stds = edges_stds**2
#edges_std = (np.sum(edges_stds, axis = 0) / num_graphs)**0.5
#
#for i in range(num_graphs):
#    all_current_states[i]['edges'] = (all_current_states[i]['edges'] - edges_mean) / edges_std
#    all_current_states[i]['edges'] = np.nan_to_num(all_current_states[i]['edges'])

#all_current_states, nodes_mean, nodes_std = NormaliseGraphAttribute(all_current_states, 'nodes')   #Leave these unnormalised as they are all the same
all_current_states, edges_mean, edges_std = NormaliseGraphAttribute(all_current_states, 'edges')

all_desired_states, edges_mean, edges_std = NormaliseGraphAttribute(all_desired_states, 'edges')

normalised_episodes = {}
for ep in episodes:
    normalised_episode = copy.deepcopy(episodes[ep])
    episode_input, _, _ =  NormaliseGraphAttribute(normalised_episode['input'], 'edges', edges_mean, edges_std)
    episode_label, _, _ =  NormaliseGraphAttribute(normalised_episode['label'], 'edges', edges_mean, edges_std)
    normalised_episodes[ep] = {"input": episode_input, "label": episode_label}

nan_count = 0
for state in all_current_states:
    nan_count += np.sum(np.isnan(state['edges']))
    nan_count += np.sum(np.isnan(state['nodes']))
    nan_count += np.sum(np.isnan(state['globals']))
assert(nan_count == 0)

#current_states_mean = np.mean(all_current_states, axis=0)
#current_states_std = np.std(all_current_states, axis=0)
#all_current_states = (all_current_states - current_states_mean)/current_states_std
#all_current_states = np.nan_to_num(all_current_states, copy=True)   # Removing NaN for fields with 0 variance

#desired_states_mean = np.mean(all_desired_states, axis=0)
#desired_states_std = np.std(all_desired_states, axis=0)
#all_desired_states = (all_desired_states - current_states_mean)/current_states_std
#all_desired_states = np.nan_to_num(all_desired_states, copy=True)   # Removing NaN for fields with 0 variance

##### SHUFFLE THE DATASET #####
#https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

all_current_states, all_desired_states = unison_shuffled_copies(all_current_states, all_desired_states)


"""class MyMLP(snt.Module):
    def __init__(self, name=None):
        super(MyMLP, self).__init__(name=name)
        self.hidden1 = snt.Linear(1024, name="hidden1")
        self.output = snt.Linear(1, name="output")
        
    def __call__(self, x):
        x = self.hidden1(x)
        x = tf.nn.relu(x)
        x = self.output(x)
        return x"""


# Create the model
tf.reset_default_graph()

"""graph_network = modules.GraphNetwork(
    #edge_model_fn=lambda: snt.Linear(output_size=OUTPUT_EDGE_SIZE),
    edge_model_fn=lambda: snt.nets.MLP([256, 256, 1]),
    #edge_model_fn=lambda: MyMLP("edge_gnn"),
    #node_model_fn=lambda: snt.Linear(output_size=OUTPUT_NODE_SIZE),
    node_model_fn=lambda: snt.nets.MLP([256, 256, 1]),
    #global_model_fn=lambda: snt.Linear(output_size=OUTPUT_GLOBAL_SIZE))
    global_model_fn=lambda: snt.nets.MLP([256, 256, 1]))"""

graph_network = models.EncodeProcessDecode(edge_output_size=1)

def concat_in_list(numpy_list):
    return np.concatenate([nl for nl in numpy_list], axis = -1)

# Create placeholders from current states and desired states
graphs_tuple_ph = utils_tf.placeholders_from_data_dicts(all_current_states[0:BATCH_SIZE])
training_desired_ph = utils_tf.placeholders_from_data_dicts(all_desired_states[0:BATCH_SIZE])

# Initialise other tensorflow variables
epoch_ph = tf.placeholder(tf.float32)
start_learning_rate = tf.constant(START_LEARNING_RATE)
learning_rate_decay = tf.constant(LEARNING_RATE_DECAY)
dynamic_learning_rate = start_learning_rate / (1.0 + epoch_ph/learning_rate_decay)

# Pass placeholder of current state to graph to make a prediction
graph_predictions = graph_network(graphs_tuple_ph, PROCESSING_STEPS_TR)

# Loss is MSE between edges of current graph and predicted graph
# Edges are the angles of the DoF for the model
def custom_loss_ops(target_graph, predicted_graphs):
    loss_ops = [ 
            tf.losses.mean_squared_error(target_graph.edges, predicted_graph.edges)
            for predicted_graph in predicted_graphs
            ]
    return loss_ops

# Loss is divided across the batch
losses = custom_loss_ops(training_desired_ph, graph_predictions)

loss = sum(losses) / PROCESSING_STEPS_TR

#loss = custom_loss_ops(training_desired_ph, graph_prediction)

# Training optimiser to reduce the loss
optimiser = tf.train.AdamOptimizer(learning_rate=dynamic_learning_rate).minimize(loss)
#optimiser = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

def NStepError(all_step_preds, desired_labels, N):
        n_step_pred = [step[N-1] for step in all_pred_steps]
        n_step_error = [(n_step_pred[i][-1]['edges'] - desired_labels[i+(N-1)]['edges'])**2 for i in range(len(n_step_pred)-(N-1))]
        mean_n_step_error_bodypart = np.mean(n_step_error, axis = 0)
        mean_n_step_error_frame = np.mean(n_step_error, axis = 1)
        mean_n_step_error = np.mean(n_step_error)
        return mean_n_step_error, mean_n_step_error_frame, mean_n_step_error_bodypart
    
# Function to collapse the bidirection edges into a single value
def GraphsTupleToQpos(graphs_tuple, mean, std, env_len):
    edges = utils_np.graphs_tuple_to_data_dicts(graphs_tuple)[0]['edges']
    # Initialise an array with all zeros (which will initialise the root position to 0)
    qpos = [0]*env_len
    for i in range(int(len(edges)/2)):
        # edges are copied over as a mean of edges going both ways
        # +7 is required to skip over the root pos and quat
        qpos[i+7] = (edges[i*2] + edges[(i*2)+1])/2
        qpos[i+7] = (qpos[i+7]*std[i+7]) + mean[i+7]
    return np.array(qpos).flatten()

def EdgesArrayToQpos(edges_array, env_len):
    qpos = [0]*env_len
    for i in range(int(len(edges_array)/2)):
        qpos[i+7] = (edges_array[i*2] + edges_array[(i*2)+1])/2
    return np.array(qpos).flatten()

rollout_ep = list(normalised_episodes.keys())[ROLLOUT_EPISODE]
normalised_episode = normalised_episodes[rollout_ep]
episode = episodes[rollout_ep]

# START THE TENSORFLOW SESSION
with tf.Session() as sess:
    # Initialise all constants eg weights in GNN
    sess.run(tf.global_variables_initializer())
    
    # Number of batches based on number of samples and batch size
    num_iter = np.floor((len(all_current_states)-1)/BATCH_SIZE)
    num_iter = np.int(num_iter)
    
    # Arrays to store results
    preds = []
    
    # Each training epoch
    for epoch in range(NUM_EPOCHS):
        recorded_losses = []
        
        print("EPOCH: {0}".format(epoch))
        
        # DATA SHUFFLE EACH EPOCH
        #combined = list(zip(all_current_states, all_desired_states))
        #random.shuffle(combined)
        #all_current_states, all_desired_states = zip(*combined)
        
        # TRAINING
        for i in range(num_iter):
            
            # Form the feed ditionary
            # This is how the values are delivered into the variables of the DFG
            # Here, a batch from the list of current states and desired states is delivered into the DFG
            feed_dict = {graphs_tuple_ph: utils_np.data_dicts_to_graphs_tuple(all_current_states[i*BATCH_SIZE:(i+1)*BATCH_SIZE]),
                         training_desired_ph: utils_np.data_dicts_to_graphs_tuple(all_desired_states[i*BATCH_SIZE:(i+1)*BATCH_SIZE]),
                         epoch_ph: float(epoch)}
            
            # 1st argument is the values which are returned into train_values as a dictionary
            # 2nd argument is to pass in the feed dict
            # The values to return and resolve must be calculable from the feed dict
            train_values = sess.run({"optimiser": optimiser,
                                     "loss": loss,
                                     "preds": graph_predictions}, feed_dict=feed_dict)
            
            # Store the loss into a list
            recorded_losses.append(train_values["loss"])
            preds.append(train_values["preds"])
            
            #percentage complete of epoch
            if (i % int(num_iter/5) == 0):
                print("% done: {0} loss: {1}".format(i/num_iter, np.mean(recorded_losses)))
            
    ##### ROLLOUT PREDICITON #####
    predicted_path = []
    
    starting_position = episode['input'][ROLLOUT_START:ROLLOUT_START+1].copy()
    
    feed_dict = {graphs_tuple_ph: utils_np.data_dicts_to_graphs_tuple(starting_position)}
    predicted_next_step = sess.run(graph_predictions, feed_dict)
    predicted_path.append(predicted_next_step[-1])
    
    while (len(predicted_path) < ROLLOUT_LEN):
        current_step = predicted_path[-1]
        current_step = utils_np.graphs_tuple_to_data_dicts(current_step)
        current_step[0]['nodes'] = starting_position[0]['nodes'].copy()
        current_step[0]['globals'] = starting_position[0]['globals'].copy()
        feed_dict = {graphs_tuple_ph: utils_np.data_dicts_to_graphs_tuple(current_step)}
        predicted_next_steps = sess.run(graph_predictions, feed_dict)
        predicted_path.append(predicted_next_steps[-1])
        
    predicted_path_qpos = []
        
    episode_len = len(episode["label"])
    plt.figure()
    plt.plot([EdgesArrayToQpos(episode["label"][i]['edges'], 28)[22][0] for i in range(episode_len)], label = "right_shoulder1")
    plt.plot([EdgesArrayToQpos(episode["label"][i]['edges'], 28)[23][0] for i in range(episode_len)], label = "right_shoulder2")
    plt.plot([EdgesArrayToQpos(episode["label"][i]['edges'], 28)[24][0] for i in range(episode_len)], label = "right_elbow")
    plt.xlabel('Frame')
    plt.ylabel('Rotation (radians)')
    plt.legend()
    
    episode_len = len(episode["label"])
    plt.figure()
    plt.plot([GraphsTupleToQpos(predicted_path[i], edges_mean, edges_std, 28)[22] for i in range(len(predicted_path))], label = "right_shoulder1")
    plt.plot([GraphsTupleToQpos(predicted_path[i], edges_mean, edges_std, 28)[23] for i in range(len(predicted_path))], label = "right_shoulder2")
    plt.plot([GraphsTupleToQpos(predicted_path[i], edges_mean, edges_std, 28)[24] for i in range(len(predicted_path))], label = "right_elbow")
    plt.xlabel('Frame')
    plt.ylabel('Rotation (radians)')
    plt.legend()
        
    ##### N STEP PREDICTION #####
    all_pred_steps = []
    # Input must be normalised
    for i in range(len(normalised_episode["input"])):
        
        pred_steps = []
        
        starting_position = normalised_episode["input"][i:i+1].copy()
    
        feed_dict = {graphs_tuple_ph: utils_np.data_dicts_to_graphs_tuple(starting_position)}
        predicted_next_step = sess.run(graph_predictions, feed_dict)
        pred_steps.append(utils_np.graphs_tuple_to_data_dicts(predicted_next_step[-1]))
        
        while (len(pred_steps) < MAX_N_STEPS_PRED):
            current_step = pred_steps[-1]
            current_step[0]['nodes'] = starting_position[0]['nodes'].copy()
            current_step[0]['globals'] = starting_position[0]['globals'].copy()
            feed_dict = {graphs_tuple_ph: utils_np.data_dicts_to_graphs_tuple(current_step)}
            predicted_next_steps = sess.run(graph_predictions, feed_dict)
            pred_steps.append(utils_np.graphs_tuple_to_data_dicts(predicted_next_steps[-1]))
            
        # Denormalise the output
        #print("{0} BEFORE: {1}".format(i, pred_steps[0][0]['edges']))
        for j in range(MAX_N_STEPS_PRED):
            pred_steps[j][-1]['edges'] = pred_steps[j][-1]['edges']*edges_std + edges_mean
        #print("{0} AFTER: {1}".format(i, pred_steps[0][0]['edges']))
        
        all_pred_steps.append(copy.deepcopy(pred_steps))
        
    # Comparing results must be denormalised
    mean_one_step_error, mean_one_step_error_frame, mean_one_step_error_bodypart = NStepError(all_pred_steps, episode["label"], 1)
    mean_three_step_error, mean_three_step_error_frame, mean_three_step_error_bodypart = NStepError(all_pred_steps, episode["label"], 3)
    mean_five_step_error, mean_five_step_error_frame, mean_five_step_error_bodypart = NStepError(all_pred_steps, episode["label"], 5)
    mean_ten_step_error, mean_ten_step_error_frame, mean_ten_step_error_bodypart = NStepError(all_pred_steps, episode["label"], 10)
    
mean_one_step_error_bodypart = EdgesArrayToQpos(mean_one_step_error_bodypart.flatten(), 28)
mean_three_step_error_bodypart = EdgesArrayToQpos(mean_three_step_error_bodypart.flatten(), 28)
mean_five_step_error_bodypart = EdgesArrayToQpos(mean_five_step_error_bodypart.flatten(), 28)
mean_ten_step_error_bodypart = EdgesArrayToQpos(mean_ten_step_error_bodypart.flatten(), 28)

plt.figure()
plt.plot(mean_one_step_error_frame, label="1-step")
plt.plot(mean_three_step_error_frame, label="3-step")
plt.plot(mean_five_step_error_frame, label="5-step")
plt.plot(mean_ten_step_error_frame, label="10-step")
#plt.yscale("log")
plt.xlabel('Frame')
plt.ylabel('MSE')
plt.legend()
plt.show()

qpos_names = ['root_pos_x', 'root_pos_y', 'root_pos_z',
              'root_quat_w', 'root_quat_x', 'root_quat_y', 'root_pos_z',
              'abdomen_z', 'abdomen_y', 'abdomen_x',
              'right_hip_x', 'right_hip_z', 'right_hip_y',
              'right_knee',
              'right_ankle_y', 'right_ankle_x',
              'left_hip_x', 'left_hip_z', 'left_hip_y',
              'left_knee',
              'left_ankle_y', 'left_ankle_x',
              'right_shoulder1', 'right_shoulder2',
              'right_elbow',
              'left_shoulder1', 'left_shoulder2',
              'left_elbow']
#bar_width = 0.15
#spacing = 0.15
#plt.bar(np.arange(len(mean_one_step_error_bodypart)), tick_label = qpos_names, height = mean_one_step_error_bodypart, width=bar_width)
#plt.bar(np.arange(len(mean_three_step_error_bodypart)) - spacing, tick_label = qpos_names, height = mean_three_step_error_bodypart, width=bar_width)
#plt.bar(np.arange(len(mean_five_step_error_bodypart)) - spacing*2, tick_label = qpos_names, height = mean_five_step_error_bodypart, width=bar_width)
#plt.bar(np.arange(len(mean_ten_step_error_bodypart)) - spacing*3, tick_label = qpos_names, height = mean_ten_step_error_bodypart, width=bar_width)
#plt.xticks(rotation=60)
#plt.yscale("log")
#plt.show()

plt.figure()
bar_width = 0.3
plt.bar(np.arange(len(mean_one_step_error_bodypart)), tick_label = qpos_names, height = mean_one_step_error_bodypart, width=bar_width)
plt.bar(np.arange(len(mean_three_step_error_bodypart)) - bar_width, tick_label = qpos_names, height = mean_three_step_error_bodypart, width=bar_width)
plt.xticks(rotation=60)
plt.legend(["1-step", "3-step"])
plt.show()

plt.figure()
bar_width = 0.3
plt.bar(np.arange(len(mean_three_step_error_bodypart)), tick_label = qpos_names, height = mean_three_step_error_bodypart, width=bar_width)
plt.bar(np.arange(len(mean_five_step_error_bodypart)) - bar_width, tick_label = qpos_names, height = mean_five_step_error_bodypart, width=bar_width)
plt.bar(np.arange(len(mean_ten_step_error_bodypart)) - bar_width*2, tick_label = qpos_names, height = mean_ten_step_error_bodypart, width=bar_width)
plt.xticks(rotation=60)
plt.legend(["3-step", "5-step", "10-step"])
plt.show()



for i, step in enumerate(predicted_path):
    
    action = np.zeros_like(action_spec.minimum)

    with env.physics.reset_context():        
        env.physics.named.data.qpos[:] = GraphsTupleToQpos(step, edges_mean, edges_std, len(env.physics.named.data.qpos.tolist()))
    
        temp = env.physics.named.data.qpos['root']
        temp[2] = HEIGHT_SUSPENSION
        env.physics.named.data.qpos['root'] = temp

    time_step = env.step(action)
    
    video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
                          env.physics.render(height, width, camera_id=1)])

# Function actually allows for showing the same video multiple times
def DisplayVideo(video):
    max_frame = len(video)
    
    while (True):
        print("DISPLAYING BATCH OF VIDEO")
        for i in range(max_frame):
            cv2.imshow('Frame',video[i])
            cv2.waitKey(int(1000/PLAYBACK_FRAMERATE))
        keyboard_input = input("Type q to end (or press enter to continue): ")
        if (keyboard_input == "q"):
            break
    plt.close()
    
DisplayVideo(video)
