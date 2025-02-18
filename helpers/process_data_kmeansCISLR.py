from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
import numpy as np
import os
import sys
import random  
import pickle

root_folder = "/DATA7/vaibhav/isign/Data/iSign-poses_v1.1"

def find_nearest_unmasked_frame(data, confidence_measure, start_idx):
    """
    Find the nearest unmasked frame (one without any masked values) starting from the given index.
    """
    # Look forward and backward let's make a confidence interval of 0.8 as well.
    for offset in range(1, len(data)):
        forward_idx = start_idx + offset
        backward_idx = start_idx - offset

        # Check forward 
        if forward_idx < len(data) and not np.any(data.mask[forward_idx]) and np.min(confidence_measure[forward_idx]) > 0.8:
            return forward_idx

        # Check backward
        if backward_idx >= 0 and not np.any(data.mask[backward_idx]) and np.min(confidence_measure[backward_idx]) > 0.8:
            return backward_idx

    # If no unmasked frame is found (unlikely in typical datasets)
    return start_idx
    raise ValueError('No nearest unmasked frame found in the data.')

def sample_unmasked_frames(data, confidence_measure, step=10):
    """
    Sample every `step` frame, but replace frames with masked values with the nearest unmasked frame.
    """
    sampled_frames = []
    confidence_frames = []
    
    for i in range(0, len(data), step):
        if np.any(data.mask[i]):  # Check if current frame has any masked values
            # Find the nearest frame without any masks
            nearest_idx = find_nearest_unmasked_frame(data, confidence_measure, i)
            #print(f"Frame {i} has masked values, replacing with nearest unmasked frame {nearest_idx}.")
            sampled_frames.append(data[nearest_idx])
            confidence_frames.append(confidence_measure[nearest_idx])
        else:
            sampled_frames.append(data[i])  # If no masks, keep the sampled frame
            confidence_frames.append(confidence_measure[i])

    return np.stack(sampled_frames), np.stack(confidence_frames)

#Rknee,Rankle ,Rheel,Rfootindex, Lknee,Lankle, Lheel,Lfootindex, Leye(in),Leye(out), \
#Reye(in),Reye(out), Mouth(2), Lpinky,Rpinky, Lindex,Rindex, Lthumb, Rthumb    

#Removed the 23 - left hip 24 - right hip due to low confidence and maybe it is not that effective.
mediapipe_exclude = [26,28, 30,32, 25,27, 29,31, 1,3, \
                        4,6, 9,10, 17,18, 19,20, 21,22, \
                            23,24]

body_sample_indices = []
for i in range(33):
    if i not in mediapipe_exclude:
        body_sample_indices.append(i)


#11 keypoints.

#Mediapipe keypoints similar to openpose
# 23 keypoints
mouth_right = 61
mouth_left = 291
lipsLowerOuter = [17]
lipsUpperOuter = [0]
rightEyebrowUpper = [70, 105, 107]
leftEyebrowUpper = [300, 334, 336]
rightEyeUpper0 = [161,158]
rightEyeLower0 = [33,163,153,133]
leftEyeUpper0 = [388, 385]
leftEyeLower0 = [263,390,380,362]
nose_top = 9


face_sample_indices = (
    [mouth_right, mouth_left] + 
    lipsLowerOuter + lipsUpperOuter + 
    rightEyebrowUpper + leftEyebrowUpper + 
    rightEyeUpper0 + rightEyeLower0 + 
    leftEyeUpper0 + leftEyeLower0 + 
    [nose_top]
)


def get_pose_keypoints(pose, component, step_frames):
    # Normalize the pose to make the centre of the shoulders the centre point.
    pose = pose.get_components([component])
    confidence_measure = pose.body.confidence
    confidence_measure = np.squeeze(confidence_measure, axis=1)
    if component == 'POSE_LANDMARKS':
        # Pose normalize to the centre of the shoulders.
        pose.normalize(pose.header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER") ))
        numpy_data = pose.body.data[:, :, body_sample_indices, :]
        confidence_measure = confidence_measure[:, body_sample_indices]
    
    elif component == 'FACE_LANDMARKS':
        # Face normalize to nose.
        pose.normalize(pose.header.normalization_info(
                p1=("FACE_LANDMARKS", "48"),
                p2=("FACE_LANDMARKS", "278") ))
        numpy_data = pose.body.data[:, :, face_sample_indices, :]
        confidence_measure = confidence_measure[:,face_sample_indices]
    
    elif component == 'LEFT_HAND_LANDMARKS':
        # Left hand normalize to wrist and thumb MCP(palm approx).
        # pose.normalize(pose.header.normalization_info(
        #         p1=("LEFT_HAND_LANDMARKS", "WRIST"),
        #         p2=("LEFT_HAND_LANDMARKS", "THUMB_MCP") ))
        pose.normalize(pose.header.normalization_info(
                p1=("LEFT_HAND_LANDMARKS", "INDEX_FINGER_MCP"),
                p2=("LEFT_HAND_LANDMARKS", "PINKY_MCP") ))
        numpy_data = pose.body.data
    elif component == 'RIGHT_HAND_LANDMARKS':
        # Right hand normalize to wrist and thumb MCP(palm approx).
        pose.normalize(pose.header.normalization_info(
                p1=("RIGHT_HAND_LANDMARKS", "INDEX_FINGER_MCP"),
                p2=("RIGHT_HAND_LANDMARKS", "PINKY_MCP") ))
        numpy_data = pose.body.data
    

    numpy_data = np.squeeze(numpy_data, axis=1)
    
    #Every step_frame
    keypoints, confidence_measure = sample_unmasked_frames(numpy_data[:,:,:2], confidence_measure,  step=step_frames)
    
    #keypoints = numpy_data[::15, :, :2]
    min_vals = np.min(keypoints, axis=(0, 1))  # Shape (2,) -> min for x and y
    max_vals = np.max(keypoints, axis=(0, 1))  # Shape (2,) -> max for x and y
    keypoints = 2 * (keypoints - min_vals) / (max_vals - min_vals) - 1
    keypoints = keypoints.reshape((keypoints.shape[0], -1))
    return keypoints , confidence_measure

def save_to_csv(keypoints, filepath):
    with open(filepath, 'a') as f:
        np.savetxt(f, keypoints, delimiter=',')
    #print(f'file {filepath.split('/')[-1]} done')


def process_pose(filename):
    data_buffer = open(filename, "rb").read()
    pose = Pose.read(data_buffer)
    #print(pose.body.data.shape)
    # for i in range(0, len(pose.header.components)):
    #     if pose.header.components[i].name == "POSE_LANDMARKS":
    #         print(pose.header.components[i].points)
    #     print(pose.header.components[i].name)
    
    # Extract the x,y of the keypoints and store them to the dataframe
    hand_left_keypoints, confidence_measure = get_pose_keypoints(pose, 'LEFT_HAND_LANDMARKS')
    if np.any(hand_left_keypoints.mask):
        raise ValueError('Some masked entries in hand_left_keypoints')
    #print(hand_left_keypoints.shape)
    hand_right_keypoints, confidence_measure = get_pose_keypoints(pose, 'RIGHT_HAND_LANDMARKS')
    if np.any(hand_right_keypoints.mask):
        raise ValueError('Some masked entries in right_left_keypoints')
    #print(hand_right_keypoints.shape)

    
    pose_keypoints, confidence_measure = get_pose_keypoints(pose, 'POSE_LANDMARKS')
    if np.any(pose_keypoints.mask):
        raise ValueError('Some masked entries in pose_keypoints')
    

    face_keypoints, confidence_measure = get_pose_keypoints(pose, 'FACE_LANDMARKS')
    if np.any(face_keypoints.mask):
        raise ValueError('Some masked entries in face_keypoints')
    

    data_path = '/DATA7/vaibhav/tokenization/kmeans_train_data/'
    save_to_csv(hand_left_keypoints, os.path.join(data_path, 'hand_left.csv'))
    save_to_csv(hand_right_keypoints, os.path.join(data_path, 'hand_right.csv'))
    save_to_csv(pose_keypoints, os.path.join(data_path, 'pose.csv'))
    save_to_csv(face_keypoints, os.path.join(data_path, 'face.csv'))
    print(f'file {filename.split("/")[-1]} done')




 
# data = np.loadtxt('/DATA7/vaibhav/tokenization/kmeans_train_data/hand_left.csv', delimiter=',')
# print(data.shape)
if __name__ == "__main__":
    # get all the files in the root_folder and call the make_pkl on each file
    pose_files = []
    pose_files_filtered = []
    with open('filter_list.pkl', 'rb') as file:
        loaded_list = pickle.load(file)

    loaded_list_name_trim = []
    for _ in loaded_list:
        loaded_list_name_trim.append(_.split("/")[-1])  

    for file in os.listdir(root_folder):
        if file.endswith(".pose") :
            pose_files.append(os.path.join(root_folder,file))

    for file in os.listdir(root_folder):
        if file.endswith(".pose") and file not in loaded_list_name_trim:
            pose_files_filtered.append(os.path.join(root_folder,file))

    print(f'Poses file original {len(pose_files)}')
    print(f'Poses file after filter(all masked even after fix removed) {len(pose_files_filtered)}')


    # Take a random sample of 100 filenames

    sampled_filenames = random.sample(pose_files_filtered, 10000)

    for file in sampled_filenames:
        process_pose(file)


    """
    POSE_LANDMARKS
    (336, 1, 33, 3)
    FACE_LANDMARKS
    (336, 1, 468, 3)
    LEFT_HAND_LANDMARKS
    (336, 1, 21, 3)
    RIGHT_HAND_LANDMARKS
    (336, 1, 21, 3)
    POSE_WORLD_LANDMARKS
    (336, 1, 33, 3)
    """

