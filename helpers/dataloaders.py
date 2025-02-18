import os
import torch
import random
import numpy as np
from datasets import Dataset
from process_data_kmeansCISLR import get_pose_keypoints
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer



def add_gaussian_noise(keypoints, mean=0, std=0.06):
    """Add Gaussian noise to keypoints"""
    noise = np.random.normal(mean, std, keypoints.shape)
    noisy_keypoints = keypoints + noise
    return noisy_keypoints




class DecoderOnlyDatasetIsign(Dataset):
    def __init__(self, video_uids, tokenizer_target, max_frames, video_dir,
                 labels=None, step_frames=None, add_noise=False):
        self.video_uids = video_uids
        self.labels = labels
        self.max_frames = max_frames
        self.video_dir = video_dir
        self.tokenizer_target = tokenizer_target
        self.step_frames = step_frames
        self.add_noise = add_noise

        
    def __len__(self):
        return len(self.video_uids)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self._get_single_item(i) for i in idx]
            return self._collate_batch(batch)
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx):
        # Load pose features
        video_uid = self.video_uids[idx]
        video_path = os.path.join(self.video_dir, f'{video_uid}.pose')
        pose_features = self.load_and_preprocess_features(video_path)
        previous_pose_features_length = min(len(pose_features), self.max_frames)
        
        # Pad or truncate pose features
        if len(pose_features) < self.max_frames:
            padding = np.zeros((self.max_frames - len(pose_features), pose_features.shape[1]))
            pose_features = np.vstack((pose_features, padding))
            pose_mask = np.zeros(self.max_frames, dtype=np.float32)  # Initialize with zeros
            pose_mask[:previous_pose_features_length] = 1  # Set 1s only for valid features
        else:
            pose_features = pose_features[:self.max_frames]
            pose_mask = np.ones(self.max_frames, dtype=np.float32)

        # Prepare text tokens
        if self.labels is not None:
            text_tokens = self.labels[idx]
            #text_tokens = [self.tokenizer_target.bos_token_id] + text_tokens + [self.tokenizer_target.eos_token_id]
        else:
            text_tokens = []

        # Combine pose features and text tokens
        input_ids = [self.tokenizer_target.convert_tokens_to_ids('<pose>')]  # Start with <pose> token
        input_ids += [self.tokenizer_target.pad_token_id] * self.max_frames  # Placeholder for pose features
        input_ids += [self.tokenizer_target.convert_tokens_to_ids('<English>')]  # Start of text
        input_ids += text_tokens  # Add text tokens

        # Create attention mask
        eos_position = text_tokens.index(self.tokenizer_target.eos_token_id) if self.tokenizer_target.eos_token_id in text_tokens else len(text_tokens)

        attention_mask = []
        attention_mask += [1]  # <pose> token
        attention_mask += [1] * previous_pose_features_length  # Active frames
        attention_mask += [0] * (self.max_frames - previous_pose_features_length)  # Padding frames
        attention_mask += [1]  # <English> token
        attention_mask += [1] * eos_position  # Text tokens until EOS
        attention_mask += [0] * (len(text_tokens) - eos_position)  # Padding after EOS

        
        # Create labels (only for text tokens, pose features are masked with -100)
        labels = [-100] * (1 + self.max_frames + 1)  # Mask <pose>, pose features, and <English>
        labels += [-100] + text_tokens[1:]  # Shift text tokens for autoregressive training

        # Convert to tensors
        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pose_features': torch.tensor(pose_features, dtype=torch.float),
            'pose_mask': torch.tensor(pose_mask, dtype=torch.float)
        }
        return item

    def _collate_batch(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        pose_features = torch.stack([item['pose_features'] for item in batch])
        pose_mask = torch.stack([item['pose_mask'] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pose_features': pose_features,
            'pose_mask': pose_mask
        }

    def load_and_preprocess_features(self, video_path):
        data_buffer = open(video_path, "rb").read()
        pose = Pose.read(data_buffer)
        if self.step_frames is None:
            self.step_frames = np.random.randint(2, 16)
        hand_left_keypoints, _ = get_pose_keypoints(pose, 'LEFT_HAND_LANDMARKS', self.step_frames)
        hand_right_keypoints, _ = get_pose_keypoints(pose, 'RIGHT_HAND_LANDMARKS', self.step_frames)
        pose_keypoints, _ = get_pose_keypoints(pose, 'POSE_LANDMARKS', self.step_frames)
        face_keypoints, _ = get_pose_keypoints(pose, 'FACE_LANDMARKS', self.step_frames)
        
        if self.add_noise:
            hand_left_keypoints = add_gaussian_noise(hand_left_keypoints)
            hand_right_keypoints = add_gaussian_noise(hand_right_keypoints)
            pose_keypoints = add_gaussian_noise(pose_keypoints)
            face_keypoints = add_gaussian_noise(face_keypoints)

        all_keypoints = np.concatenate((hand_left_keypoints, hand_right_keypoints, pose_keypoints, face_keypoints), axis=1)
        return all_keypoints 



class DecoderOnlyDatasetCISLR(Dataset):
    def __init__(self, video_uids, tokenizer_target,randomize_word_order,
                max_frames, video_dir,labels=None,   
                 step_frames=None, add_noise=False):
        self.video_uids = video_uids
        self.labels = labels
        self.max_frames = max_frames
        self.video_dir = video_dir
        self.tokenizer_target = tokenizer_target
        self.randomize_word_order = randomize_word_order
        self.step_frames = step_frames
        self.add_noise = add_noise

        
    def __len__(self):
        return len(self.video_uids)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self._get_single_item(i) for i in idx]
            return self._collate_batch(batch)
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx):
        # Load pose features
        video_uid = self.video_uids[idx]
        pose_features = self.load_and_preprocess_features(video_uid)
        previous_pose_features_length = min(len(pose_features), self.max_frames)
        
        # Pad or truncate pose features
        if len(pose_features) < self.max_frames:
            padding = np.zeros((self.max_frames - len(pose_features), pose_features.shape[1]))
            pose_features = np.vstack((pose_features, padding))
            pose_mask = np.zeros(self.max_frames, dtype=np.float32)  # Initialize with zeros
            pose_mask[:previous_pose_features_length] = 1  # Set 1s only for valid features
        else:
            pose_features = pose_features[:self.max_frames]
            pose_mask = np.ones(self.max_frames, dtype=np.float32)

        # Prepare text tokens
        if self.labels is not None:
            text_tokens = self.labels[idx]
            #text_tokens = [self.tokenizer_target.bos_token_id] + text_tokens + [self.tokenizer_target.eos_token_id]
        else:
            text_tokens = []

        # Combine pose features and text tokens
        input_ids = [self.tokenizer_target.convert_tokens_to_ids('<pose>')]  # Start with <pose> token
        input_ids += [self.tokenizer_target.pad_token_id] * self.max_frames  # Placeholder for pose features
        input_ids += [self.tokenizer_target.convert_tokens_to_ids('<English>')]  # Start of text
        input_ids += text_tokens  # Add text tokens

        # Create attention mask
        eos_position = text_tokens.index(self.tokenizer_target.eos_token_id) if self.tokenizer_target.eos_token_id in text_tokens else len(text_tokens)

        attention_mask = []
        attention_mask += [1]  # <pose> token
        attention_mask += [1] * previous_pose_features_length  # Active frames
        attention_mask += [0] * (self.max_frames - previous_pose_features_length)  # Padding frames
        attention_mask += [1]  # <English> token
        attention_mask += [1] * eos_position  # Text tokens until EOS
        attention_mask += [0] * (len(text_tokens) - eos_position)  # Padding after EOS

        
        # Create labels (only for text tokens, pose features are masked with -100)
        labels = [-100] * (1 + self.max_frames + 1)  # Mask <pose>, pose features, and <English>
        labels += [-100] + text_tokens[1:]  # Shift text tokens for autoregressive training

        # Convert to tensors
        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pose_features': torch.tensor(pose_features, dtype=torch.float),
            'pose_mask': torch.tensor(pose_mask, dtype=torch.float)
        }
        return item

    def _collate_batch(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        pose_features = torch.stack([item['pose_features'] for item in batch])
        pose_mask = torch.stack([item['pose_mask'] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pose_features': pose_features,
            'pose_mask': pose_mask
        }

    def load_and_preprocess_features(self, video_paths):
        all_features = []
        if self.randomize_word_order:
            random.shuffle(video_paths)
        for video_path in video_paths:
            # Construct the full path to the pose file
            if len(video_path) == 0:
                continue
            else:
                # Random sample a video from the list of videos
                video_path = random.choice(video_path)
            pose_file_path = os.path.join(self.video_dir, video_path + '.pose')
            # Load the pose file
            data_buffer = open(pose_file_path, "rb").read()
            pose = Pose.read(data_buffer)
            
            if self.step_frames is None:
                self.step_frames = np.random.randint(2, 16)
            # Extract keypoints
            hand_left_keypoints, confidence_measure = get_pose_keypoints(pose, 'LEFT_HAND_LANDMARKS', self.step_frames)
            hand_right_keypoints, confidence_measure = get_pose_keypoints(pose, 'RIGHT_HAND_LANDMARKS', self.step_frames)
            pose_keypoints, confidence_measure = get_pose_keypoints(pose, 'POSE_LANDMARKS', self.step_frames)
            face_keypoints, confidence_measure = get_pose_keypoints(pose, 'FACE_LANDMARKS', self.step_frames)

            if self.add_noise:
                hand_left_keypoints = add_gaussian_noise(hand_left_keypoints)
                hand_right_keypoints = add_gaussian_noise(hand_right_keypoints)
                pose_keypoints = add_gaussian_noise(pose_keypoints)
                face_keypoints = add_gaussian_noise(face_keypoints)


            # Concatenate keypoints for the current pose file
            keypoints = np.concatenate((hand_left_keypoints, hand_right_keypoints, pose_keypoints, face_keypoints), axis=1)
            # Append the keypoints to the list of all features
            all_features.append(keypoints)
        
        # Stack all features along the frames (axis 0)
        final_features = np.vstack(all_features)
        return final_features



class FeatureVectorDataset_Isign(Dataset):
    def __init__(self, video_uids, tokenizer_target, max_frames, video_dir,
                labels=None, step_frames=None, add_noise = False):
        self.video_uids = video_uids
        self.labels = labels
        self.max_frames = max_frames
        self.video_dir = video_dir
        self.tokenizer_target = tokenizer_target
        self.step_frames = step_frames
        self.add_noise = add_noise

    def __len__(self):
        return len(self.video_uids)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self._get_single_item(i) for i in idx]
            return self._collate_batch(batch)
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx):
        video_uid = self.video_uids[idx]
        video_path = os.path.join(self.video_dir, f'{video_uid}.pose')
        feature_vector = self.load_and_preprocess_features(video_path)
        
        attention_mask = np.ones(len(feature_vector), dtype=np.float32)
        if len(feature_vector) < self.max_frames:
            padding = np.zeros((self.max_frames - len(feature_vector), feature_vector.shape[1]))
            feature_vector = np.vstack((feature_vector, padding))
            attention_mask = np.concatenate((attention_mask, np.zeros(self.max_frames - len(attention_mask), dtype=np.float32)))
        else:
            feature_vector = feature_vector[:self.max_frames]
            attention_mask = attention_mask[:self.max_frames]

        item = {
            'input_ids': torch.tensor(feature_vector, dtype=torch.float),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float)
        }
        if self.labels is not None:
            labels = self.labels[idx]
            labels = [(label if label != self.tokenizer_target.pad_token_id else -100) for label in labels]
            item['labels'] = torch.tensor(labels, dtype=torch.long)
        return item

    def _collate_batch(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch]) if 'labels' in batch[0] else None
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels} if labels is not None else {'input_ids': input_ids, 'attention_mask': attention_mask}

    def load_and_preprocess_features(self, video_path):
        data_buffer = open(video_path, "rb").read()
        pose = Pose.read(data_buffer)
        if self.step_frames is None:
            self.step_frames = np.random.randint(1, 4) # Changed from 2, 16
        hand_left_keypoints, _ = get_pose_keypoints(pose, 'LEFT_HAND_LANDMARKS', self.step_frames)
        hand_right_keypoints, _ = get_pose_keypoints(pose, 'RIGHT_HAND_LANDMARKS', self.step_frames)
        pose_keypoints, _ = get_pose_keypoints(pose, 'POSE_LANDMARKS', self.step_frames)
        face_keypoints, _ = get_pose_keypoints(pose, 'FACE_LANDMARKS', self.step_frames)
        
        if self.add_noise:
            hand_left_keypoints = add_gaussian_noise(hand_left_keypoints)
            hand_right_keypoints = add_gaussian_noise(hand_right_keypoints)
            pose_keypoints = add_gaussian_noise(pose_keypoints)
            face_keypoints = add_gaussian_noise(face_keypoints)

        all_keypoints = np.concatenate((hand_left_keypoints, hand_right_keypoints, pose_keypoints, face_keypoints), axis=1)
        return all_keypoints



#Choose random from a list of uids.
class FeatureVectorDataset(Dataset):
    def __init__(self, video_uids, tokenizer_target,randomize_word_order, 
                 max_frames, video_dir, labels=None,  
                 step_frames=None, add_noise = False):
        self.video_uids = video_uids
        self.labels = labels
        self.max_frames = max_frames
        self.video_dir = video_dir
        self.tokenizer_target = tokenizer_target
        self.randomize_word_order = randomize_word_order
        self.step_frames = step_frames
        self.add_noise = add_noise


    def __len__(self):
        return len(self.video_uids)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self._get_single_item(i) for i in idx]
            return self._collate_batch(batch)
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx):
        video_uid = self.video_uids[idx]
        try:
            feature_vector = self.load_and_preprocess_features(video_uid)
        except ValueError as e:
            print(e)
            print(f'Error in video_uid: {self.labels[idx]}')
        # Pad or truncate the feature vector to the maximum number of frames
        attention_mask = np.ones(len(feature_vector), dtype=np.float32)
        if len(feature_vector) < self.max_frames:
            padding = np.zeros((self.max_frames - len(feature_vector), feature_vector.shape[1]))
            feature_vector = np.vstack((feature_vector, padding))
            attention_mask = np.concatenate((attention_mask, np.zeros(self.max_frames - len(attention_mask), dtype=np.float32)))
        else:
            feature_vector = feature_vector[:self.max_frames]
            attention_mask = attention_mask[:self.max_frames]

        item = {
            'input_ids': torch.tensor(feature_vector, dtype=torch.float),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float)
        }
        if self.labels is not None:
            labels = self.labels[idx]
            labels = [(label if label != self.tokenizer_target.pad_token_id else -100) for label in labels]
            item['labels'] = torch.tensor(labels, dtype=torch.long)
        return item

    def _collate_batch(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch]) if 'labels' in batch[0] else None
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels} if labels is not None else {'input_ids': input_ids, 'attention_mask': attention_mask}


    def load_and_preprocess_features(self, video_paths):
        all_features = []
        if self.randomize_word_order:
            random.shuffle(video_paths)
        for video_path in video_paths:
            # Construct the full path to the pose file
            if len(video_path) == 0:
                continue
            else:
                # Random sample a video from the list of videos
                video_path = random.choice(video_path)
            pose_file_path = os.path.join(self.video_dir, video_path + '.pose')
            # Load the pose file
            data_buffer = open(pose_file_path, "rb").read()
            pose = Pose.read(data_buffer)
            
            if self.step_frames is None:
                self.step_frames = np.random.randint(1, 4) # Changed from 2, 16
            else:
                self.step_frames += np.random.randint(1, 4)
            # Extract keypoints
            hand_left_keypoints, confidence_measure = get_pose_keypoints(pose, 'LEFT_HAND_LANDMARKS', self.step_frames)
            hand_right_keypoints, confidence_measure = get_pose_keypoints(pose, 'RIGHT_HAND_LANDMARKS', self.step_frames)
            pose_keypoints, confidence_measure = get_pose_keypoints(pose, 'POSE_LANDMARKS', self.step_frames)
            face_keypoints, confidence_measure = get_pose_keypoints(pose, 'FACE_LANDMARKS', self.step_frames)

            if self.add_noise:
                hand_left_keypoints = add_gaussian_noise(hand_left_keypoints, std=0.01)
                hand_right_keypoints = add_gaussian_noise(hand_right_keypoints, std=0.01)
                pose_keypoints = add_gaussian_noise(pose_keypoints, std=0.1)
                face_keypoints = add_gaussian_noise(face_keypoints, std=0.01)


            # Concatenate keypoints for the current pose file
            keypoints = np.concatenate((hand_left_keypoints, hand_right_keypoints, pose_keypoints, face_keypoints), axis=1)
            
            # # randomly remove starting and ending 15% of the keypoints
            # start = int(0.1 * len(keypoints))
            # end = int(0.9 * len(keypoints))
            # keypoints = keypoints[start:end]
            
            # Append the keypoints to the list of all features
            all_features.append(keypoints)
        
        # Stack all features along the frames (axis 0)
        if len(all_features) == 0:
            raise ValueError(f'No features found for {video_paths}')
        final_features = np.vstack(all_features)
        return final_features





#Choose random from a list of uids.
class FeatureVectorDatasetBPE(Dataset):
    def __init__(self, video_uids, tokenizer_target,randomize_word_order, 
                 max_frames, video_dir, labels=None,  
                 step_frames=None, add_noise = False):
        self.video_uids = video_uids
        self.labels = labels
        self.max_frames = max_frames
        self.video_dir = video_dir
        self.tokenizer_target = tokenizer_target
        self.randomize_word_order = randomize_word_order
        self.step_frames = step_frames
        self.add_noise = add_noise


    def __len__(self):
        return len(self.video_uids)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self._get_single_item(i) for i in idx]
            return self._collate_batch(batch)
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx):
        video_uid = self.video_uids[idx]
        try:
            feature_vector = self.load_and_preprocess_features(video_uid)
        except ValueError as e:
            print(e)
            print(f'Error in video_uid: {self.labels[idx]}')
        # Pad or truncate the feature vector to the maximum number of frames
        attention_mask = np.ones(len(feature_vector), dtype=np.float32)
        if len(feature_vector) < self.max_frames:
            padding = np.zeros((self.max_frames - len(feature_vector), feature_vector.shape[1]))
            feature_vector = np.vstack((feature_vector, padding))
            attention_mask = np.concatenate((attention_mask, np.zeros(self.max_frames - len(attention_mask), dtype=np.float32)))
        else:
            feature_vector = feature_vector[:self.max_frames]
            attention_mask = attention_mask[:self.max_frames]

        item = {
            'input_ids': torch.tensor(feature_vector, dtype=torch.float),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float)
        }
        if self.labels is not None:
            labels = self.labels[idx]
            labels = [(label if label != self.tokenizer_target.get_vocab().get("<pad>") else -100) for label in labels]
            item['labels'] = torch.tensor(labels, dtype=torch.long)
        return item

    def _collate_batch(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch]) if 'labels' in batch[0] else None
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels} if labels is not None else {'input_ids': input_ids, 'attention_mask': attention_mask}


    def load_and_preprocess_features(self, video_paths):
        all_features = []
        if self.randomize_word_order:
            random.shuffle(video_paths)
        for video_path in video_paths:
            # Construct the full path to the pose file
            if len(video_path) == 0:
                continue
            else:
                # Random sample a video from the list of videos
                video_path = random.choice(video_path)
            pose_file_path = os.path.join(self.video_dir, video_path + '.pose')
            # Load the pose file
            data_buffer = open(pose_file_path, "rb").read()
            pose = Pose.read(data_buffer)
            
            if self.step_frames is None:
                self.step_frames = np.random.randint(1, 4) # Changed from 2, 16
            else:
                self.step_frames += np.random.randint(1, 4)
            # Extract keypoints
            hand_left_keypoints, confidence_measure = get_pose_keypoints(pose, 'LEFT_HAND_LANDMARKS', self.step_frames)
            hand_right_keypoints, confidence_measure = get_pose_keypoints(pose, 'RIGHT_HAND_LANDMARKS', self.step_frames)
            pose_keypoints, confidence_measure = get_pose_keypoints(pose, 'POSE_LANDMARKS', self.step_frames)
            face_keypoints, confidence_measure = get_pose_keypoints(pose, 'FACE_LANDMARKS', self.step_frames)

            if self.add_noise:
                hand_left_keypoints = add_gaussian_noise(hand_left_keypoints, std=0.01)
                hand_right_keypoints = add_gaussian_noise(hand_right_keypoints, std=0.01)
                pose_keypoints = add_gaussian_noise(pose_keypoints, std=0.1)
                face_keypoints = add_gaussian_noise(face_keypoints, std=0.01)


            # Concatenate keypoints for the current pose file
            keypoints = np.concatenate((hand_left_keypoints, hand_right_keypoints, pose_keypoints, face_keypoints), axis=1)
            
            # # randomly remove starting and ending 15% of the keypoints
            # start = int(0.1 * len(keypoints))
            # end = int(0.9 * len(keypoints))
            # keypoints = keypoints[start:end]
            
            # Append the keypoints to the list of all features
            all_features.append(keypoints)
        
        # Stack all features along the frames (axis 0)
        if len(all_features) == 0:
            raise ValueError(f'No features found for {video_paths}')
        final_features = np.vstack(all_features)
        return final_features



class FeatureVectorDataset_IsignBPE(Dataset):
    def __init__(self, video_uids, tokenizer_target, max_frames, video_dir,
                labels=None, step_frames=None, add_noise = False):
        self.video_uids = video_uids
        self.labels = labels
        self.max_frames = max_frames
        self.video_dir = video_dir
        self.tokenizer_target = tokenizer_target
        self.step_frames = step_frames
        self.add_noise = add_noise

    def __len__(self):
        return len(self.video_uids)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self._get_single_item(i) for i in idx]
            return self._collate_batch(batch)
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx):
        video_uid = self.video_uids[idx]
        video_path = os.path.join(self.video_dir, f'{video_uid}.pose')
        feature_vector = self.load_and_preprocess_features(video_path)
        
        attention_mask = np.ones(len(feature_vector), dtype=np.float32)
        if len(feature_vector) < self.max_frames:
            padding = np.zeros((self.max_frames - len(feature_vector), feature_vector.shape[1]))
            feature_vector = np.vstack((feature_vector, padding))
            attention_mask = np.concatenate((attention_mask, np.zeros(self.max_frames - len(attention_mask), dtype=np.float32)))
        else:
            feature_vector = feature_vector[:self.max_frames]
            attention_mask = attention_mask[:self.max_frames]

        item = {
            'input_ids': torch.tensor(feature_vector, dtype=torch.float),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float)
        }
        if self.labels is not None:
            labels = self.labels[idx]
            labels = [(label if label != self.tokenizer_target.get_vocab().get("<pad>") else -100) for label in labels]
            item['labels'] = torch.tensor(labels, dtype=torch.long)
        return item

    def _collate_batch(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch]) if 'labels' in batch[0] else None
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels} if labels is not None else {'input_ids': input_ids, 'attention_mask': attention_mask}

    def load_and_preprocess_features(self, video_path):
        data_buffer = open(video_path, "rb").read()
        pose = Pose.read(data_buffer)
        if self.step_frames is None:
            self.step_frames = np.random.randint(1, 4) # Changed from 2, 16
        hand_left_keypoints, _ = get_pose_keypoints(pose, 'LEFT_HAND_LANDMARKS', self.step_frames)
        hand_right_keypoints, _ = get_pose_keypoints(pose, 'RIGHT_HAND_LANDMARKS', self.step_frames)
        pose_keypoints, _ = get_pose_keypoints(pose, 'POSE_LANDMARKS', self.step_frames)
        face_keypoints, _ = get_pose_keypoints(pose, 'FACE_LANDMARKS', self.step_frames)
        
        if self.add_noise:
            hand_left_keypoints = add_gaussian_noise(hand_left_keypoints)
            hand_right_keypoints = add_gaussian_noise(hand_right_keypoints)
            pose_keypoints = add_gaussian_noise(pose_keypoints)
            face_keypoints = add_gaussian_noise(face_keypoints)

        all_keypoints = np.concatenate((hand_left_keypoints, hand_right_keypoints, pose_keypoints, face_keypoints), axis=1)
        return all_keypoints

