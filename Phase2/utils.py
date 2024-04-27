import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from loss import PoseLoss, TransformationLoss
from Network import VisionDataNetwork, InertialDataNetwork, CombinedVIONetwork
from tqdm import tqdm
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R
from torch.utils.data.dataloader import default_collate




class ProcessData(Dataset):
    def __init__(self, imu_csv_file, ground_truth_csv_file, image_folder, split_ratios=(0.8, 0.1, 0.1), mode='train', collate_fn=default_collate):

        
        # Load IMU data and prepend a row of zeros
        imu_data = pd.read_csv(imu_csv_file)
        zero_row = pd.DataFrame([np.zeros(imu_data.shape[1])], columns=imu_data.columns)
        self.imu_data = pd.concat([zero_row, imu_data], ignore_index=True)
        self.imu_data_timestamps = np.arange(0, len(self.imu_data))

        # Load ground truth data and prepend a row of zeros
        ground_truth_data = pd.read_csv(ground_truth_csv_file)
        zero_row_gt = pd.DataFrame([np.zeros(ground_truth_data.shape[1])], columns=ground_truth_data.columns)
        self.ground_truth = pd.concat([zero_row_gt, ground_truth_data], ignore_index=True)
        self.ground_truth_timestamps = np.arange(0, len(self.ground_truth))

        # Load and sort image files
        self.image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])
        # print(f"Found {len(self.image_files)} image files.")
        self.images_timestamps = np.arange(0, ((len(self.image_files) - 1) * 10) + 1, 10)
        self.images_timestamps[0] = 0

        # Calculate indices for data splits
        total_images = len(self.image_files)
        train_end = int(total_images * split_ratios[0])
        val_end = train_end + int(total_images * split_ratios[1])

        self.indices = range(0, total_images)
        
        if mode == 'train':
            self.indices = range(0, train_end)
        elif mode == 'val':
            self.indices = range(train_end, val_end)
        elif mode == 'test':
            self.indices = range(val_end, total_images)
        else:
            raise ValueError("Invalid mode specified.")

    def __len__(self):
        return len(self.indices) - 1  # Since we need pairs of images

    def __getitem__(self, idx):
        # print(f'Index: {idx}')
        index = self.indices[idx]
        # print(f'Index1: {index}')
        # print(f'Index2: {self.images_timestamps[index]}')
        img1_path = self.image_files[index]
        img2_path = self.image_files[index + 1]

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return None, None, None, None

        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)

        img1 = torch.tensor(img1, dtype=torch.float32)
        img2 = torch.tensor(img2, dtype=torch.float32)

        img1_timestamp = self.images_timestamps[index]
        img2_timestamp = self.images_timestamps[index + 1]

        # Fetching IMU data based on index
        # print(f'imu_time:{(self.imu_data_timestamps >= img1_timestamp) & (self.imu_data_timestamps < img2_timestamp)}')
        imu_sequence = self.imu_data[(self.imu_data_timestamps >= img1_timestamp) & (self.imu_data_timestamps <= img2_timestamp)]

        imu_sequence = imu_sequence.iloc[:,0:].values.astype(np.float32)

        imu_data = torch.tensor(imu_sequence, dtype=torch.float32)

        # Extract and compute relative ground truth poses
        # print(self.imu_data_timestamps)
        gt_pose1 = self.ground_truth[self.ground_truth_timestamps == img1_timestamp]
        gt_pose2 = self.ground_truth[self.ground_truth_timestamps == img2_timestamp]

        if gt_pose1.empty or gt_pose2.empty:
            return None

        gt_pose1 = gt_pose1.iloc[0, 0:7].values  # Extract position and orientation only
        gt_pose2 = gt_pose2.iloc[0, 0:7].values  # Extract position and orientation only

        if np.all(gt_pose1[3:] == 0) or np.all(gt_pose2[3:] == 0):
            # Handling zero quaternion by using a default quaternion (no rotation)
            gt_pose1[3:] = [1, 0, 0, 0]
            gt_pose2[3:] = [1, 0, 0, 0]

        # Compute relative position and orientation
        gt_rel_position = gt_pose2[:3] - gt_pose1[:3]
        quat1_inv = R.from_quat(gt_pose1[3:]).inv()
        quat_rel = quat1_inv * R.from_quat(gt_pose2[3:])
        gt_rel_orientation = quat_rel.as_quat()
        gt_rel_pose = np.concatenate([gt_rel_position, gt_rel_orientation])

        gt_rel_pose = torch.tensor(gt_rel_pose, dtype=torch.float32)
        # print(img1.shape, img2.shape, imu_data.shape, gt_rel_pose.shape)

        return img1, img2, imu_data, gt_rel_pose