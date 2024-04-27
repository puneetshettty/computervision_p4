import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from loss import PoseLoss, TransformationLoss
from Network import VisionDataNetwork, InertialDataNetwork, CombinedVIONetwork
from tqdm import tqdm
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as R
from torch.utils.data.dataloader import default_collate
from utils import ProcessData

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# Loss functions
inertial_loss_fn = TransformationLoss()
visual_inertial_loss_fn = TransformationLoss()
pose_loss_fn = PoseLoss(alpha=0.5)

# Set your hyperparameters
learning_rate = 1e-4

# Create instances of the network
vision_network = VisionDataNetwork()
inertial_network = InertialDataNetwork()
combined_network = CombinedVIONetwork()

# Move the networks to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_network.to(device)
inertial_network.to(device)
combined_network.to(device)

# Create optimizers for each network
vision_optimizer = torch.optim.Adam(vision_network.parameters(), lr=learning_rate)
inertial_optimizer = torch.optim.Adam(inertial_network.parameters(), lr=learning_rate)
combined_optimizer = torch.optim.Adam(combined_network.parameters(), lr=learning_rate)



def train_vision_network(model, optimizer, img1, img2, groundtruth_pose, device, train=True, checkpoint = None):
    """ Train or evaluate the vision-only network based on images and ground truth relative pose. """
    if train:
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))
        model.train()  # Set model to training mode
    else:
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))
        model.eval()   # Set model to evaluation mode
    
    optimizer.zero_grad()  # Clear previous gradients

    # Move data to the appropriate device (GPU or CPU)
    img1 = img1.to(device)
    img2 = img2.to(device)
    groundtruth_pose = groundtruth_pose.to(device)
    
    # Forward pass to get the predicted relative pose
    predicted_pose = model(img1, img2)
    
    # Compute loss using the pose loss function with an alpha parameter
    loss = pose_loss_fn(predicted_pose, groundtruth_pose)

    if train:
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters

    return loss.item()

def train_inertial_network(model, optimizer, imu_data, groundtruth_pose, device, train=True, checkpoint = None):
    """ Train or evaluate the inertial network based on IMU data and ground truth relative pose. """
    if train:
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))
        model.train()
    else:
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))
        model.eval()
    
    optimizer.zero_grad()

    # Move data to device
    imu_data = imu_data.to(device)
    groundtruth_pose = groundtruth_pose.to(device)

    # Predict relative pose based on IMU data
    predicted_pose = model(imu_data)
    # print(predicted_pose.shape)

    # Calculate loss
    loss = pose_loss_fn(predicted_pose, groundtruth_pose)

    if train:
        loss.backward()
        optimizer.step()

    return loss.item()

def train_combined_network(model, optimizer, img1, img2, imu_data, groundtruth_pose, device, train=True, checkpoint = None):
    """ Train or evaluate the combined visual-inertial network. """
    if train:
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))
        model.train()
    else:
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))
        model.eval()

    optimizer.zero_grad()

    # Move inputs and ground truth to device
    img1 = img1.to(device)
    img2 = img2.to(device)
    imu_data = imu_data.to(device)
    groundtruth_pose = groundtruth_pose.to(device)

    # Compute predictions
    predicted_pose = model(img1, img2, imu_data)


    # Calculate loss
    loss = pose_loss_fn(predicted_pose, groundtruth_pose)

    if train:
        loss.backward()
        optimizer.step()

    return loss.item()


def train_networks(num_epochs, train_dataloader, val_dataloader, device, checkpoints = None):
    # Lists to store training and validation losses for each network
    losses = {
        "vision": {"train": [], "val": []},
        "inertial": {"train": [], "val": []},
        "visual_inertial": {"train": [], "val": []}
    }

    # Main training and validation loop
    for epoch in tqdm(range(num_epochs), desc="Epoch Progress"):
        # print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train and validate each network
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                torch.set_grad_enabled(True)
            else:
                dataloader = val_dataloader
                torch.set_grad_enabled(False)

            # Initialize epoch losses
            epoch_losses = {
                "vision": 0,
                "inertial": 0,
                "visual_inertial": 0
            }
            num_batches = 0

            for data in dataloader:
                if data is None:
                    continue
                img1, img2, imu_data, groundtruth_pose = data
                # imu_data, groundtruth_pose = data

                if img1 is None or img2 is None:
                    print("Skipping batch due to missing image(s)")
                    continue

                if phase == 'train':
                    if checkpoints is None:
                        epoch_losses["vision"] += train_vision_network(vision_network, vision_optimizer, img1, img2, groundtruth_pose, device)
                        epoch_losses["inertial"] += train_inertial_network(inertial_network, inertial_optimizer, imu_data, groundtruth_pose, device)
                        epoch_losses["visual_inertial"] += train_combined_network(combined_network, combined_optimizer, img1, img2, imu_data, groundtruth_pose, device)
                    else:
                        epoch_losses["vision"] += train_vision_network(vision_network, vision_optimizer, img1, img2, groundtruth_pose, device, checkpoint=checkpoints[1])
                        epoch_losses["inertial"] += train_inertial_network(inertial_network, inertial_optimizer, imu_data, groundtruth_pose, device, checkpoint=checkpoints[0])
                        epoch_losses["visual_inertial"] += train_combined_network(combined_network, combined_optimizer, img1, img2, imu_data, groundtruth_pose, device, checkpoint=checkpoints[2])
                else:
                    if checkpoints is None:
                        epoch_losses["vision"] += train_vision_network(vision_network, vision_optimizer, img1, img2, groundtruth_pose, device, train=False)
                        epoch_losses["inertial"] += train_inertial_network(inertial_network, inertial_optimizer, imu_data, groundtruth_pose, device, train=False)
                        epoch_losses["visual_inertial"] += train_combined_network(combined_network, combined_optimizer, img1, img2, imu_data, groundtruth_pose, device, train=False)
                    else:
                        epoch_losses["vision"] += train_vision_network(vision_network, vision_optimizer, img1, img2, groundtruth_pose, device, train=False, checkpoint=checkpoints[1])
                        epoch_losses["inertial"] += train_inertial_network(inertial_network, inertial_optimizer, imu_data, groundtruth_pose, device, train=False, checkpoint=checkpoints[0])
                        epoch_losses["visual_inertial"] += train_combined_network(combined_network, combined_optimizer, img1, img2, imu_data, groundtruth_pose, device, train=False, checkpoint=checkpoints[2])

                num_batches += 1

            # Store average loss for the epoch
            for key in epoch_losses:
                losses[key][phase].append(epoch_losses[key] / max(num_batches, 1))

        # Optional: Save model checkpoints
        torch.save(vision_network.state_dict(), f"checkpoints/vision_model_epoch_{epoch + 1}.pth")
        torch.save(inertial_network.state_dict(), f"checkpoints/inertial_model_epoch_{epoch + 1}.pth")
        torch.save(combined_network.state_dict(), f"checkpoints/visual_inertial_model_epoch_{epoch + 1}.pth")

    return losses

def plot_losses(train_losses, val_losses, network_name):
    plt.figure(figsize=(10, 10))
    plt.plot(range(1, num_epochs + 1), train_losses, label=f'Training {network_name}')
    plt.plot(range(1, num_epochs + 1), val_losses, label=f'Validation {network_name}')
    print(f"Final Training {network_name} Loss: {train_losses[-1]}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{network_name} Training and Validation Loss vs Epoch')
    plt.legend()
    plt.grid()
    plt.savefig(f'plots/{network_name}_losses.png')

# Hyperparameters and DataLoader setup
num_epochs = 50
batch_size = 32

for directory in os.listdir('Data'):
    checkpoints = []
    if directory != 'triangle-movement-with-ground-truth':
        continue
    #check for highest checkpoint in 'checkpoints' folder
    # for f in os.listdir('checkpoints'):
    #     if f.endswith('50.pth'):
    #         checkpoints.append(f'checkpoints/{f}')

    
    # print(checkpoints)
    
    # print(f'Data/{directory}/imu_with_noise.csv',
    #                         f'Data/{directory}/ground_truth.csv',
    #                         f'Data/{directory}/Frames')
    
    train_dataset = ProcessData(f'Data/{directory}/imu_with_noise.csv',
                            f'Data/{directory}/ground_truth.csv',
                            f'Data/{directory}/Frames',
                            mode='train')

    val_dataset = ProcessData(f'Data/{directory}/imu_with_noise.csv',
                            f'Data/{directory}/ground_truth.csv',
                            f'Data/{directory}/Frames',
                            mode='val')

    print(f"Training dataset: {directory}")


    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(checkpoints) > 10:
        # Execute training
        print('rre')
        network_losses = train_networks(num_epochs, train_dataloader, val_dataloader, device, checkpoints = checkpoints)

    else:
        # Execute training
        network_losses = train_networks(num_epochs, train_dataloader, val_dataloader, device)

    break

    # plot_losses(network_losses['vision']['train'], network_losses['vision']['val'], 'Visual Network')
    # plot_losses(network_losses['inertial']['train'], network_losses['inertial']['val'], 'Inertial Network')
    # plot_losses(network_losses['visual_inertial']['train'], network_losses['visual_inertial']['val'], 'Visual-Inertial Network')
