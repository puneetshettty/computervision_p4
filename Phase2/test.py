import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from loss import PoseLoss, TransformationLoss
from utils import ProcessData
from Network import VisionDataNetwork, InertialDataNetwork, CombinedVIONetwork
from tqdm import tqdm
from PIL import Image
import cv2
from torch.utils.data.dataloader import default_collate
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion

pose_loss = PoseLoss(alpha=0.5)

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# from train import vision_loss_fn, inertial_loss_fn, visual_inertial_loss_fn

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_gt_positions = []
    all_pred_positions = []
    all_gt_rotations = []
    all_pred_rotations = []

    all_pred_rel_pose = []
    all_gt_rel_pose = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((len(dataloader), 1))

    with torch.no_grad():
        for it, data in enumerate(dataloader):
            if data is None:
                continue
            
            img1, img2, imu_data, gt_rel_pose = [d.to(device) for d in data]

            x1 = torch.randn(img1.shape).cuda()
            x2 = torch.randn(img2.shape).cuda()
            x3 = torch.randn(imu_data.shape).cuda()
            x4 = torch.randn(gt_rel_pose.shape).cuda()

            if isinstance(model, VisionDataNetwork):
                for _ in range(10):
                    _ = model(x1, x2)
                starter.record()
                pred_rel_pose = model(img1, img2)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[it] = curr_time
            elif isinstance(model, InertialDataNetwork):
                for _ in range(10):
                    _ = model(x3)
                starter.record()
                pred_rel_pose = model(imu_data)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[it] = curr_time
            elif isinstance(model, CombinedVIONetwork):
                for _ in range(10):
                    _ = model(x1, x2, x3)
                starter.record()
                pred_rel_pose = model(img1, img2, imu_data)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[it] = curr_time
            else:
                raise ValueError("Unknown model type")

            loss = pose_loss(pred_rel_pose, gt_rel_pose)
            total_loss += loss.item() #* gt_rel_pose.size(0)
            total_samples += gt_rel_pose.size(0)

            all_pred_rel_pose.append(pred_rel_pose)
            all_gt_rel_pose.append(gt_rel_pose)

            # gt_positions, pred_positions, gt_rotations, pred_rotations = integrate_poses(gt_rel_pose, pred_rel_pose, device)
            # all_gt_positions.extend(gt_positions)
            # all_pred_positions.extend(pred_positions)
            # all_gt_rotations.extend(gt_rotations)
            # all_pred_rotations.extend(pred_rotations)

    mean_syn = np.sum(timings) / len(dataloader)
    print("Inference time for {} is {}".format(model.__class__.__name__, mean_syn))
    mean_loss = total_loss / total_samples
    return mean_loss, all_pred_rel_pose, all_gt_rel_pose

# def integrate_poses(gt_rel_pose, pred_rel_pose, device):
#     gt_positions = [torch.zeros((3,), device=device)]  # Initial position at (0, 0, 0)
#     pred_positions = [torch.zeros((3,), device=device)]
#     gt_rotations = [torch.zeros((4,), device=device)]  # Initial rotation angles (0, 0, 0, 0)
#     pred_rotations = [torch.zeros((4,), device=device)]

#     for i in range(len(gt_rel_pose)):
#         gt_positions.append(gt_positions[-1] + gt_rel_pose[i, :3].to(device))
#         pred_positions.append(pred_positions[-1] + pred_rel_pose[i, :3].to(device))
#         gt_rotations.append(gt_rotations[-1] + gt_rel_pose[i, 3:].to(device))
#         pred_rotations.append(pred_rotations[-1] + pred_rel_pose[i, 3:].to(device))

#     return gt_positions, pred_positions, gt_rotations, pred_rotations


def plot_trajectory(gt_positions, pred_positions, title):

    fig, axs = plt.subplots(3, 1, figsize=(10, 20))
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        axs[i].plot(gt_positions[:, i], label="Ground Truth")
        axs[i].plot(pred_positions[:, i], label="Prediction")
        axs[i].set_xlabel("Time Step")
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        axs[i].grid()

    fig.suptitle(title)
    plt.savefig(f'plots/{title}.png')

def plot_rotations(gt_rotations, pred_rotations, title):
    gt_rotations_np = gt_rotations
    pred_rotations_np = pred_rotations

    gt_euler_angles = np.array([Rotation.from_quat(q).as_euler('xyz') for q in gt_rotations_np if np.linalg.norm(q) != 0])
    pred_euler_angles = np.array([Rotation.from_quat(q).as_euler('xyz') for q in pred_rotations_np if np.linalg.norm(q) != 0])
    print(f'For {title} - GT Euler Angles: {gt_euler_angles.shape}, Pred Euler Angles: {pred_euler_angles.shape}')

    fig, axs = plt.subplots(3, 1, figsize=(10, 20))
    labels = ['Roll', 'Pitch', 'Yaw']
    for i in range(3):
        axs[i].plot(gt_euler_angles[:, i], label="Ground Truth")
        axs[i].plot(pred_euler_angles[:, i], label="Prediction")
        axs[i].set_xlabel("Time Step")
        axs[i].set_ylabel(labels[i] + " (Radians)")
        axs[i].legend()
        axs[i].grid()

    fig.suptitle(title)
    plt.savefig(f'plots/{title}.png')


# def scatter_trajectory_3d(gt_positions, pred_positions, gt_rotations, pred_rotations, title):
#     gt_positions = torch.stack(gt_positions).cpu().numpy()
#     pred_positions = torch.stack(pred_positions).cpu().numpy()
#     gt_rotations = torch.stack(gt_rotations).cpu().numpy()
#     pred_rotations = torch.stack(pred_rotations).cpu().numpy()

#     valid_gt_indices = np.linalg.norm(gt_rotations, axis=1) != 0
#     valid_pred_indices = np.linalg.norm(pred_rotations, axis=1) != 0

#     gt_positions = gt_positions[valid_gt_indices]
#     gt_rotations = gt_rotations[valid_gt_indices]

#     pred_positions = pred_positions[valid_pred_indices]
#     pred_rotations = pred_rotations[valid_pred_indices]

#     gt_euler_angles = np.array([Rotation.from_quat(q).as_euler('xyz') for q in gt_rotations])
#     pred_euler_angles = np.array([Rotation.from_quat(q).as_euler('xyz') for q in pred_rotations])

#     # Combine translation and rotation values for full pose visualization
#     gt_full_poses = np.hstack([gt_positions, gt_euler_angles])
#     pred_full_poses = np.hstack([pred_positions, pred_euler_angles])

#     fig = plt.figure(figsize=(10, 10))
#     ax1 = fig.add_subplot(121, projection='3d')

#     # Plotting lines instead of scatter points
#     ax1.plot(gt_full_poses[:, 0], gt_full_poses[:, 1], gt_full_poses[:, 2], label="Ground Truth", marker='o')
#     ax1.set_xlabel("X")
#     ax1.set_ylabel("Y")
#     ax1.set_zlabel("Z")
#     ax1.legend()
#     ax1.set_title(title)
    
#     ax2 = fig.add_subplot(122, projection='3d')

#     ax2.plot(pred_full_poses[:, 0], pred_full_poses[:, 1], pred_full_poses[:, 2], label="Prediction", marker='^')
#     ax2.set_xlabel("X")
#     ax2.set_ylabel("Y")
#     ax2.set_zlabel("Z")
#     ax2.legend()
#     ax2.set_title(title)
#     plt.savefig(f'plots/{title}.png')

#     #create overlay plot
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(gt_full_poses[:, 0], gt_full_poses[:, 1], gt_full_poses[:, 2], label="Ground Truth", marker='o')
#     ax.plot(pred_full_poses[:, 0], pred_full_poses[:, 1], pred_full_poses[:, 2], label="Prediction", marker='^')
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.legend()
#     ax.set_title(title)
#     plt.savefig(f'plots/{title}_overlay.png')




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained models
vision_model = VisionDataNetwork().to(device)
inertial_model = InertialDataNetwork().to(device)
visual_inertial_model = CombinedVIONetwork().to(device)

vision_model.load_state_dict(torch.load("checkpoints/vision_model_epoch_50.pth"))
inertial_model.load_state_dict(torch.load("checkpoints/inertial_model_epoch_50.pth"))
visual_inertial_model.load_state_dict(torch.load("checkpoints/visual_inertial_model_epoch_50.pth"))

# Loss functions
# vision_loss_fn = TranslationRotationLoss()
# inertial_loss_fn = TranslationRotationLoss()
# visual_inertial_loss_fn = TranslationRotationLoss()

test_dataset = ProcessData('Data/triangle-movement-with-ground-truth/imu_raw.csv', '/home/eclement/workspace/cv/eclement_p4/Phase2/Data/triangle-movement-with-ground-truth/ground_truth.csv', 'Data/triangle-movement-with-ground-truth/Frames', mode='test', collate_fn=custom_collate)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
# Evaluate models
vision_loss, vision_pred_rel_pose, vision_gt_rel_pose= evaluate_model(vision_model, test_dataloader, device)
inertial_loss, inertial_pred_rel_pose, inertial_gt_rel_pose = evaluate_model(inertial_model, test_dataloader, device)
visual_inertial_loss, visual_inertial_pred_rel_pose, visual_inertial_gt_rel_pose = evaluate_model(visual_inertial_model, test_dataloader, device)

print("Vision Only Loss:", vision_loss) 
print("Inertial Only Loss:", inertial_loss)
print("Visual Inertial Loss:", visual_inertial_loss)

def plot_raw_groundtruth(filepath):
    data = pd.read_csv(filepath)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data['x'], data['y'], data['z'])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Ground Truth Trajectory")
    plt.savefig("plots/ground_truth_trajectory.png")



def cumulative_poses(pred_rel_pose, gt_rel_pose):

    pred_positions = [tensor[:, :3] for tensor in pred_rel_pose]
    pred_rotations = [tensor[:, 3:] for tensor in pred_rel_pose]
    gt_positions = [tensor[:, :3] for tensor in gt_rel_pose]
    gt_rotations = [tensor[:, 3:] for tensor in gt_rel_pose]

    gt_positions = torch.stack(gt_positions).cpu().numpy()
    pred_positions = torch.stack(pred_positions).cpu().numpy()
    gt_rotations = torch.stack(gt_rotations).cpu().numpy()
    pred_rotations = torch.stack(pred_rotations).cpu().numpy()

    pred_positions = pred_positions.squeeze()
    pred_rotations = pred_rotations.squeeze()
    gt_positions = gt_positions.squeeze()
    gt_rotations = gt_rotations.squeeze()

    valid_pred_indices = np.linalg.norm(pred_rotations, axis=1) != 0

    pred_positions = pred_positions[valid_pred_indices]
    pred_rotations = pred_rotations[valid_pred_indices]
    gt_positions = gt_positions[valid_pred_indices]
    gt_rotations = gt_rotations[valid_pred_indices]

    # Initialize tensors to store the cumulative positions and rotations
    cumulative_positions = np.zeros_like(pred_positions)
    cumulative_rotations = np.zeros_like(pred_rotations)  # This may need adjustments if rotations need to be cumulative
    cumulative_gt_positions = np.zeros_like(gt_positions)
    cumulative_gt_rotations = np.zeros_like(gt_rotations)
    cumulative_positions[0] = gt_positions[0]
    cumulative_gt_positions[0] = gt_positions[0]
    # print(pred_rotations[0])
    #first one should be an identity
    cumulative_rotations[0] = gt_rotations[0]
    cumulative_gt_rotations[0] = gt_rotations[0]

    for i in range(1, len(pred_positions)):
        # Convert quaternion to rotation matrix, apply to the position
        prev_rot = Rotation.from_quat(cumulative_rotations[i-1])
        curr_rel_rot = Rotation.from_quat(pred_rotations[i])

        prev_gt_rot = Rotation.from_quat(cumulative_gt_rotations[i-1])
        curr_gt_rel_rot = Rotation.from_quat(gt_rotations[i])

        transformed_position = prev_rot.apply(pred_positions[i])

        transformed_gt_position = prev_gt_rot.apply(gt_positions[i])

        # Cumulative sum of positions
        cumulative_positions[i] = cumulative_positions[i-1] + transformed_position
        prev_rot_inv = prev_rot.inv()
        rot = curr_rel_rot * prev_rot
        cumulative_rotations[i] = rot.as_quat()


        cumulative_gt_positions[i] = cumulative_gt_positions[i-1] + transformed_gt_position
        prev_gt_rot_inv = prev_gt_rot.inv()
        gt_rot = curr_gt_rel_rot * prev_gt_rot
        cumulative_gt_rotations[i] = gt_rot.as_quat()

    return cumulative_positions, cumulative_rotations, cumulative_gt_positions, cumulative_gt_rotations


def plot_final_trajectory(positions, gt_positions, title="Cumulative Trajectory", type="combined"):
    """
    Plot a 3D trajectory from positions.
    positions: numpy array of shape (N, 3)
    """
    if type == "combined":
        title = "Cumulative Combined Trajectory"
    if type == "vision":
        title = "Cumulative Vision Only Trajectory"
    if type == "inertial":
        title = "Cumulative Inertial Only Trajectory"
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', linestyle='-', label="Prediction")
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], marker='^', linestyle='-', label="Ground Truth")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.savefig(f'plots/{title}.png')

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', linestyle='-', label="Prediction")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Prediction')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], marker='^', linestyle='-', label="Ground Truth")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Ground Truth')

    plt.savefig(f'plots/{title}_separate.png')
    # plt.show()



# Calculate cumulative poses
cumulative_positions, cumulative_rotations, cumulative_gt_positions, cumulative_gt_rotations = cumulative_poses(visual_inertial_pred_rel_pose, visual_inertial_gt_rel_pose)
cumulative_positions_vision, cumulative_rotations_vision, cumulative_gt_positions_vision, cumulative_gt_rotations_vision = cumulative_poses(vision_pred_rel_pose, vision_gt_rel_pose)
cumulative_positions_inertial, cumulative_rotations_inertial, cumulative_gt_positions_inertial, cumulative_gt_rotations_inertial = cumulative_poses(inertial_pred_rel_pose, inertial_gt_rel_pose)

ground_truth_data = pd.read_csv('Data/triangle-movement-with-ground-truth/ground_truth.csv')
zero_row_gt = pd.DataFrame([np.zeros(ground_truth_data.shape[1])], columns=ground_truth_data.columns)
ground_truth = pd.concat([zero_row_gt, ground_truth_data], ignore_index=True)
ground_truth_positions = ground_truth[['x', 'y', 'z']].values
ground_truth_orientations = ground_truth[['qw', 'qx', 'qy', 'qz']].values
# Plot the ground truth trajectory
plot_raw_groundtruth('Data/triangle-movement-with-ground-truth/ground_truth.csv')

plot_rotations(cumulative_gt_rotations_vision, cumulative_rotations_vision, "Vision Only Network - Rotation")
plot_rotations(cumulative_rotations_inertial, cumulative_rotations_inertial, "Inertial Only Network - Rotation")
plot_rotations(cumulative_gt_rotations, cumulative_rotations, "Visual Inertial Network - Rotation")

plot_trajectory(cumulative_gt_positions, cumulative_positions, "Visual Inertial Network - Translation")
plot_trajectory(cumulative_gt_positions_vision, cumulative_positions_vision, "Vision Only Network - Translation")
plot_trajectory(cumulative_gt_positions_inertial, cumulative_positions_inertial, "Inertial Only Network - Translation")

# Plot the trajectory
plot_final_trajectory(cumulative_positions, cumulative_gt_positions, type="combined")
plot_final_trajectory(cumulative_positions_vision, cumulative_gt_positions_vision, type="vision")
plot_final_trajectory(cumulative_positions_inertial, cumulative_gt_positions_inertial, type="inertial")