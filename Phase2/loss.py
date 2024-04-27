import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

class PoseLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(PoseLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, predicted_pose, gt_pose):
        # Calculate position losses
        position_mse = self.mse_loss(predicted_pose[:, :3], gt_pose[:, :3])
        position_cosine_loss = 1 - self.cosine_similarity(predicted_pose[:, :3], gt_pose[:, :3]).mean()

        # Calculate orientation losses
        orientation_mse = self.mse_loss(predicted_pose[:, 3:], gt_pose[:, 3:])
        orientation_cosine_loss = 1 - self.cosine_similarity(predicted_pose[:, 3:], gt_pose[:, 3:]).mean()

        # Combine position and orientation losses
        combined_position_loss = self.alpha * position_mse + (1 - self.alpha) * position_cosine_loss
        combined_orientation_loss = self.alpha * orientation_mse + (1 - self.alpha) * orientation_cosine_loss

        return combined_position_loss + combined_orientation_loss

class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def forward(self, prediction, target):
        batch_size = prediction.shape[0]

        # print("Prediction shape:", prediction.shape)
        # print("Target shape:", target.shape)

        prediction_rotations = R.from_quat(prediction.view(batch_size, 4).detach().cpu().numpy())
        target_rotations = R.from_quat(target.view(batch_size, 4).detach().cpu().numpy())

        geodesic_distances = prediction_rotations.inv() * target_rotations
        # print("Geodesic distances shape:", geodesic_distances.shape)
        # print("target_rotations shape:", target_rotations.shape)
        # print("prediction_rotations shape:", prediction_rotations.shape)
        geodesic_angles = geodesic_distances.magnitude()

        geodesic_loss = torch.mean(torch.tensor(geodesic_angles ** 2, device=prediction.device))
        
        return geodesic_loss


class TransformationLoss(nn.Module):
    def __init__(self, translation_weight=1.0, rotation_weight=1.0):
        super(TransformationLoss, self).__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.translation_loss = nn.MSELoss()
        self.rotation_loss = GeodesicLoss()

    def forward(self, prediction, target):
        # Separate translation and rotation components
        translation_prediction = prediction[:, :3]
        translation_target = target[:, :3]
        rotation_prediction = prediction[:, 3:]
        rotation_target = target[:, 3:]

        # Calculate translation and rotation losses
        translation_loss = self.translation_loss(translation_prediction, translation_target)
        rotation_loss = self.rotation_loss(rotation_prediction, rotation_target)

        total_loss = self.translation_weight * translation_loss + self.rotation_weight * rotation_loss

        # Weighted sum of translation and rotation losses
        return total_loss


