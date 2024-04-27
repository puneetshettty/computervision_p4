import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionDataNetwork(nn.Module):
    def __init__(self):
        super(VisionDataNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.01), 
            # nn.PReLU(),
            nn.ReLU(), # Original
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.01),
            # nn.PReLU(),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.LeakyReLU(0.01),
            # nn.PReLU(),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d(output_size=(6, 8)),
            nn.Flatten()
        )

        cnn_output_size = 128 * 6 * 8  # Calculate the output size of the last CNN layer (C * H * W)
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True  # Making LSTM bidirectional
        )

        self.dropout = nn.Dropout(0.5)  # Dropout to prevent overfitting
        self.fc = nn.Linear(128 * 2, 7)  # Adjust for bidirectional output

    def forward(self, img1, img2):
        # Process each image through the CNN
        cnn_img1 = self.cnn(img1)
        cnn_img2 = self.cnn(img2)

        # Prepare features for LSTM
        cnn_features = torch.stack([cnn_img1, cnn_img2], dim=1)  # Stack along sequence dimension

        # Pass features through LSTM
        lstm_out, _ = self.lstm(cnn_features)
        lstm_features = lstm_out[:, -1, :]  # Get the last time step features

        # Dropout and fully connected layer
        lstm_features = self.dropout(lstm_features)
        output = self.fc(lstm_features)

        return output



class InertialDataNetwork(nn.Module):
    def __init__(self):
        super(InertialDataNetwork, self).__init__()

        # LSTM to process the sequential inertial data
        self.lstm = nn.LSTM(
            input_size=6,  # Input size corresponds to IMU data dimensions
            hidden_size=128,
            num_layers=2,  # Using two layers for more complex sequence modeling
            batch_first=True
        )

        # Fully connected layers to map LSTM output to pose estimation
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            # nn.PReLU(),
            nn.ReLU(),
            nn.Linear(128, 7)  # Output: 3 position and 4 quaternion values
        )

    def forward(self, imu_data):
        # Process IMU data through LSTM
        # imu_data = imu_data.unsqueeze(1)  # Add a sequence dimension
        # print(imu_data.shape)
        _, (hidden_states, _) = self.lstm(imu_data)
        # print(hidden_states.shape)
        
        # Use the last layer's hidden state
        last_hidden_state = hidden_states[-1]
        # print(last_hidden_state.shape)

        # Predict the relative pose
        predicted_pose = self.fc(last_hidden_state)
        # print(predicted_pose.shape)

        # Normalize the quaternion part of the pose to ensure a valid quaternion
        quaternion = F.normalize(predicted_pose[:, 3:], dim=1)
        combined_pose = torch.cat((predicted_pose[:, :3], quaternion), dim=1)

        return combined_pose



class CombinedVIONetwork(nn.Module):
    def __init__(self):
        super(CombinedVIONetwork, self).__init__()

        # Subnetwork for processing visual data
        self.visual_subnetwork = VisionDataNetwork()
        # Subnetwork for processing inertial data
        self.inertial_subnetwork = InertialDataNetwork()

        # Fully connected layers to integrate and predict the final pose
        self.fc = nn.Sequential(
            nn.Linear(14, 256),  # 14 = 7 (vision) + 7 (inertial)
            # nn.PReLU(),
            nn.ReLU(),
            nn.Linear(256, 7)  # Output: 3 position and 4 quaternion values
        )

    def forward(self, img1, img2, imu_data):
        # Get pose estimates from the visual subnetwork
        visual_pose = self.visual_subnetwork(img1, img2)
        # Get pose estimates from the inertial subnetwork
        inertial_pose = self.inertial_subnetwork(imu_data)

        # Concatenate the outputs from both subnetworks
        combined_features = torch.cat((visual_pose, inertial_pose), dim=1)
        # Predict the final relative pose
        final_pose = self.fc(combined_features)

        return final_pose