import numpy as np
from scipy.stats import chi2

from utils import *
from feature import Feature

import time
from collections import namedtuple



class IMUState(object):
    # id for next IMU state
    next_id = 0

    # Gravity vector in the world frame
    gravity = np.array([0., 0., -9.81])

    # Transformation offset from the IMU frame to the body frame. 
    # The transformation takes a vector from the IMU frame to the 
    # body frame. The z axis of the body frame should point upwards.
    # Normally, this transform should be identity.
    T_imu_body = Isometry3d(np.identity(3), np.zeros(3))

    def __init__(self, new_id=None):
        # An unique identifier for the IMU state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the IMU (body) frame.
        self.orientation = np.array([0., 0., 0., 1.])

        # Position of the IMU (body) frame in the world frame.
        self.position = np.zeros(3)
        # Velocity of the IMU (body) frame in the world frame.
        self.velocity = np.zeros(3)

        # Bias for measured angular velocity and acceleration.
        self.gyro_bias = np.zeros(3)
        self.acc_bias = np.zeros(3)

        # These three variables should have the same physical
        # interpretation with `orientation`, `position`, and
        # `velocity`. There three variables are used to modify
        # the transition matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0., 0., 0., 1.])
        self.position_null = np.zeros(3)
        self.velocity_null = np.zeros(3)

        # Transformation between the IMU and the left camera (cam0)
        self.R_imu_cam0 = np.identity(3)
        self.t_cam0_imu = np.zeros(3)


class CAMState(object):
    # Takes a vector from the cam0 frame to the cam1 frame.
    R_cam0_cam1 = None
    t_cam0_cam1 = None

    def __init__(self, new_id=None):
        # An unique identifier for the CAM state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the camera frame.
        self.orientation = np.array([0., 0., 0., 1.])

        # Position of the camera frame in the world frame.
        self.position = np.zeros(3)

        # These two variables should have the same physical
        # interpretation with `orientation` and `position`.
        # There two variables are used to modify the measurement
        # Jacobian matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0., 0., 0., 1.])
        self.position_null = np.zeros(3)

        
class StateServer(object):
    """
    Store one IMU states and several camera states for constructing 
    measurement model.
    """
    def __init__(self):
        self.imu_state = IMUState()
        self.cam_states = dict()   # <CAMStateID, CAMState>, ordered dict

        # State covariance matrix
        self.state_cov = np.zeros((21, 21))
        self.continuous_noise_cov = np.zeros((12, 12))



class MSCKF(object):
    def __init__(self, config):
        self.config = config
        self.optimization_config = config.optimization_config

        # IMU data buffer
        # This is buffer is used to handle the unsynchronization or
        # transfer delay between IMU and Image messages.
        self.imu_msg_buffer = []

        # State vector
        self.state_server = StateServer()
        # Features used
        self.map_server = dict()   # <FeatureID, Feature>

        # Chi squared test table.
        # Initialize the chi squared test table with confidence level 0.95.
        self.chi_squared_test_table = dict()
        for i in range(1, 100):
            self.chi_squared_test_table[i] = chi2.ppf(0.05, i)

        # Set the initial IMU state.
        # The intial orientation and position will be set to the origin implicitly.
        # But the initial velocity and bias can be set by parameters.
        # TODO: is it reasonable to set the initial bias to 0?
        self.state_server.imu_state.velocity = config.velocity
        self.reset_state_cov()

        continuous_noise_cov = np.identity(12)
        continuous_noise_cov[:3, :3] *= self.config.gyro_noise
        continuous_noise_cov[3:6, 3:6] *= self.config.gyro_bias_noise
        continuous_noise_cov[6:9, 6:9] *= self.config.acc_noise
        continuous_noise_cov[9:, 9:] *= self.config.acc_bias_noise
        self.state_server.continuous_noise_cov = continuous_noise_cov

        # Gravity vector in the world frame
        IMUState.gravity = config.gravity

        # Transformation between the IMU and the left camera (cam0)
        T_cam0_imu = np.linalg.inv(config.T_imu_cam0)
        self.state_server.imu_state.R_imu_cam0 = T_cam0_imu[:3, :3].T
        self.state_server.imu_state.t_cam0_imu = T_cam0_imu[:3, 3]

        # Extrinsic parameters of camera and IMU.
        T_cam0_cam1 = config.T_cn_cnm1
        CAMState.R_cam0_cam1 = T_cam0_cam1[:3, :3]
        CAMState.t_cam0_cam1 = T_cam0_cam1[:3, 3]
        Feature.R_cam0_cam1 = CAMState.R_cam0_cam1
        Feature.t_cam0_cam1 = CAMState.t_cam0_cam1
        IMUState.T_imu_body = Isometry3d(
            config.T_imu_body[:3, :3],
            config.T_imu_body[:3, 3])

        # Tracking rate.
        self.tracking_rate = None

        # Indicate if the gravity vector is set.
        self.is_gravity_set = False
        # Indicate if the received image is the first one. The system will 
        # start after receiving the first image.
        self.is_first_img = True

    def imu_callback(self, imu_msg):
        """
        Callback function for the imu message.
        """
        # IMU msgs are pushed backed into a buffer instead of being processed 
        # immediately. The IMU msgs are processed when the next image is  
        # available, in which way, we can easily handle the transfer delay.
        self.imu_msg_buffer.append(imu_msg)

        if not self.is_gravity_set:
            if len(self.imu_msg_buffer) >= 200:
                self.initialize_gravity_and_bias()
                self.is_gravity_set = True

    def feature_callback(self, feature_msg):
        """
        Callback function for feature measurements.
        """
        if not self.is_gravity_set:
            return
        start = time.time()

        # Start the system if the first image is received.
        # The frame where the first image is received will be the origin.
        if self.is_first_img:
            self.is_first_img = False
            self.state_server.imu_state.timestamp = feature_msg.timestamp

        t = time.time()

        # Propogate the IMU state.
        # that are received before the image msg.
        self.batch_imu_processing(feature_msg.timestamp)

        print('---batch_imu_processing    ', time.time() - t)
        t = time.time()

        # Augment the state vector.
        self.state_augmentation(feature_msg.timestamp)

        print('---state_augmentation      ', time.time() - t)
        t = time.time()

        # Add new observations for existing features or new features 
        # in the map server.
        self.add_feature_observations(feature_msg)

        print('---add_feature_observations', time.time() - t)
        t = time.time()

        # Perform measurement update if necessary.
        # And prune features and camera states.
        self.remove_lost_features()

        print('---remove_lost_features    ', time.time() - t)
        t = time.time()

        self.prune_cam_state_buffer()

        print('---prune_cam_state_buffer  ', time.time() - t)
        print('---msckf elapsed:          ', time.time() - start, f'({feature_msg.timestamp})')

        try:
            # Publish the odometry.
            return self.publish(feature_msg.timestamp)
        finally:
            # Reset the system if necessary.
            self.online_reset()



    def initialize_gravity_and_bias(self):
        """
        Estimate and initialize the gyroscopic bias and the IMU's initial orientation.
        The gyroscopic bias is estimated as the average angular velocity over the initial readings.

        Initialize the IMU bias and initial orientation based on the 
        first few IMU readings.

        The initial orientation is set by aligning the estimated gravity vector with the true gravity direction.
        """
        # Accumulators for gyroscopic angular velocities and accelerometer linear accelerations
        accumulated_angular_velocity = np.zeros(3)
        accumulated_linear_acceleration = np.zeros(3)
        
        # Sum the angular velocity and linear acceleration from all IMU messages in the buffer
        for imu_message in self.imu_msg_buffer:
            accumulated_angular_velocity += imu_message.angular_velocity
            accumulated_linear_acceleration += imu_message.linear_acceleration

        # Calculate the average gyroscopic bias from the accumulated angular velocities
        average_angular_velocity = accumulated_angular_velocity / len(self.imu_msg_buffer)
        self.state_server.imu_state.gyro_bias = average_angular_velocity

        # Calculate the average gravity vector from the accumulated linear accelerations
        average_gravity_vector = accumulated_linear_acceleration / len(self.imu_msg_buffer)

        # Normalize the average gravity vector to obtain its magnitude
        gravity_vector_magnitude = np.linalg.norm(average_gravity_vector)

        # Set the IMUState's gravity vector to point downwards with the calculated magnitude
        IMUState.gravity = np.array([0., 0., -gravity_vector_magnitude])

        # Compute the initial IMU orientation by aligning the estimated gravity direction
        # with the true direction of gravity (downwards)
        # The 'from_two_vectors' function computes the quaternion representing the rotation
        # that aligns the estimated gravity direction with the negative z-axis
        self.state_server.imu_state.orientation = from_two_vectors(-IMUState.gravity,average_gravity_vector)

    
    def batch_imu_processing(self, time_bound):
        """
        Processes all IMU messages up to a specified time bound.
        Updates the system's IMU state by applying the process model to each relevant IMU message.

        Parameters:
        - time_bound (float): The upper time limit until which IMU messages will be processed.
        """
        # Count the number of IMU messages used in this batch
        used_imu_msg_count = 0

        # Process each IMU message in the buffer
        for imu_msg in self.imu_msg_buffer:
            imu_msg_time = imu_msg.timestamp

            # Skip any messages that are timestamped before the last recorded state timestamp
            if imu_msg_time < self.state_server.imu_state.timestamp:
                used_imu_msg_count += 1
                continue

            # Stop processing if the message timestamp exceeds the time bound
            if imu_msg_time > time_bound:
                break

            # Apply the process model to update the state
            self.process_model(imu_msg_time, imu_msg.angular_velocity, imu_msg.linear_acceleration)
            self.state_server.imu_state.timestamp = imu_msg_time
            used_imu_msg_count += 1

        # Update the current IMU state ID and increment the global next ID after processing
        self.state_server.imu_state.id = IMUState.next_id
        IMUState.next_id += 1

        # Remove all processed IMU messages from the buffer
        self.imu_msg_buffer = self.imu_msg_buffer[used_imu_msg_count:]

    def process_model(self, time, m_gyro, m_acc):
        """
        IMPLEMENT THIS!!!!!
        """
        """
        Section III.A: The dynamics of the error IMU state following equation (2) in the "MSCKF" paper.
        """
        # Reference to IMU state for easier access
        imu_state = self.state_server.imu_state

        # Compensate for the biases in gyro and accelerometer measurements
        corrected_gyro = measured_gyro - imu_state.gyro_bias
        corrected_acc = measured_acc - imu_state.acc_bias
        
        # Time step calculation
        time_step = current_time - imu_state.timestamp
        imu_state.timestamp = current_time

        # Conversion of orientation from quaternion to rotation matrix
        world_to_imu_rotation = to_rotation(imu_state.orientation)

        # Initialize the state transition and noise mapping matrices
        state_transition = np.zeros((21, 21))
        noise_mapping = np.zeros((21, 12))

        # Fill the state transition matrix based on the IMU measurements
        state_transition[:3, :3] = -skew(corrected_gyro)
        state_transition[:3, 3:6] = -np.identity(3)
        state_transition[6:9, :3] = -world_to_imu_rotation.T @ skew(corrected_acc)
        state_transition[6:9, 9:12] = -world_to_imu_rotation.T
        state_transition[12:15, 6:9] = np.identity(3)

        # Fill the noise mapping matrix
        noise_mapping[:3, :3] = -np.identity(3)
        noise_mapping[3:6, 3:6] = np.identity(3)
        noise_mapping[6:9, 6:9] = -world_to_imu_rotation.T
        noise_mapping[9:12, 9:12] = np.identity(3)

        # Approximate matrix exponential using a 3rd order Taylor expansion for propagation
        Fdt = state_transition * time_step
        Fdt_square = Fdt @ Fdt
        Fdt_cube = Fdt_square @ Fdt
        transition_matrix = np.identity(21) + Fdt + Fdt_square / 2.0 + Fdt_cube / 6.0

        # State propagation using 4th order Runge-Kutta method
        self.predict_new_state(time_step, corrected_gyro, corrected_acc)

        # Adjust the transition matrix for the null space constraints
        initial_orientation_matrix = to_rotation(imu_state.orientation_null)
        transition_matrix[:3, :3] = world_to_imu_rotation @ initial_orientation_matrix.T

        gravity_vector_projected = initial_orientation_matrix @ IMUState.gravity
        gravity_projection_scalar = gravity_vector_projected / (gravity_vector_projected @ gravity_vector_projected)

        # Correct the position-related parts of the transition matrix
        position_correction_factor = skew(time_step * imu_state.velocity_null + imu_state.position_null - imu_state.position) @ IMUState.gravity
        transition_matrix[12:15, :3] -= (transition_matrix[12:15, :3] @ gravity_vector_projected - position_correction_factor)[:, None] * gravity_projection_scalar

        # Propagate the state covariance matrix
        process_noise_covariance = transition_matrix @ noise_mapping @ self.state_server.continuous_noise_cov @ noise_mapping.T @ transition_matrix.T * time_step
        self.state_server.state_cov[:21, :21] = (transition_matrix @ self.state_server.state_cov[:21, :21] @ transition_matrix.T + process_noise_covariance)

        # Update cross-covariance if camera states are present
        if len(self.state_server.cam_states) > 0:
            self.state_server.state_cov[:21, 21:] = transition_matrix @ self.state_server.state_cov[:21, 21:]
            self.state_server.state_cov[21:, :21] = self.state_server.state_cov[21:, :21] @ transition_matrix.T

        # Ensure the covariance matrix is symmetric
        self.state_server.state_cov = (self.state_server.state_cov + self.state_server.state_cov.T) / 2

        # Update the null space components of the state
        imu_state.orientation_null = imu_state.orientation
        imu_state.position_null = imu_state.position
        imu_state.velocity_null = imu_state.velocity
            

    def predict_new_state(self, dt, gyro, acc):
        """
        Propagate the IMU state using 4th order Runge-Kutta integration.
        This method integrates the orientation, velocity, and position of the IMU based on gyroscopic and accelerometer inputs.
        """
        # Compute the norm of the gyroscopic measurements
        gyro_norm = np.linalg.norm(gyro)
        
        # Define the Omega matrix used for quaternion integration
        omega_matrix = np.zeros((4, 4))
        omega_matrix[:3, :3] = -skew(gyro)  # Populate the skew-symmetric part
        omega_matrix[:3, 3] = gyro  # Right column for quaternion multiplication
        omega_matrix[3, :3] = -gyro  # Bottom row for quaternion multiplication
        
        # Extract the current orientation, velocity, and position from the state
        current_orientation = self.state_server.imu_state.orientation
        current_velocity = self.state_server.imu_state.velocity
        current_position = self.state_server.imu_state.position
        
        # Determine the first order and half order quaternion derivatives for RK4
        if gyro_norm > 1e-5:
            dq_dt = (np.cos(gyro_norm * dt * 0.5) * np.identity(4) +
                    np.sin(gyro_norm * dt * 0.5) / gyro_norm * omega_matrix) @ current_orientation
            dq_dt2 = (np.cos(gyro_norm * dt * 0.25) * np.identity(4) +
                    np.sin(gyro_norm * dt * 0.25) / gyro_norm * omega_matrix) @ current_orientation
        else:
            # Use first order Taylor expansion for small angles to avoid division by zero
            dq_dt = (np.identity(4) + omega_matrix * dt * 0.5) @ current_orientation
            dq_dt2 = (np.identity(4) + omega_matrix * dt * 0.25) @ current_orientation

        # Convert quaternion derivatives to rotation matrices
        rotation_matrix_dq_dt_T = to_rotation(dq_dt).T
        rotation_matrix_dq_dt2_T = to_rotation(dq_dt2).T

        # Apply 4th order Runge-Kutta for position and velocity integration
        k1_velocity_derivative = current_velocity
        k1_position_derivative = to_rotation(current_orientation).T @ acc + IMUState.gravity

        k1_velocity = current_velocity + k1_velocity_derivative * dt / 2.
        k2_position_derivative = k1_velocity
        k2_velocity_derivative = rotation_matrix_dq_dt2_T @ acc + IMUState.gravity
        
        k2_velocity = current_velocity + k2_velocity_derivative * dt / 2
        k3_position_derivative = k2_velocity
        k3_velocity_derivative = rotation_matrix_dq_dt2_T @ acc + IMUState.gravity
        
        k3_velocity = current_velocity + k3_velocity_derivative * dt
        k4_position_derivative = k3_velocity
        k4_velocity_derivative = rotation_matrix_dq_dt_T @ acc + IMUState.gravity

        # Update state variables with RK4 integration results
        final_orientation = dq_dt / np.linalg.norm(dq_dt)  # Normalize the quaternion to maintain unit length
        final_velocity = current_velocity + (k1_velocity_derivative + 2 * k2_velocity_derivative + 2 * k3_velocity_derivative + k4_velocity_derivative) * dt / 6.
        final_position = current_position + (k1_position_derivative + 2 * k2_position_derivative + 2 * k3_position_derivative + k4_position_derivative) * dt / 6.

        # Update the state server with the new IMU state
        self.state_server.imu_state.orientation = final_orientation
        self.state_server.imu_state.velocity = final_velocity
        self.state_server.imu_state.position = final_position

    
    def state_augmentation(self, current_time):
        """
        Augment the state covariance matrix to include a new camera state based on the current IMU state.

        Compute the state covariance matrix in equation (3) in the "MSCKF" paper.
        """
        imu_state = self.state_server.imu_state
        rotation_imu_to_camera = imu_state.R_imu_cam0
        translation_camera_to_imu = imu_state.t_cam0_imu

        # Compute the camera's orientation and position in the world frame
        rotation_world_to_imu = to_rotation(imu_state.orientation)
        rotation_world_to_camera = rotation_imu_to_camera @ rotation_world_to_imu
        translation_camera_to_world = imu_state.position + rotation_world_to_imu.T @ translation_camera_to_imu

        # Create a new camera state and initialize it
        new_cam_state = CAMState(imu_state.id)
        new_cam_state.timestamp = current_time
        new_cam_state.orientation = to_quaternion(rotation_world_to_camera)
        new_cam_state.position = translation_camera_to_world
        new_cam_state.orientation_null = new_cam_state.orientation
        new_cam_state.position_null = new_cam_state.position
        self.state_server.cam_states[imu_state.id] = new_cam_state

        # Augment the state covariance matrix
        num_existing_states = self.state_server.state_cov.shape[0]
        new_cov_size = num_existing_states + 6
        augmented_covariance = np.zeros((new_cov_size, new_cov_size))
        augmented_covariance[:num_existing_states, :num_existing_states] = self.state_server.state_cov

        # Jacobian matrix for the transformation from IMU state to camera state
        jacobian = np.zeros((6, 21))
        jacobian[:3, :3] = rotation_imu_to_camera
        jacobian[:3, 15:18] = np.identity(3)
        jacobian[3:6, :3] = skew(rotation_world_to_imu.T @ translation_camera_to_imu)
        jacobian[3:6, 12:15] = np.identity(3)
        jacobian[3:6, 18:21] = np.identity(3)

        # Update cross-correlation terms in the covariance matrix
        augmented_covariance[num_existing_states:, :num_existing_states] = jacobian @ augmented_covariance[:21, :num_existing_states]
        augmented_covariance[:num_existing_states, num_existing_states:] = augmented_covariance[num_existing_states:, :num_existing_states].T
        augmented_covariance[num_existing_states:, num_existing_states:] = jacobian @ augmented_covariance[:21, :21] @ jacobian.T

        # Ensure the augmented covariance matrix is symmetric
        self.state_server.state_cov = (augmented_covariance + augmented_covariance.T) / 2

    def add_feature_observations(self, feature_msg):
        """
        Update the map of features with new observations from the feature message.
        This function tracks new and existing features and updates the tracking rate.
        """
        # Get the current IMU state ID and the number of features currently being tracked
        state_id = self.state_server.imu_state.id
        curr_feature_count = len(self.map_server)
        tracked_feature_count = 0

        # Process each feature in the message
        for feature in feature_msg.features:
            feature_observation = np.array([feature.u0, feature.v0, feature.u1, feature.v1])
            if feature.id not in self.map_server:
                # Initialize a new Feature object for new features
                new_feature = Feature(feature.id, self.optimization_config)
                new_feature.observations[state_id] = feature_observation
                self.map_server[feature.id] = new_feature
            else:
                # Update existing feature with new observations
                self.map_server[feature.id].observations[state_id] = feature_observation
                tracked_feature_count += 1

        # Update the tracking rate, handling the case where no features are currently tracked
        if curr_feature_count > 0:
            self.tracking_rate = tracked_feature_count / curr_feature_count
        else:
            self.tracking_rate = tracked_feature_count / 1e-5  # Avoid division by zero if no features are currently tracked

    def measurement_jacobian(self, cam_state_id, feature_id):
        """
        This function is used to compute the measurement Jacobian
        for a single feature observed at a single camera frame.
        """
        # Prepare all the required data.
        cam_state = self.state_server.cam_states[cam_state_id]
        feature = self.map_server[feature_id]

        # Cam0 pose.
        R_w_c0 = to_rotation(cam_state.orientation)
        t_c0_w = cam_state.position

        # Cam1 pose.
        R_w_c1 = CAMState.R_cam0_cam1 @ R_w_c0
        t_c1_w = t_c0_w - R_w_c1.T @ CAMState.t_cam0_cam1

        # 3d feature position in the world frame.
        # And its observation with the stereo cameras.
        p_w = feature.position
        z = feature.observations[cam_state_id]

        # Convert the feature position from the world frame to
        # the cam0 and cam1 frame.
        p_c0 = R_w_c0 @ (p_w - t_c0_w)
        p_c1 = R_w_c1 @ (p_w - t_c1_w)

        # Compute the Jacobians.
        dz_dpc0 = np.zeros((4, 3))
        dz_dpc0[0, 0] = 1 / p_c0[2]
        dz_dpc0[1, 1] = 1 / p_c0[2]
        dz_dpc0[0, 2] = -p_c0[0] / (p_c0[2] * p_c0[2])
        dz_dpc0[1, 2] = -p_c0[1] / (p_c0[2] * p_c0[2])

        dz_dpc1 = np.zeros((4, 3))
        dz_dpc1[2, 0] = 1 / p_c1[2]
        dz_dpc1[3, 1] = 1 / p_c1[2]
        dz_dpc1[2, 2] = -p_c1[0] / (p_c1[2] * p_c1[2])
        dz_dpc1[3, 2] = -p_c1[1] / (p_c1[2] * p_c1[2])

        dpc0_dxc = np.zeros((3, 6))
        dpc0_dxc[:, :3] = skew(p_c0)
        dpc0_dxc[:, 3:] = -R_w_c0

        dpc1_dxc = np.zeros((3, 6))
        dpc1_dxc[:, :3] = CAMState.R_cam0_cam1 @ skew(p_c0)
        dpc1_dxc[:, 3:] = -R_w_c1

        dpc0_dpg = R_w_c0
        dpc1_dpg = R_w_c1

        H_x = dz_dpc0 @ dpc0_dxc + dz_dpc1 @ dpc1_dxc   # shape: (4, 6)
        H_f = dz_dpc0 @ dpc0_dpg + dz_dpc1 @ dpc1_dpg   # shape: (4, 3)

        # Modifty the measurement Jacobian to ensure observability constrain.
        A = H_x   # shape: (4, 6)
        u = np.zeros(6)
        u[:3] = to_rotation(cam_state.orientation_null) @ IMUState.gravity
        u[3:] = skew(p_w - cam_state.position_null) @ IMUState.gravity

        H_x = A - (A @ u)[:, None] * u / (u @ u)
        H_f = -H_x[:4, 3:6]

        # Compute the residual.
        r = z - np.array([*p_c0[:2]/p_c0[2], *p_c1[:2]/p_c1[2]])

        # H_x: shape (4, 6)
        # H_f: shape (4, 3)
        # r  : shape (4,)
        return H_x, H_f, r

    def feature_jacobian(self, feature_id, cam_state_ids):
        """
        This function computes the Jacobian of all measurements viewed 
        in the given camera states of this feature.
        """
        feature = self.map_server[feature_id]

        # Check how many camera states in the provided camera id 
        # camera has actually seen this feature.
        valid_cam_state_ids = []
        for cam_id in cam_state_ids:
            if cam_id in feature.observations:
                valid_cam_state_ids.append(cam_id)

        jacobian_row_size = 4 * len(valid_cam_state_ids)

        cam_states = self.state_server.cam_states
        H_xj = np.zeros((jacobian_row_size, 
            21+len(self.state_server.cam_states)*6))
        H_fj = np.zeros((jacobian_row_size, 3))
        r_j = np.zeros(jacobian_row_size)

        stack_count = 0
        for cam_id in valid_cam_state_ids:
            H_xi, H_fi, r_i = self.measurement_jacobian(cam_id, feature.id)

            # Stack the Jacobians.
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            H_xj[stack_count:stack_count+4, 21+6*idx:21+6*(idx+1)] = H_xi
            H_fj[stack_count:stack_count+4, :3] = H_fi
            r_j[stack_count:stack_count+4] = r_i
            stack_count += 4

        # Project the residual and Jacobians onto the nullspace of H_fj.
        # svd of H_fj
        U, _, _ = np.linalg.svd(H_fj)
        A = U[:, 3:]

        H_x = A.T @ H_xj
        r = A.T @ r_j

        return H_x, r

    def measurement_update(self, H, r):
        """
        Perform the measurement update for the Multi-State Constraint Kalman Filter (MSCKF).
        This method updates the state of the system using the provided Jacobian matrix (H)
        and residual vector (r), applying corrections based on observations.

        Section III.B: by stacking multiple observations, we can compute the residuals in equation (6) in "MSCKF" paper 

        Parameters:
        - H (numpy.array): Stacked Jacobian matrix from all relevant observations.
        - r (numpy.array): Stacked residual vector from all relevant observations.
        """
        # Ensure that neither the Jacobian nor the residuals are empty
        if H.size == 0 or r.size == 0:
            print("Warning: Jacobian or residuals are empty. No update performed.")
            return

        # Optimize the computation by reducing the dimensionality when appropriate
        if H.shape[0] > H.shape[1]:
            # Use QR decomposition to reduce the Jacobian for tall matrices
            Q, R = np.linalg.qr(H)
            reduced_jacobian = R
            transformed_residuals = Q.T @ r
        else:
            reduced_jacobian = H
            transformed_residuals = r

        # Retrieve the current state covariance matrix
        current_covariance = self.state_server.state_cov

        # Compute the innovation covariance matrix and the Kalman gain
        observation_noise_matrix = self.config.observation_noise * np.eye(reduced_jacobian.shape[0])
        innovation_covariance = reduced_jacobian @ current_covariance @ reduced_jacobian.T + observation_noise_matrix
        kalman_gain = np.linalg.solve(innovation_covariance, reduced_jacobian @ current_covariance).T

        # Apply the Kalman gain to the residuals to compute the state update
        state_update = kalman_gain @ transformed_residuals

        # Apply state updates to the IMU and camera states within the filter
        imu_state_update = state_update[:21]
        camera_state_updates = state_update[21:]

        # Update IMU state elements
        imu_state = self.state_server.imu_state
        imu_state.orientation = quaternion_multiplication(small_angle_quaternion(imu_state_update[:3]), imu_state.orientation)
        imu_state.gyro_bias += imu_state_update[3:6]
        imu_state.velocity += imu_state_update[6:9]
        imu_state.acc_bias += imu_state_update[9:12]
        imu_state.position += imu_state_update[12:15]

        # Update camera states
        for i in range(len(self.state_server.cam_states)):
            cam_state = self.state_server.cam_states[i]
            cam_update = camera_state_updates[i*6:(i+1)*6]
            cam_state.orientation = quaternion_multiplication(small_angle_quaternion(cam_update[:3]), cam_state.orientation)
            cam_state.position += cam_update[3:6]

        # Update the covariance matrix
        identity_matrix = np.eye(current_covariance.shape[0])
        updated_covariance = (identity_matrix - kalman_gain @ reduced_jacobian) @ current_covariance

        # Ensure the covariance matrix is symmetric
        self.state_server.state_cov = (updated_covariance + updated_covariance.T) / 2

    def gating_test(self, H, r, dof):
        P1 = H @ self.state_server.state_cov @ H.T
        P2 = self.config.observation_noise * np.identity(len(H))
        gamma = r @ np.linalg.solve(P1+P2, r)

        if(gamma < self.chi_squared_test_table[dof]):
            return True
        else:
            return False

    def remove_lost_features(self):
        # Remove the features that lost track.
        # BTW, find the size the final Jacobian matrix and residual vector.
        jacobian_row_size = 0
        invalid_feature_ids = []
        processed_feature_ids = []

        for feature in self.map_server.values():
            # Pass the features that are still being tracked.
            if self.state_server.imu_state.id in feature.observations:
                continue
            if len(feature.observations) < 3:
                invalid_feature_ids.append(feature.id)
                continue

            # Check if the feature can be initialized if it has not been.
            if not feature.is_initialized:
                # Ensure there is enough translation to triangulate the feature
                if not feature.check_motion(self.state_server.cam_states):
                    invalid_feature_ids.append(feature.id)
                    continue

                # Intialize the feature position based on all current available 
                # measurements.
                ret = feature.initialize_position(self.state_server.cam_states)
                if ret is False:
                    invalid_feature_ids.append(feature.id)
                    continue

            jacobian_row_size += (4 * len(feature.observations) - 3)
            processed_feature_ids.append(feature.id)

        # Remove the features that do not have enough measurements.
        for feature_id in invalid_feature_ids:
            del self.map_server[feature_id]

        # Return if there is no lost feature to be processed.
        if len(processed_feature_ids) == 0:
            return

        H_x = np.zeros((jacobian_row_size, 
            21+6*len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)
        stack_count = 0

        # Process the features which lose track.
        for feature_id in processed_feature_ids:
            feature = self.map_server[feature_id]

            cam_state_ids = []
            for cam_id, measurement in feature.observations.items():
                cam_state_ids.append(cam_id)

            H_xj, r_j = self.feature_jacobian(feature.id, cam_state_ids)

            if self.gating_test(H_xj, r_j, len(cam_state_ids)-1):
                H_x[stack_count:stack_count+H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count+len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            # Put an upper bound on the row size of measurement Jacobian,
            # which helps guarantee the executation time.
            if stack_count > 1500:
                break

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform the measurement update step.
        self.measurement_update(H_x, r)

        # Remove all processed features from the map.
        for feature_id in processed_feature_ids:
            del self.map_server[feature_id]

    def find_redundant_cam_states(self):
        # Move the iterator to the key position.
        cam_state_pairs = list(self.state_server.cam_states.items())

        key_cam_state_idx = len(cam_state_pairs) - 4
        cam_state_idx = key_cam_state_idx + 1
        first_cam_state_idx = 0

        # Pose of the key camera state.
        key_position = cam_state_pairs[key_cam_state_idx][1].position
        key_rotation = to_rotation(
            cam_state_pairs[key_cam_state_idx][1].orientation)

        rm_cam_state_ids = []

        # Mark the camera states to be removed based on the
        # motion between states.
        for i in range(2):
            position = cam_state_pairs[cam_state_idx][1].position
            rotation = to_rotation(
                cam_state_pairs[cam_state_idx][1].orientation)
            
            distance = np.linalg.norm(position - key_position)
            angle = 2 * np.arccos(to_quaternion(
                rotation @ key_rotation.T)[-1])

            if angle < 0.2618 and distance < 0.4 and self.tracking_rate > 0.5:
                rm_cam_state_ids.append(cam_state_pairs[cam_state_idx][0])
                cam_state_idx += 1
            else:
                rm_cam_state_ids.append(cam_state_pairs[first_cam_state_idx][0])
                first_cam_state_idx += 1
                cam_state_idx += 1

        # Sort the elements in the output list.
        rm_cam_state_ids = sorted(rm_cam_state_ids)
        return rm_cam_state_ids


    def prune_cam_state_buffer(self):
        if len(self.state_server.cam_states) < self.config.max_cam_state_size:
            return

        # Find two camera states to be removed.
        rm_cam_state_ids = self.find_redundant_cam_states()

        # Find the size of the Jacobian matrix.
        jacobian_row_size = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue
            if len(involved_cam_state_ids) == 1:
                del feature.observations[involved_cam_state_ids[0]]
                continue

            if not feature.is_initialized:
                # Check if the feature can be initialize.
                if not feature.check_motion(self.state_server.cam_states):
                    # If the feature cannot be initialized, just remove
                    # the observations associated with the camera states
                    # to be removed.
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

                ret = feature.initialize_position(self.state_server.cam_states)
                if ret is False:
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

            jacobian_row_size += 4*len(involved_cam_state_ids) - 3

        # Compute the Jacobian and residual.
        H_x = np.zeros((jacobian_row_size, 21+6*len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)

        stack_count = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue

            H_xj, r_j = self.feature_jacobian(feature.id, involved_cam_state_ids)

            if self.gating_test(H_xj, r_j, len(involved_cam_state_ids)):
                H_x[stack_count:stack_count+H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count+len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            for cam_id in involved_cam_state_ids:
                del feature.observations[cam_id]

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform measurement update.
        self.measurement_update(H_x, r)

        for cam_id in rm_cam_state_ids:
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            cam_state_start = 21 + 6*idx
            cam_state_end = cam_state_start + 6

            # Remove the corresponding rows and columns in the state
            # covariance matrix.
            state_cov = self.state_server.state_cov.copy()
            if cam_state_end < state_cov.shape[0]:
                size = state_cov.shape[0]
                state_cov[cam_state_start:-6, :] = state_cov[cam_state_end:, :]
                state_cov[:, cam_state_start:-6] = state_cov[:, cam_state_end:]
            self.state_server.state_cov = state_cov[:-6, :-6]

            # Remove this camera state in the state vector.
            del self.state_server.cam_states[cam_id]

    def reset_state_cov(self):
        """
        Reset the state covariance.
        """
        state_cov = np.zeros((21, 21))
        state_cov[ 3: 6,  3: 6] = self.config.gyro_bias_cov * np.identity(3)
        state_cov[ 6: 9,  6: 9] = self.config.velocity_cov * np.identity(3)
        state_cov[ 9:12,  9:12] = self.config.acc_bias_cov * np.identity(3)
        state_cov[15:18, 15:18] = self.config.extrinsic_rotation_cov * np.identity(3)
        state_cov[18:21, 18:21] = self.config.extrinsic_translation_cov * np.identity(3)
        self.state_server.state_cov = state_cov

    def reset(self):
        """
        Reset the VIO to initial status.
        """
        # Reset the IMU state.
        imu_state = IMUState()
        imu_state.id = self.state_server.imu_state.id
        imu_state.R_imu_cam0 = self.state_server.imu_state.R_imu_cam0
        imu_state.t_cam0_imu = self.state_server.imu_state.t_cam0_imu
        self.state_server.imu_state = imu_state

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Reset the state covariance.
        self.reset_state_cov()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Clear the IMU msg buffer.
        self.imu_msg_buffer.clear()

        # Reset the starting flags.
        self.is_gravity_set = False
        self.is_first_img = True

    def online_reset(self):
        """
        Reset the system online if the uncertainty is too large.
        """
        # Never perform online reset if position std threshold is non-positive.
        if self.config.position_std_threshold <= 0:
            return

        # Check the uncertainty of positions to determine if 
        # the system can be reset.
        position_x_std = np.sqrt(self.state_server.state_cov[12, 12])
        position_y_std = np.sqrt(self.state_server.state_cov[13, 13])
        position_z_std = np.sqrt(self.state_server.state_cov[14, 14])

        if max(position_x_std, position_y_std, position_z_std 
            ) < self.config.position_std_threshold:
            return

        print('Start online reset...')

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Reset the state covariance.
        self.reset_state_cov()

    def publish(self, time):
        imu_state = self.state_server.imu_state
        print('+++publish:')
        print('   timestamp:', imu_state.timestamp)
        print('   orientation:', imu_state.orientation)
        print('   position:', imu_state.position)
        print('   velocity:', imu_state.velocity)
        print()
        
        T_i_w = Isometry3d(
            to_rotation(imu_state.orientation).T,
            imu_state.position)
        T_b_w = IMUState.T_imu_body * T_i_w * IMUState.T_imu_body.inverse()
        body_velocity = IMUState.T_imu_body.R @ imu_state.velocity

        R_w_c = imu_state.R_imu_cam0 @ T_i_w.R.T
        t_c_w = imu_state.position + T_i_w.R @ imu_state.t_cam0_imu
        T_c_w = Isometry3d(R_w_c.T, t_c_w)

        return namedtuple('vio_result', ['timestamp', 'pose', 'velocity', 'cam0_pose'])(
            time, T_b_w, body_velocity, T_c_w)