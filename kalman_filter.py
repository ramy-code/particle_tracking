import numpy as np
from scipy.ndimage import label, center_of_mass
import cv2

class KalmanFilter:
    def __init__(self, initial_position, dt=1, process_noise=1, measurement_noise=1):
        self.state = np.array([initial_position[0], initial_position[1], 0, 0])
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.Q = process_noise * np.eye(4)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = measurement_noise * np.eye(2)
        self.P = np.eye(4)

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

    def update(self, measurement):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = measurement - np.dot(self.H, self.state)
        self.state += np.dot(K, y)
        I = np.eye(self.H.shape[1])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_position(self):
        return self.state[:2]


def track_with_kalman_filter(frames,threshold=50,max_distance=10):
    def detect_particles(frame):
        mask = frame > threshold
        labeled_array, num_features = label(mask)
        return np.array(center_of_mass(frame, labeled_array, range(1, num_features + 1)))

    particle_positions = [detect_particles(frame) for frame in frames]
    initial_positions = particle_positions[0]
    kalman_filters = [KalmanFilter(pos) for pos in initial_positions]

    trajectories = [[] for _ in kalman_filters]

    for frame_positions in particle_positions[1:]:
        for i, kf in enumerate(kalman_filters):
            kf.predict()
            if len(frame_positions) > 0:
                distances = np.linalg.norm(frame_positions - kf.get_position(), axis=1)
                min_idx = np.argmin(distances)
                if distances[min_idx] < max_distance:
                    kf.update(frame_positions[min_idx])
                    frame_positions = np.delete(frame_positions, min_idx, axis=0)
            trajectories[i].append(kf.get_position())

    return [np.array(traj) for traj in trajectories]

# Parameters for handling particle disappearance
max_missed_frames = 5  # Number of frames to wait before discarding a particle

# Initialize Kalman filter function
def initialize_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)  # 4 dynamic params, 2 measured params
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kf

def track_with_modified_kalman(frames, threshold=50, max_distance=15):
    trajectories = []  # To store active trajectories
    kalman_filters = []  # Kalman filters for each active particle
    missed_frames = []  # Track missed frames for each particle

    for frame_idx, frame in enumerate(frames):
        # Detect particles (using your detection method)
        mask = frame > threshold
        labeled_array, num_features = label(mask)
        detected_positions = np.array(center_of_mass(frame, labeled_array, range(1, num_features + 1)))

        # Update or add particles
        for i, (kf, traj, misses) in enumerate(zip(kalman_filters, trajectories, missed_frames)):
            if len(detected_positions) > 0:
                # Find nearest detected position to the Kalman-predicted position
                pred_position = kf.predict()[:2].flatten()
                distances = np.linalg.norm(detected_positions - pred_position, axis=1)
                nearest_idx = np.argmin(distances)
                nearest_distance = distances[nearest_idx]

                # If the closest position is within max_distance, update the Kalman filter
                if nearest_distance < max_distance:
                    measurement = np.array(detected_positions[nearest_idx], np.float32)
                    kf.correct(measurement)  # Update Kalman filter with the measurement
                    traj.append(measurement)  # Update trajectory
                    missed_frames[i] = 0  # Reset missed frame count

                    # Remove the used position
                    detected_positions = np.delete(detected_positions, nearest_idx, axis=0)
                else:
                    missed_frames[i] += 1  # Increment missed frames if no match found
            else:
                missed_frames[i] += 1

        # Remove particles that have exceeded the max missed frames
        active_particles = [(kf, traj, misses) for kf, traj, misses in zip(kalman_filters, trajectories, missed_frames) if misses <= max_missed_frames]
        kalman_filters, trajectories, missed_frames = map(list, zip(*active_particles)) if active_particles else ([], [], [])

        # Initialize Kalman filter for new particles that appear in this frame
        for new_position in detected_positions:
            kf = initialize_kalman_filter()
            kf.statePre[:2] = np.array([[new_position[0]], [new_position[1]]], np.float32)
            trajectories.append([new_position])  # Start a new trajectory with the new position
            kalman_filters.append(kf)
            missed_frames.append(0)  # Initialize missed frame count

    return trajectories