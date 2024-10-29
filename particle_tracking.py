import numpy as np
from scipy.ndimage import label, center_of_mass
import trackpy as tp
import pandas as pd 

def track_with_nearest_neighbor(frames,threshold=50,max_distance=10):
    def detect_particles(frame):
        mask = frame > threshold
        labeled_array, num_features = label(mask)
        return np.array(center_of_mass(frame, labeled_array, range(1, num_features + 1)))

    particle_positions = [detect_particles(frame) for frame in frames]
    
    # Initialize trajectories for each detected particle in the first frame
    initial_positions = particle_positions[0]
    trajectories = [[pos] for pos in initial_positions]

    # Link particles across frames
    for frame_positions in particle_positions[1:]:
        new_trajectories = [[] for _ in trajectories]

        for i, traj in enumerate(trajectories):
            if len(traj) > 0:
                last_position = traj[-1]
                distances = np.linalg.norm(frame_positions - last_position, axis=1)
                if len(distances) > 0:
                    min_idx = np.argmin(distances)
                    # Link if the nearest particle is within max_distance
                    if distances[min_idx] < max_distance:
                        new_trajectories[i] = traj + [frame_positions[min_idx]]
                        frame_positions = np.delete(frame_positions, min_idx, axis=0)
                    else:
                        new_trajectories[i] = traj
                else:
                    new_trajectories[i] = traj

        for pos in frame_positions:
            new_trajectories.append([pos])

        trajectories = new_trajectories

    return [np.array(traj) for traj in trajectories]

def track_with_trackpy(frames,threshold=50,max_distance=10):
    particles = []
    for frame_idx, frame in enumerate(frames):
        features = tp.locate(frame, diameter=11, minmass=100, threshold=threshold)
        features['frame'] = frame_idx
        particles.append(features)
    particles_df = pd.concat(particles)
    
    # Track particles using trackpy's linking function
    trajectories = tp.link_df(particles_df, search_range=max_distance)
    
    # Convert trackpy's format to a list of trajectories (each particle's positions)
    trackpy_trajectories = []
    for particle_id in trajectories['particle'].unique():
        particle_data = trajectories[trajectories['particle'] == particle_id]
        trackpy_trajectories.append(particle_data[['x', 'y']].values)
    
    return trackpy_trajectories

