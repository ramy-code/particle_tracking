import tifffile
import cv2
import numpy as np

def save_as_tiff(colored_sequence, filename):
    # Convert each colored frame to a proper format if it's grayscale
    rgb_frames = []
    for frame in colored_sequence:
        if frame.ndim == 2:  # Grayscale case
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb_frame = frame.astype(np.uint8)
        
        rgb_frames.append(rgb_frame)

    # Stack frames to create a 4D array for saving as a TIFF
    rgb_array = np.stack(rgb_frames, axis=0)  # Shape: (num_frames, height, width, 3)

    # Use tifffile to save the array as a TIFF file
    tifffile.imwrite(filename, rgb_array, photometric='rgb')


def calculate_metrics(trajectories, method_name):
    lengths = [len(traj) for traj in trajectories]
    avg_length = np.mean(lengths)
    
    # Calculate the number of trajectories
    num_trajectories = len(trajectories)
    
    # Calculate the average displacement per frame
    displacements = []
    for traj in trajectories:
        # Calculate the Euclidean distance between each consecutive pair of points in a trajectory
        displacement = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
        displacements.extend(displacement)
    
    avg_displacement = np.mean(displacements)
    
    # Print the results
    print(f"Method: {method_name}")
    print(f"Number of Trajectories: {num_trajectories}")
    print(f"Average Length of Trajectories: {avg_length:.2f} frames")
    print(f"Average Displacement per Frame: {avg_displacement:.2f} pixels")
    print("=" * 40)
    
    return {
        "method": method_name,
        "num_trajectories": num_trajectories,
        "avg_length": avg_length,
        "avg_displacement": avg_displacement
    }

def save_as_video(colored_sequence, filename, fps=30):
    # Check if the colored_sequence is not empty
    if not colored_sequence:
        raise ValueError("The colored sequence is empty.")

    # Get the dimensions from the first frame
    height, width, _ = colored_sequence[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use 'XVID' or 'MP4V' or other codecs
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in colored_sequence:
        out.write(frame)

    # Release the VideoWriter object
    out.release()
    print(f"Video saved as: {filename}")