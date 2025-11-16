import cv2
import numpy as np

def make_video(frame_matrix, frame_h, frame_w, output_path='video.mp4', frame_rate=30):
    """
    Saves a matrix of flattened frames into a grayscale video file.

    Args:
        frame_matrix (numpy.ndarray): 2D array, shape (num_images, height * width), containing the flattened frame data.
        frame_h (int): The height of a single frame.
        frame_w (int): The width of a single frame.
        output_path (str): The file path for the saved video (e.g., "./output.mp4").
        frame_rate (int): The frames per second for the output video.

    Returns:
        None.
    """
    
    # Get the total number of images from the matrix shape
    num_images = frame_matrix.shape[0]
    
    # Reshape back to video create a #D (n_frames, height, width) video tensor.
    video_tensor = frame_matrix.reshape((num_images, frame_h, frame_w))
    
    # Define the video codec
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Define the output video dimensions (width, height) for OpenCV
    video_dims = (frame_w, frame_h)

    # Initialize the video writing object
    video_writer = cv2.VideoWriter(
        output_path, 
        codec, 
        frame_rate, 
        video_dims, 
        isColor=False # We are writing grayscale frames
    )
    
    for current_frame in video_tensor:        
        # Normalize and convert to uint8 for video writing
        # This handles cases where S has negative values or L > 255
        # Clip values to be in valid 0-255 range
        writable_frame = current_frame.clip(0, 255).astype(np.uint8)
        
        # Write the processed frame to the video file
        video_writer.write(writable_frame)
    # Release the video writer to finalize the file
    video_writer.release()
    print(f"Video successfully saved to {output_path}")