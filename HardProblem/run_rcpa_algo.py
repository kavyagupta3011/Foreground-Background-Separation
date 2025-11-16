import shutil       # To delete temp files
import cv2          # For reading images
import numpy as np  # For stacking frames
import os           # For building file paths
import glob         # To find all the image files
import time         # To track execution time
import matplotlib.pyplot as plt 
import frame_preprocessor

try:
    import make_video
    import RPCA  
    import evaluation 
except ImportError:
    print("Error: Could not import 'make_video.py' or 'RPCA.py' or 'evaluation.py'.")
    exit()


def analyze_results(X, L, S, rank):    
    # 1. Reconstruction Error (Frobenius Norm)
    # How close is (L + S) to the original X?
    reconstruction_error = np.linalg.norm(X - (L + S), 'fro') / np.linalg.norm(X, 'fro')
    print(f"1. Reconstruction Error (Relative): {reconstruction_error:.2e}")
    if reconstruction_error < 1e-4:
        print(" Mathematical decomposition holds (M approx L + S).")
    else:
        print(" WARNING: High reconstruction error. Algorithm may not have converged.")

    # 2. Sparsity Analysis
    # How many pixels are considered 'Foreground'?
    total_pixels = S.size
    # We count pixels that are not essentially zero
    nonzero_pixels = np.count_nonzero(np.abs(S) > 5.0)
    sparsity_ratio = nonzero_pixels / total_pixels
    print(f"2. Foreground Sparsity: {sparsity_ratio:.2%}")
    
    # 3. Low-Rank Analysis
    print(f"3. Background Rank: {rank}")
    if rank < 5:
        print(" Background is extremely low-rank (stable).")
    else:
        print(" NOTE: Background rank is higher.")

def run_rpca(video_path, chosen_video, hyperparams):
    """
    Runs the RPCA algorithm on a folder of images.
    
    Args:
        video_path (str): Path to the folder containing input .jpg frames.
        chosen_video (str): Prefix for the output video files (e.g., "highway").
        hyperparams (dict): A dictionary containing all algorithm settings.
    """
    start_time = time.time()

    # Find all jpg files in that folder
    search_path = os.path.join(video_path, "*.jpg")
    print(f"Searching for files in: {search_path}")
    all_frame_files = sorted(glob.glob(search_path))

    # Incase missing or error 
    if not all_frame_files:
        print(f"Error: No .jpg files found in path: {video_path}")
        return # Use return, not exit()

    # Get files to be processed 
    processed_frame_paths = frame_preprocessor.preprocess_frames(
        all_frame_files, 
        chosen_video
    )
    print(f"Found {len(all_frame_files)} images. Processing {len(processed_frame_paths)} frames.")

    # This list will store the flattened numpy arrays
    frame_data_list = [] 
    
    # Read the first frame 
    frame1 = cv2.imread(processed_frame_paths[0], cv2.IMREAD_GRAYSCALE)
    if frame1 is None:
        print(f"Error: Could not read first frame {processed_frame_paths[0]}")
        return
    
    height, width = frame1.shape[:2]
    print(f"Detected video dimensions: ({height},{width})")

   
    #  Add frame1 data is list
    frame_data_list.append(frame1.flatten())

    # Loop over all frames
    for frame_path in processed_frame_paths[1:]:
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Skipping bad frame: {frame_path}")
            continue
        
        # add pixel data to list
        frame_data_list.append(frame.flatten())

    # Stack list that only contains pixel data 
    X = np.stack(frame_data_list)
    print(f"Data matrix X created with shape: {X.shape}")
    
    # setup outputs 
    output_dir = "./output_rpca" 
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")
    
    # Save the original video
    original_video_path = f'{output_dir}/{chosen_video}_original.mp4'
    make_video.make_video(X, height, width, output_path=original_video_path)

    # call rcpa algorithm
    print("Starting RPCA optimization...")
    L, S, r, metrics = RPCA.rpca(
        X, height, width, 
        hyperparams['lambda'],
        hyperparams['mu'], 
        hyperparams['max_iter'], 
        hyperparams['eps_primal'], 
        hyperparams['eps_dual'], 
        hyperparams['rho'], 
        hyperparams['initial_sv'], 
        hyperparams['max_mu'], 
        hyperparams['verbose'], 
        hyperparams['save_interval']
    )

    print("RPCA optimization complete.")

    end_time = time.time()   
    total_time = end_time - start_time
    
    # Plot Cost Function
    plt.figure(figsize=(10, 5))
    plt.yscale('log')
    plt.plot(metrics["iterations"], metrics["objective_cost"], color='green')
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Verification: Objective Function Minimization")
    plt.grid(True)
    plt.savefig(f'{output_dir}/{chosen_video}_objective_cost.png')
    print(f"Saved verification plot to {output_dir}")

    # save results 
    bg_video_path = f'{output_dir}/{chosen_video}_background.mp4'
    fg_video_path = f'{output_dir}/{chosen_video}_foreground.mp4'
    
    # Using the refactored function/parameter names
    make_video.make_video(L, height, width, output_path=bg_video_path)
    make_video.make_video(S, height, width, output_path=fg_video_path)

    print(" --- Evaluations ---")
    print(f"Execution Time: {total_time:.2f} seconds")
    print(f"Final Objective Value: {metrics['objective_cost'][-1]:.4f}")

    analyze_results(X,L,S,r)    
    gt_folder = video_path.replace("input", "groundtruth")

    if os.path.exists(gt_folder):
        # Save our generated Sparse matrix (S) as images to compare
        temp_gen_folder = "./temp_eval_frames"
        os.makedirs(temp_gen_folder, exist_ok=True)
        print()
        print("Saving frames for evaluation...")
        # S is shape (n_frames, height*width). We need to reshape to #D and save
        S_3d = S.reshape((S.shape[0], height, width))
        
        kernel_open = np.ones((5,5), np.uint8)  # 5x5 kernel for "salt" noise
        kernel_close = np.ones((13,13), np.uint8) # 9x9 kernel to fill holes
        min_area = 100  # Minimum pixel area to be considered a "real" object

        for i, frame_data in enumerate(S_3d):
            # Normalize to 0-255
            frame_norm = np.clip(np.abs(frame_data), 0, 255).astype(np.uint8)
            
            # Threshold to make it binary (Black/White) for comparison
            # Any pixel > 20 becomes white. This helps remove the weak noise.
            _, frame_binary = cv2.threshold(frame_norm, 20, 255, cv2.THRESH_BINARY)
            # 2. Opening (removes "salt" noise)
            mask_opened = cv2.morphologyEx(frame_binary, cv2.MORPH_OPEN, kernel_open)
            
            # 3. Closing (fills "pepper" holes inside objects)
            mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)

            # 4. Connected Components (removes all small noise blobs)
            clean_mask = np.zeros_like(mask_closed) # Start with a new black mask
            
            # Find all separate blobs
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed, 4, cv2.CV_32S)
            
            # Loop from 1 to skip the background (label 0)
            for j in range(1, num_labels):
                area = stats[j, cv2.CC_STAT_AREA]
                
                # If the blob is "big enough," keep it
                if area >= min_area:
                    clean_mask[labels == j] = 255
            
            # Save the final CLEAN mask
            cv2.imwrite(f"{temp_gen_folder}/frame_{i:04d}.png", clean_mask)

        # Run Evaluation
        f1, iou = evaluation.evaluate_performance(temp_gen_folder, gt_folder, chosen_video, "RPCA")
        evaluation.log_results_to_file("RPCA (Hard)", f1, iou, total_time)

        # Cleanup temp folder
        shutil.rmtree(temp_gen_folder)
        print("Cleaned up temporary evaluation frames.")
        return f1, iou, total_time 
    else:
        print(f"Skipping Evaluation: No ground truth folder found at {gt_folder}")
        return None, None, total_time
        
# This block allows you to run this file directly for testing
if __name__ == '__main__':
    print("Running rpca_algorithm.py in standalone mode for testing.")
    
    DEFAULT_HYPERPARAMS = {
        "lambda": None,
        "mu": None,
        "max_iter": 1000,
        "eps_primal": 1e-7,
        "eps_dual": 1e-5,
        "rho": 1.6,
        "initial_sv": 10,
        "max_mu": 1e6,
        "verbose": True,
        "save_interval": 5
    }
    
    default_test_path = "../dataset/baseline/highway/input" 
    default_video_name = "highway"
    
    print(f"Using test path: {default_test_path}")
    run_rpca(default_test_path, default_video_name, DEFAULT_HYPERPARAMS)
