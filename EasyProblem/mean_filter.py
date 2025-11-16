import shutil       # To delete temp files
import cv2          # For reading images
import numpy as np  # For stacking frames
import os           # For building file paths
import glob         # To find all the image files
import time         # To track execution time
import matplotlib.pyplot as plt 
import frame_preprocessor

# Try to import the evaluation module
try:
    import evaluation
except ImportError:
    print("Warning: 'evaluation.py' not found. Performance metrics will be skipped.")
    evaluation = None


def run_mean_filter(video_path, chosen_video):
    # Find all jpg files in that folder
    start_time = time.time()
    search_path = os.path.join(video_path, "*.jpg")
    print(f"Searching for files in: {search_path}")
    all_frame_files = sorted(glob.glob(search_path))

    # Incase missing or error 
    if not all_frame_files:
        print(f"Error: No .jpg files found in path: {video_path}")
        exit()

    # pre process the frames based on the chosen video 
    frame_files = frame_preprocessor.preprocess_frames(
        all_frame_files, 
        chosen_video
    )
    print(f"Found {len(all_frame_files)} images. Processing {len(frame_files)} frames.")

    # Read the first frame from the file list to get the shape of the data 
    frame1 = cv2.imread(frame_files[0])
    if frame1 is None:
        print(f"Error: Could not read first frame: {frame_files[0]}")
        exit()

    # Get the shape (height, width)
    img_height, img_width = frame1.shape[:2]

    # Convert the first frame to grayscale to initialize the mean
    bg_mean = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Set a standard starting variance
    initial_variance = 150.0
    var = np.ones([img_height, img_width], np.float64) * initial_variance
    bg_variance = np.full([img_height, img_width], initial_variance, dtype=np.float64)

    # Initialize Parameters
    learning_rate = 0.1          # Learning rate for the running average, after testing a few this is best (bw 0.1 and 0.3)
    rho = 1.0 - learning_rate    # This is the weight given to old background while calculating new background
    foreground_threshold = 3.0   # after testing a few avlues, this is best 
    EPSILON = 1e-6               # We add 1e-6 to prevent division by zero


    # Pre-define 255 (white, foreground) and 0 (black, background)
    a_fore = np.uint8(255)
    b_back = np.uint8(0)

    # Create a 3x3 kernel for noise removal
    kernel = np.ones([3,3], np.uint8) # TESTED [3,3], [5,5] AND [7,7]
    # it is better to keep [3,3] as some cateories have lot of noise 
    # Setup the evaluation folder 
    temp_eval_folder = "./temp_eval_mean"
    if evaluation:
        if os.path.exists(temp_eval_folder):
            shutil.rmtree(temp_eval_folder)
        os.makedirs(temp_eval_folder, exist_ok=True)
    
    foreground_activity = []

    # This loop now iterates over every file
    for frame_path in frame_files:
        # Read the frame from the file
        frame1 = cv2.imread(frame_path)
        if frame1 is None:
            print(f"Skipping bad frame: {frame_path}")
            continue
        
        # Convert the color frame to grayscale
        frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        # Convert to float for mathematical operations
        frame_float = frame.astype(np.float64)
        
        # --- Mean filter Algorithm ---
        # 1. Calculate new mean and variance
        new_mean = cv2.addWeighted(
            bg_mean,            # src1: old background mean 
            rho,                # alpha (weight for src1)
            frame_float,        # src2: new information.. current new frame
            learning_rate,      # beta (weight for src2)
            0.0                 # gamma (a constant added at the end), don't need to add any extra brightness offset
        )

        # squared difference between the new frame and the old mean
        pixel_diff = cv2.absdiff(frame_float, bg_mean)
        pixel_diff_squared = cv2.pow(pixel_diff, 2)
        
        new_var = cv2.addWeighted(
            bg_variance,            # history 
            rho,                    # history weight
            pixel_diff_squared,     # new variance info .. observed variance in this one frame
            learning_rate,          # new info weight
            0.0
        )

        # 2. Calculate normalized difference (Z-score)
        # A simple pixel_diff (absolute difference) is unreliable.. 20 diff in trees is fine but on road unusual and is prolly a car 
        std_dev = cv2.sqrt(bg_variance + EPSILON)
        # How many standard deviations away from the normal mean is this pixel
        normalized_diff = pixel_diff / std_dev

        # 3. Update the background model (mean and var)
        # Create a boolean mask where 'True' means it's background
        is_background = (normalized_diff < foreground_threshold) 
    
        # For all pixels where 'is_background' is True, we update the bg_mean to the new_mean and update the bg_variance to the new_var
        # Only learn from pixels that I am confident are part of the true background
        # prevents model contamination/ghosting .. car will get false and their pixel value wont be used to update... 
        # so model will freeze that part of road remembering only underneath the car 
        # else car color will slowly absorb into bg_mean and ghost of car in background happens 
        bg_mean[is_background] = new_mean[is_background]
        bg_variance[is_background] = new_var[is_background]

        # 4. Segment the foreground
        # Mark pixels that are not background as foreground 
        foreground_mask = np.where(normalized_diff >= foreground_threshold, a_fore, b_back)
        
        # Remove noise
        # 1. Shrink all white areas 
        # Small, isolated white specks (noise) will be shrunk down to nothing and disappear. 
        # Larger, real objects (like a car) will also shrink, but they will still be present (just slightly smaller)
        foreground_mask = cv2.erode(foreground_mask, kernel)
        # 2. expand all white areas 
        # It grows the remaining white objects (the real ones) back to their original size. 
        #  Since the small noise specks were already gone, they can't be grown back.
        foreground_mask = cv2.dilate(foreground_mask, kernel)

        # Calculate percentage of image that is Foreground
        total_pixels = foreground_mask.size
        white_pixels = np.count_nonzero(foreground_mask)
        activity_ratio = (white_pixels / total_pixels) * 100
        foreground_activity.append(activity_ratio)

        # Save the frame for evaluation
        if evaluation:
            # We use the original filename but change extension to .png
            filename = os.path.basename(frame_path)
            name, _ = os.path.splitext(filename)
            save_path = os.path.join(temp_eval_folder, f"{name}.png")
            cv2.imwrite(save_path, foreground_mask)

        # Create the colored foreground for display
        foreground_colored = cv2.bitwise_and(
            frame1, 
            frame1, 
            mask=foreground_mask
        )
        
        # Create the background image for display (convert mean back to 8-bit)
        background_display = bg_mean.astype(np.uint8)
        
        # Show all three windows
        cv2.imshow('Original Video', frame1)
        cv2.imshow('Background', background_display)
        cv2.imshow('Foreground', foreground_colored) 
        
        # If 'ESC' is pressed, break
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            print("User pressed 'Esc' or 'q'. Exiting...")
            break
    cv2.destroyAllWindows()

    # Create an output folder to store plot if it doesn't exist
    plot_output_dir = f"./evaluation_results/MeanFilter/{chosen_video}"
    os.makedirs(plot_output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(foreground_activity, color='blue', linewidth=1)
    plt.xlabel("Frame Number")
    plt.ylabel("Foreground Activity (%)")
    plt.title(f"Mean Filter: Motion Activity over Time ({chosen_video})")
    plt.grid(True)
    
    # Save plot 
    plot_filename = os.path.join(plot_output_dir, "activity_plot.png")
    plt.savefig(plot_filename)
    print(f"Saved activity plot to {plot_filename}")

    # Evaluation 
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Execution Time: {total_time:.2f} seconds")

    if evaluation:
        print("\n--- Starting Evaluation ---")
        gt_folder = video_path.replace("input", "groundtruth")
        
        if os.path.exists(gt_folder):
            f1, iou = evaluation.evaluate_performance(temp_eval_folder, gt_folder, chosen_video, "MeanFilter")
            evaluation.log_results_to_file("Mean Filter (Easy)", f1, iou, total_time)
            shutil.rmtree(temp_eval_folder)    # Clean up
            print("Cleaned up temporary evaluation frames.")
        else:
            print(f"Skipping Evaluation: No ground truth folder found at {gt_folder}")

# This block allows us to run file in standalone mode 
if __name__ == "__main__":
    print("Running mean_filter.py in standalone mode for testing.")
    # Use a default path when running directly
    default_test_path = "../dataset/baseline/highway/input" 
    run_mean_filter(default_test_path, "highway")