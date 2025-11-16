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
    print("Warning: 'evaluations.py' not found. Performance metrics will be skipped.")
    evaluation = None


def run_median_filter(video_path, chosen_video):
    # Find all jpg files in that folder
    start_time = time.time()
    search_path = os.path.join(video_path, "*.jpg")
    print(f"Searching for files in: {search_path}")
    all_frame_files = sorted(glob.glob(search_path))

    # Incase missing or error 
    if not all_frame_files:
        print(f"Error: No .jpg files found in path: {video_path}")
        exit()

    # pre process the frames based on the input 
    frame_files = frame_preprocessor.preprocess_frames(
        all_frame_files, 
        chosen_video
    )
    print(f"Found {len(all_frame_files)} images. Processing {len(frame_files)} frames.")

    # Read the first frame from the file list to get the shape
    frame1 = cv2.imread(frame_files[0])
    if frame1 is None:
        print(f"Error: Could not read first frame: {frame_files[0]}")
        exit()

    # Get the shape (height, width)
    img_height, img_width = frame1.shape[:2]

    # Initialize Parameters
    # This algorithm initializes its background model to first frame
    bg_model = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    
    foreground_threshold = 20   # Set the threshold for foreground detection... tested [10,20,30,40,50] .. increasing is decreasing all scroes

    # Pre-define 255 (white, foreground) and 0 (black, background)
    a_fore = np.uint8(255)
    b_back = np.uint8(0)

    # Create a 5x5 kernel for noise removal... 3x3 doestn capture noise and 7x7 removes car parts also 
    kernel = np.ones([5,5], np.uint8)

    #  Setup evaluation folder 
    temp_eval_folder = "./temp_eval_median"
    if evaluation:
        if os.path.exists(temp_eval_folder):
            shutil.rmtree(temp_eval_folder)
        os.makedirs(temp_eval_folder, exist_ok=True)

    foreground_activity = []

    # This loop now iterates over every file  
    for frame_path in frame_files[1:]:
        # Read the frame from the file
        frame1 = cv2.imread(frame_path)
        if frame1 is None:
            print(f"Skipping bad frame: {frame_path}")
            continue
        
        # Convert the color frame to grayscale
        frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        
        # --- Median Filter Algorithm ---
        # If the pixel is brighter than the model, increment the model
        # If it's darker, decrement the model. Over time, it settles.

        # 1. Update the background model
        learning_rate = 1 # background model can only change by one brightness value per frame.
        # very slow .. good for static(avoid ghosting), bad for dynamic
        bg_model = np.where(frame > bg_model, bg_model + learning_rate, bg_model - learning_rate)
        
        # 2. Segment the foreground
        pixel_diff = cv2.absdiff(bg_model, frame)
        foreground_mask = np.where(
            pixel_diff > foreground_threshold, 
            a_fore, 
            b_back
        )
        
        # Remove noise
        foreground_mask = cv2.erode(foreground_mask, kernel)
        foreground_mask = cv2.dilate(foreground_mask, kernel)

        total_pixels = foreground_mask.size
        white_pixels = np.count_nonzero(foreground_mask)
        foreground_activity.append((white_pixels / total_pixels) * 100)

        # Save frame for evaluation 
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
        
        # Create the background image for display (No conversion needed, as bg_model is already uint8)
        background_display = bg_model
        
        # Show all three windows
        cv2.imshow('Original Video', frame1)
        cv2.imshow('Background', background_display) # Standardized title
        cv2.imshow('Foreground', foreground_colored) # Standardized title
        
        # If 'ESC' or 'q' is pressed, break
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            print("User pressed 'Esc' or 'q'. Exiting...")
            break
    cv2.destroyAllWindows()

    # Create an output folder to store plot if it doesn't exist
    plot_output_dir = f"./evaluation_results/MedianFilter/{chosen_video}"
    os.makedirs(plot_output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(foreground_activity, color='orange', linewidth=1) # Orange for Median
    plt.xlabel("Frame Number")
    plt.ylabel("Foreground Activity (%)")
    plt.title(f"Median Filter: Motion Activity over Time ({chosen_video})")
    plt.grid(True)
    
    # Save plot 
    plt.savefig(os.path.join(plot_output_dir, "activity_plot.png"))
    print(f"Saved activity plot to {plot_output_dir}")

    # Evaluation 
    end_time = time.time() 
    total_time = end_time - start_time
    print(f"Execution Time: {total_time:.2f} seconds")

    if evaluation:
        print("\n--- Starting Evaluation ---")
        gt_folder = video_path.replace("input", "groundtruth")
        
        if os.path.exists(gt_folder):
            f1, iou = evaluation.evaluate_performance(temp_eval_folder, gt_folder, chosen_video, "MedianFilter")
            evaluation.log_results_to_file("Median Filter (Easy)", f1, iou, total_time)
            shutil.rmtree(temp_eval_folder)    # Clean up 
            print("Cleaned up temporary evaluation frames.")
        else:
            print(f" Skipping Evaluation: No ground truth folder found at {gt_folder}")

# This block allows us to run file in standalone mode 
if __name__ == "__main__":
    print("Running median_filter.py in standalone mode for testing.")
    default_test_path = "../dataset/baseline/highway/input" 
    run_median_filter(default_test_path, "highway")